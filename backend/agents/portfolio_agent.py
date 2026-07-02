"""
Agente de Análisis de Portafolio

Coordina los agentes existentes para analizar un conjunto de activos
y calcula métricas de portafolio según la teoría moderna de carteras:
- Retorno esperado y volatilidad anualizada
- Ratio de Sharpe
- Value at Risk (VaR) paramétrico
- Matriz de correlación
- Optimización de Markowitz: mínima varianza y máximo Sharpe
- Frontera eficiente
"""
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    from scipy.optimize import minimize
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy no disponible — optimización de Markowitz desactivada")

TRADING_DAYS = 252
RISK_FREE_RATE_PCT = 4.5   # 4.5% anual en %


@dataclass
class AssetAnalysis:
    ticker: str
    weight: float
    market: object
    prediction: object
    sentiment: object
    recommendation: object
    sec_data: object
    expected_return: float      # % anualizado (blend histórico/ML ingenuo — solo comparación)
    volatility: float           # % anualizado
    price: float
    # Black-Litterman (poblados en analizar_portafolio tras _calcular_black_litterman)
    bl_prior: Optional[float] = None          # Π_i, equilibrio de mercado, % anualizado
    bl_view: Optional[float] = None           # Q_i, view combinada de agentes, % anualizado
    bl_posterior: Optional[float] = None      # μ_BL_i, retorno usado por el optimizador
    view_confidence: Optional[float] = None   # confianza combinada usada para Ω_ii


@dataclass
class PortfolioMetrics:
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    diversification_ratio: float
    correlation_matrix: Dict[str, Dict[str, float]]
    num_activos: int
    beta_portfolio: Optional[float] = None
    max_drawdown: Optional[float] = None


@dataclass
class EfficientFrontierPoint:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float


@dataclass
class PortfolioOptimization:
    max_sharpe_weights: Dict[str, float]
    max_sharpe_return: float
    max_sharpe_volatility: float
    max_sharpe_sharpe: float
    min_variance_weights: Dict[str, float]
    min_variance_return: float
    min_variance_volatility: float
    efficient_frontier: List[EfficientFrontierPoint]
    disponible: bool = True
    # Hierarchical Risk Parity (López de Prado, 2016)
    hrp_weights: Dict[str, float] = field(default_factory=dict)
    hrp_return: float = 0.0
    hrp_volatility: float = 0.0
    hrp_sharpe: float = 0.0
    # Black-Litterman — transparencia de los parámetros usados por el optimizador
    delta_aversion: float = 0.0
    tau: float = 0.0
    fuente_delta: str = ""


@dataclass
class BlackLittermanResult:
    """
    Retorno posterior de Black-Litterman: combina el equilibrio de mercado (Π)
    con "views" derivadas de ModelAgent/SentimentAgent/SECAgent (Q), ponderadas
    por su confianza (Ω). Referencias: He & Litterman (1999), Idzorek (2005).
    """
    tickers: List[str]
    pi_prior: Dict[str, float] = field(default_factory=dict)
    q_view: Dict[str, float] = field(default_factory=dict)
    view_confidence: Dict[str, float] = field(default_factory=dict)
    mu_posterior: Dict[str, float] = field(default_factory=dict)
    delta_aversion: float = 0.0
    delta_mercado_referencia: Optional[float] = None
    tau: float = 0.05
    w_mkt: Dict[str, float] = field(default_factory=dict)
    fuente_delta: str = ""            # "perfil" | "mercado"
    disponible: bool = True


@dataclass
class PortfolioResult:
    activos: List[AssetAnalysis]
    metricas: PortfolioMetrics
    optimizacion: PortfolioOptimization
    recomendacion_portafolio: str
    alertas: List[dict]
    tickers: List[str]
    weights: Dict[str, float]
    fecha_analisis: datetime
    black_litterman: Optional[BlackLittermanResult] = None


class PortfolioAgent:
    """
    Agente de análisis y optimización de portafolios.

    Orquesta MarketAgent, ModelAgent, SentimentAgent,
    RecommendationAgent y SECAgent para cada activo,
    luego aplica análisis de portafolio media-varianza.
    """

    def __init__(
        self,
        market_agent,
        model_agent,
        sentiment_agent,
        recommendation_agent,
        sec_agent,
        alert_agent,
    ):
        self.market_agent = market_agent
        self.model_agent = model_agent
        self.sentiment_agent = sentiment_agent
        self.recommendation_agent = recommendation_agent
        self.sec_agent = sec_agent
        self.alert_agent = alert_agent
        logger.info("PortfolioAgent inicializado")

    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------

    def analizar_portafolio(
        self,
        tickers: List[str],
        weights: List[float],
        forzar_actualizacion: bool = False,
        max_weight: float = 0.65,
        delta_aversion: Optional[float] = None,
    ) -> PortfolioResult:
        """
        Analiza un portafolio completo.

        Args:
            tickers: Símbolos de los activos
            weights: Pesos (se normalizan a suma=1)
            forzar_actualizacion: Ignorar caché de agentes individuales
            delta_aversion: Coeficiente de aversión al riesgo (δ) del perfil del
                inversor, para Black-Litterman. Si es None, se estima a partir
                del mercado (^GSPC, fórmula de aversión implícita de He & Litterman).

        Returns:
            PortfolioResult con análisis completo y optimización
        """
        if len(tickers) != len(weights):
            raise ValueError("tickers y weights deben tener la misma longitud")
        if len(tickers) < 2:
            raise ValueError("Se requieren al menos 2 activos para análisis de portafolio")

        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("Los pesos deben ser positivos")
        weights_norm = [w / total_w for w in weights]
        tickers = [t.upper().strip() for t in tickers]
        weights_dict = dict(zip(tickers, weights_norm))

        # Análisis individual en paralelo
        activos = self._analizar_activos_paralelo(tickers, weights_dict, forzar_actualizacion)
        if len(activos) < 2:
            raise ValueError("Se necesitan al menos 2 activos con datos válidos")

        # Recalcular pesos normalizados con los activos que sí se pudieron analizar
        tickers_ok = [a.ticker for a in activos]
        raw_w = [weights_dict[t] for t in tickers_ok]
        total_ok = sum(raw_w)
        for a in activos:
            a.weight = weights_dict[a.ticker] / total_ok

        # Matriz de retornos calculada una sola vez y compartida por BL,
        # métricas y optimización (evita descargas repetidas de yfinance)
        returns_mat = self._get_returns_matrix(tickers_ok)

        bl = self._calcular_black_litterman(activos, tickers_ok, returns_mat, delta_aversion)
        for a in activos:
            a.bl_prior = bl.pi_prior.get(a.ticker)
            a.bl_view = bl.q_view.get(a.ticker)
            a.bl_posterior = bl.mu_posterior.get(a.ticker)
            a.view_confidence = bl.view_confidence.get(a.ticker)

        metricas = self._calcular_metricas(activos, returns_mat=returns_mat)
        optimizacion = self._optimizar_markowitz(
            activos, max_weight=max_weight, returns_mat=returns_mat, bl=bl,
        )
        recomendacion = self._generar_recomendacion(activos, metricas, optimizacion)
        alertas = self._detectar_alertas(activos, metricas)

        return PortfolioResult(
            activos=activos,
            metricas=metricas,
            optimizacion=optimizacion,
            recomendacion_portafolio=recomendacion,
            alertas=alertas,
            tickers=tickers_ok,
            weights={a.ticker: round(a.weight, 4) for a in activos},
            fecha_analisis=datetime.now(),
            black_litterman=bl,
        )

    # ------------------------------------------------------------------
    # Análisis individual por activo
    # ------------------------------------------------------------------

    def _analizar_activos_paralelo(
        self, tickers: List[str], weights: Dict[str, float], forzar: bool
    ) -> List[AssetAnalysis]:
        activos = []
        with ThreadPoolExecutor(max_workers=min(len(tickers), 4)) as executor:
            futures = {
                executor.submit(self._analizar_activo, t, weights[t], forzar): t
                for t in tickers
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    activos.append(fut.result(timeout=120))
                except Exception as exc:
                    logger.error(f"[{t}] Fallo en análisis individual: {exc}")

        # Preservar orden original
        order = {t: i for i, t in enumerate(tickers)}
        activos.sort(key=lambda a: order.get(a.ticker, 999))
        return activos

    def _analizar_activo(self, ticker: str, weight: float, forzar: bool) -> AssetAnalysis:
        market_data = self.market_agent.obtener_datos(ticker, forzar)
        if market_data is None:
            raise ValueError(f"Ticker '{ticker}' no encontrado en Yahoo Finance")

        prediction = self.model_agent.predecir(
            market_data.precios, ticker, forzar_actualizacion=forzar
        )
        sentiment = self.sentiment_agent.analizar(ticker)
        sec_data = self.sec_agent.analizar(ticker)

        variacion_pct = prediction.variacion_pct if prediction else 0.0
        pred_confidence = prediction.confianza if prediction else 0.5
        sentiment_score = getattr(sentiment, "score", 0.0)

        regime_raw = market_data.signal_analysis.market_regime
        regime_str = regime_raw.value if hasattr(regime_raw, "value") else str(regime_raw)
        atr = market_data.indicators.atr if market_data.indicators.atr > 0 else 2.0
        mfi = market_data.indicators.mfi

        recommendation = self.recommendation_agent.generar_recomendacion(
            ticker=ticker,
            senal_mercado=market_data.senal,
            variacion_pct=variacion_pct,
            sentimiento=sentiment.sentimiento,
            confianza_sentimiento=sentiment.confianza,
            ultimo_precio=market_data.ultimo_precio,
            volatilidad=atr,
            market_regime=regime_str,
            rsi=market_data.indicators.rsi,
            macd_signal=market_data.indicators.macd_signal,
            volume_ratio=mfi / 50.0 if mfi > 0 else 1.0,
            prediction_confidence=pred_confidence,
            sentiment_score=sentiment_score,
        )

        # Volatilidad y retorno esperado anualizados
        closes = market_data.precios["Close"].dropna()
        daily_ret = closes.pct_change().dropna()

        if len(daily_ret) > 20:
            annual_vol = float(daily_ret.std() * np.sqrt(TRADING_DAYS) * 100)
            hist_return = float(daily_ret.mean() * TRADING_DAYS * 100)
        else:
            annual_vol = 20.0
            hist_return = 0.0

        # Retorno esperado ingenuo (blend histórico/ML) — se muestra solo como
        # referencia de comparación; el optimizador usa el posterior de
        # Black-Litterman (`AssetAnalysis.bl_posterior`, calculado en
        # `analizar_portafolio` tras reunir todos los activos).
        pred_annual = variacion_pct * (TRADING_DAYS / 3)
        expected_return = hist_return * 0.7 + pred_annual * 0.3

        return AssetAnalysis(
            ticker=ticker,
            weight=weight,
            market=market_data,
            prediction=prediction,
            sentiment=sentiment,
            recommendation=recommendation,
            sec_data=sec_data,
            expected_return=round(expected_return, 3),
            volatility=round(max(annual_vol, 1.0), 3),
            price=market_data.ultimo_precio,
        )

    # ------------------------------------------------------------------
    # Métricas de portafolio
    # ------------------------------------------------------------------

    def _calcular_metricas(
        self, activos: List[AssetAnalysis], returns_mat: Optional[np.ndarray] = None,
    ) -> PortfolioMetrics:
        tickers = [a.ticker for a in activos]
        w = np.array([a.weight for a in activos])
        # Retorno esperado: posterior de Black-Litterman si está disponible,
        # si no cae al blend histórico/ML ingenuo (ver _analizar_activo).
        mu = np.array([
            a.bl_posterior if a.bl_posterior is not None else a.expected_return
            for a in activos
        ])
        vols = np.array([a.volatility for a in activos])

        if returns_mat is None:
            returns_mat = self._get_returns_matrix(tickers)

        if returns_mat is not None and returns_mat.shape[1] == len(tickers):
            cov = np.cov(returns_mat.T) * TRADING_DAYS * (100 ** 2)
            port_var = float(w @ cov @ w)
            corr = np.corrcoef(returns_mat.T)
        else:
            # Fallback: correlación 0.3 entre pares distintos
            corr = np.full((len(tickers), len(tickers)), 0.3)
            np.fill_diagonal(corr, 1.0)
            cov = np.outer(vols, vols) * corr
            port_var = float(w @ cov @ w)

        port_vol = float(np.sqrt(max(port_var, 0.01)))
        port_ret = float(w @ mu)
        sharpe = (port_ret - RISK_FREE_RATE_PCT) / port_vol if port_vol > 0 else 0.0

        z95, z99 = 1.645, 2.326
        var_95 = port_ret - z95 * port_vol
        var_99 = port_ret - z99 * port_vol

        weighted_vol = float(w @ vols)
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1.0

        corr_dict: Dict[str, Dict[str, float]] = {}
        for i, ti in enumerate(tickers):
            corr_dict[ti] = {}
            for j, tj in enumerate(tickers):
                corr_dict[ti][tj] = round(float(corr[i, j]), 3)

        beta = self._calcular_beta(tickers, w)

        # Maximum Drawdown histórico del portafolio
        if returns_mat is not None and returns_mat.shape[1] == len(tickers):
            port_rets_hist = returns_mat @ w
            cum = np.cumprod(1 + port_rets_hist)
            roll_max = np.maximum.accumulate(cum)
            mdd = round(float(np.min(cum / roll_max - 1) * 100), 2)
        else:
            mdd = None

        return PortfolioMetrics(
            expected_return=round(port_ret, 2),
            volatility=round(port_vol, 2),
            sharpe_ratio=round(sharpe, 3),
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            diversification_ratio=round(div_ratio, 3),
            correlation_matrix=corr_dict,
            num_activos=len(activos),
            beta_portfolio=beta,
            max_drawdown=mdd,
        )

    def _get_returns_matrix(self, tickers: List[str], periodo: str = "2y"):
        if not (YF_AVAILABLE and PANDAS_AVAILABLE):
            return None
        try:
            raw = yf.download(tickers, period=periodo, auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                return None

            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else None
            else:
                closes = raw[["Close"]] if len(tickers) == 1 else raw

            if closes is None or closes.empty:
                return None

            rets = closes.pct_change().dropna()
            if hasattr(rets, "columns"):
                available = [t for t in tickers if t in rets.columns]
                rets = rets[available].dropna()

            return rets.values if len(rets) > 30 else None
        except Exception as exc:
            logger.warning(f"Error obteniendo matriz de retornos: {exc}")
            return None

    def _calcular_beta(self, tickers: List[str], weights: np.ndarray) -> Optional[float]:
        if not (YF_AVAILABLE and PANDAS_AVAILABLE):
            return None
        try:
            raw = yf.download(tickers + ["^GSPC"], period="1y", auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                return None
            rets = closes.pct_change().dropna()
            if "^GSPC" not in rets.columns:
                return None
            mkt = rets["^GSPC"]
            mkt_var = float(mkt.var())
            if mkt_var == 0:
                return None
            beta_port = 0.0
            for t, w in zip(tickers, weights):
                if t in rets.columns:
                    beta_port += float(rets[t].cov(mkt) / mkt_var) * w
            return round(beta_port, 3)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Black-Litterman
    #
    # Combina el equilibrio de mercado (Π) con "views" derivadas de los
    # agentes ya calculados (ModelAgent, SentimentAgent, SECAgent) para
    # producir un retorno esperado posterior (μ_BL) que alimenta la
    # optimización de Markowitz — en vez de la media histórica cruda o de
    # un blend de pesos fijos inventado por el autor.
    #
    # Todo el cálculo se hace en retornos DECIMALES anualizados (convención
    # estándar de la literatura de Black-Litterman: He & Litterman, 1999;
    # Idzorek, 2005) y solo se convierte a porcentaje al final, para que δ
    # sea directamente comparable con los rangos citados en la literatura
    # (ej. Bodie/Kane/Marcus, coeficiente de aversión al riesgo ~2-10).
    # ------------------------------------------------------------------

    def _obtener_pesos_mercado(self, tickers: List[str]) -> Dict[str, float]:
        """
        Pesos de "mercado" del universo analizado: market cap (acciones) o
        total de activos administrados/AUM (ETFs), normalizados a suma=1.
        Es la práctica estándar en aplicaciones de Black-Litterman
        restringidas a un universo de inversión específico (Idzorek, 2005),
        ya que no existe un portafolio de mercado global observable para
        una canasta arbitraria de ETFs y acciones.

        Fallback: si no hay dato para un ticker, se le asigna el promedio
        de los tickers con dato conocido (degradación razonable, no una
        recomendación inventada).
        """
        n = len(tickers)
        if not YF_AVAILABLE:
            return {t: 1.0 / n for t in tickers}

        pesos_crudos: Dict[str, float] = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                cap = info.get("marketCap") or info.get("totalAssets")
                if cap:
                    pesos_crudos[t] = float(cap)
            except Exception as exc:
                logger.debug(f"[{t}] No se pudo obtener market cap/AUM: {exc}")

        faltantes = [t for t in tickers if t not in pesos_crudos]
        if faltantes:
            logger.warning(
                f"Pesos de mercado no disponibles para {faltantes} — "
                "se les asigna el promedio de los tickers con dato conocido"
            )

        if not pesos_crudos:
            return {t: 1.0 / n for t in tickers}

        promedio = sum(pesos_crudos.values()) / len(pesos_crudos)
        for t in faltantes:
            pesos_crudos[t] = promedio

        total = sum(pesos_crudos[t] for t in tickers)
        return {t: pesos_crudos[t] / total for t in tickers}

    def _calcular_delta_mercado(self, tickers: List[str], periodo: str = "2y") -> Optional[float]:
        """
        δ_mkt = (R_mkt anualizado − Rf) / Var(R_mkt anualizado), en retornos
        DECIMALES — la fórmula estándar de aversión al riesgo implícita por
        optimización inversa (He & Litterman, 1999), usando ^GSPC como proxy
        del mercado. Reutiliza el patrón de descarga de `_calcular_beta`.
        """
        if not (YF_AVAILABLE and PANDAS_AVAILABLE):
            return None
        try:
            raw = yf.download(tickers + ["^GSPC"], period=periodo, auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                return None
            rets = closes.pct_change().dropna()
            if "^GSPC" not in rets.columns:
                return None
            mkt = rets["^GSPC"].dropna()
            if len(mkt) < 30:
                return None
            mkt_ret_dec = float(mkt.mean() * TRADING_DAYS)
            mkt_var_dec = float(mkt.var() * TRADING_DAYS)
            if mkt_var_dec <= 0:
                return None
            rf_dec = RISK_FREE_RATE_PCT / 100.0
            delta = (mkt_ret_dec - rf_dec) / mkt_var_dec
            return round(max(delta, 0.5), 3)
        except Exception as exc:
            logger.warning(f"Error calculando delta de mercado: {exc}")
            return None

    def _construir_vista_activo(
        self, activo: AssetAnalysis, sigma_i_dec: float, pi_excess_dec_i: float,
    ) -> Tuple[float, float]:
        """
        Combina las señales ya calculadas de ModelAgent, SentimentAgent y
        SECAgent en una view de Black-Litterman para un activo.

        Returns:
            (Q_excess_dec_i, confidence_i) — view de exceso de retorno sobre
            la tasa libre de riesgo (decimal, anualizado) y confianza
            combinada (usada para construir Ω).
        """
        variacion_pct = activo.prediction.variacion_pct if activo.prediction else 0.0
        pred_annual_dec = (variacion_pct * (TRADING_DAYS / 3)) / 100.0
        s_model = math.tanh(pred_annual_dec / sigma_i_dec) if sigma_i_dec > 0 else 0.0

        s_sentiment = float(getattr(activo.sentiment, "score", 0.0) or 0.0)
        s_sec = float(getattr(activo.sec_data, "fundamental_score", 0.0) or 0.0)

        c_model = float(activo.prediction.confianza) if activo.prediction else 0.0
        c_sentiment = float(getattr(activo.sentiment, "confianza", 0.0) or 0.0)
        c_sec = min(abs(s_sec), 1.0)

        denom = c_model + c_sentiment + c_sec + 1e-6
        s_combined = (
            c_model * s_model + c_sentiment * s_sentiment + c_sec * s_sec
        ) / denom
        s_combined = max(-1.0, min(1.0, s_combined))

        q_excess_dec = pi_excess_dec_i + s_combined * sigma_i_dec
        confidence = c_model + c_sentiment + c_sec
        return q_excess_dec, confidence

    def _calcular_black_litterman(
        self,
        activos: List[AssetAnalysis],
        tickers: List[str],
        returns_mat: Optional[np.ndarray],
        delta_aversion: Optional[float],
        tau: float = 0.05,
    ) -> "BlackLittermanResult":
        n = len(tickers)
        delta_mkt = self._calcular_delta_mercado(tickers)

        def _fallback() -> "BlackLittermanResult":
            return BlackLittermanResult(
                tickers=tickers,
                mu_posterior={a.ticker: a.expected_return for a in activos},
                delta_aversion=delta_aversion or delta_mkt or 0.0,
                delta_mercado_referencia=delta_mkt,
                tau=tau,
                fuente_delta="perfil" if delta_aversion is not None else "mercado",
                disponible=False,
            )

        if returns_mat is None or returns_mat.shape[1] != n or n < 2:
            logger.warning("Black-Litterman: sin matriz de retornos válida — usando retorno ingenuo")
            return _fallback()

        try:
            sigma_dec = np.cov(returns_mat.T) * TRADING_DAYS      # decimal, anualizado
            sigma_i_dec = np.sqrt(np.maximum(np.diag(sigma_dec), 1e-10))
            rf_dec = RISK_FREE_RATE_PCT / 100.0

            w_mkt_dict = self._obtener_pesos_mercado(tickers)
            w_mkt = np.array([w_mkt_dict[t] for t in tickers])

            delta = delta_aversion if delta_aversion is not None else delta_mkt
            fuente = "perfil" if delta_aversion is not None else "mercado"
            if delta is None:
                logger.warning("Black-Litterman: no se pudo estimar δ — usando retorno ingenuo")
                return _fallback()

            pi_excess_dec = delta * (sigma_dec @ w_mkt)

            q_excess_dec = np.zeros(n)
            confidence = np.zeros(n)
            for i, activo in enumerate(activos):
                q_excess_dec[i], confidence[i] = self._construir_vista_activo(
                    activo, float(sigma_i_dec[i]), float(pi_excess_dec[i])
                )

            omega = np.diag(tau * np.diag(sigma_dec) / np.maximum(confidence, 1e-6))

            tau_sigma_inv = np.linalg.inv(tau * sigma_dec)
            omega_inv = np.linalg.inv(omega)
            posterior_cov_inv = tau_sigma_inv + omega_inv
            mu_bl_excess_dec = np.linalg.solve(
                posterior_cov_inv,
                tau_sigma_inv @ pi_excess_dec + omega_inv @ q_excess_dec,
            )

            pi_total_pct = (pi_excess_dec + rf_dec) * 100
            q_total_pct = (q_excess_dec + rf_dec) * 100
            mu_bl_total_pct = (mu_bl_excess_dec + rf_dec) * 100

            return BlackLittermanResult(
                tickers=tickers,
                pi_prior={t: round(float(pi_total_pct[i]), 3) for i, t in enumerate(tickers)},
                q_view={t: round(float(q_total_pct[i]), 3) for i, t in enumerate(tickers)},
                view_confidence={t: round(float(confidence[i]), 3) for i, t in enumerate(tickers)},
                mu_posterior={t: round(float(mu_bl_total_pct[i]), 3) for i, t in enumerate(tickers)},
                delta_aversion=round(float(delta), 4),
                delta_mercado_referencia=delta_mkt,
                tau=tau,
                w_mkt=w_mkt_dict,
                fuente_delta=fuente,
                disponible=True,
            )
        except Exception as exc:
            logger.warning(f"Black-Litterman falló ({exc}) — usando retorno ingenuo")
            return _fallback()

    # ------------------------------------------------------------------
    # Optimización de Markowitz
    # ------------------------------------------------------------------

    def _optimizar_markowitz(
        self,
        activos: List[AssetAnalysis],
        max_weight: float = 0.65,
        returns_mat: Optional[np.ndarray] = None,
        bl: Optional["BlackLittermanResult"] = None,
    ) -> PortfolioOptimization:
        tickers = [a.ticker for a in activos]
        n = len(tickers)
        equal_w = {t: round(1.0 / n, 4) for t in tickers}

        if not SCIPY_AVAILABLE:
            return PortfolioOptimization(
                max_sharpe_weights=equal_w, max_sharpe_return=0.0,
                max_sharpe_volatility=0.0, max_sharpe_sharpe=0.0,
                min_variance_weights=equal_w, min_variance_return=0.0,
                min_variance_volatility=0.0, efficient_frontier=[], disponible=False,
            )

        try:
            if returns_mat is None:
                returns_mat = self._get_returns_matrix(tickers)

            if returns_mat is not None and returns_mat.shape[1] == n:
                cov = np.cov(returns_mat.T) * TRADING_DAYS * (100 ** 2)
            else:
                vols = np.array([a.volatility for a in activos])
                corr = np.full((n, n), 0.3)
                np.fill_diagonal(corr, 1.0)
                cov = np.outer(vols, vols) * corr

            # Retorno esperado que alimenta el optimizador: posterior de
            # Black-Litterman (combina equilibrio de mercado + señales de
            # todos los agentes) cuando está disponible; si no, cae a la
            # media histórica cruda (modo degradado, logueado en
            # _calcular_black_litterman).
            if bl is not None and bl.disponible and bl.mu_posterior:
                mu = np.array([bl.mu_posterior[t] for t in tickers])
            elif returns_mat is not None and returns_mat.shape[1] == n:
                mu = np.mean(returns_mat, axis=0) * TRADING_DAYS * 100
            else:
                mu = np.array([a.expected_return for a in activos])

            rf = RISK_FREE_RATE_PCT
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bounds = [(0.01, max(0.01, min(max_weight, 0.99)))] * n
            w0 = np.ones(n) / n

            def port_ret(w): return float(w @ mu)
            def port_vol(w): return float(np.sqrt(max(w @ cov @ w, 1e-8)))
            def neg_sharpe(w):
                v = port_vol(w)
                return -(port_ret(w) - rf) / v if v > 0 else 0.0

            # Máximo Sharpe
            res_sharpe = minimize(
                neg_sharpe, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            w_sharpe = res_sharpe.x if res_sharpe.success else w0

            # Mínima varianza
            res_minvar = minimize(
                port_vol, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 1000},
            )
            w_minvar = res_minvar.x if res_minvar.success else w0

            # Frontera eficiente (15 puntos)
            frontier: List[EfficientFrontierPoint] = []
            r_min, r_max = float(mu.min()), float(mu.max())
            if r_max > r_min:
                for target_r in np.linspace(r_min, r_max, 15):
                    cons = constraints + [
                        {"type": "eq", "fun": lambda w, tr=target_r: port_ret(w) - tr}
                    ]
                    res = minimize(
                        port_vol, w0, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"ftol": 1e-9, "maxiter": 500},
                    )
                    if res.success:
                        pv = port_vol(res.x)
                        pr = port_ret(res.x)
                        ps = (pr - rf) / pv if pv > 0 else 0.0
                        frontier.append(EfficientFrontierPoint(
                            weights={t: round(float(res.x[i]), 4) for i, t in enumerate(tickers)},
                            expected_return=round(pr, 2),
                            volatility=round(pv, 2),
                            sharpe=round(ps, 3),
                        ))

            def _wdict(w):
                return {t: round(float(w[i]), 4) for i, t in enumerate(tickers)}

            # ── Hierarchical Risk Parity ──────────────────────────────────────
            hrp_w_dict: Dict[str, float] = {}
            hrp_ret = hrp_vol = hrp_sharpe_val = 0.0
            if returns_mat is not None and returns_mat.shape[1] == n:
                try:
                    hrp_w_dict = self._hrp_weights(returns_mat, tickers, max_weight=max_weight)
                    w_hrp = np.array([hrp_w_dict[t] for t in tickers])
                    hrp_ret  = round(port_ret(w_hrp), 2)
                    hrp_vol  = round(port_vol(w_hrp), 2)
                    hrp_sharpe_val = round((hrp_ret - rf) / hrp_vol, 3) if hrp_vol > 0 else 0.0
                except Exception as e:
                    logger.warning(f"HRP falló: {e}")

            return PortfolioOptimization(
                max_sharpe_weights=_wdict(w_sharpe),
                max_sharpe_return=round(port_ret(w_sharpe), 2),
                max_sharpe_volatility=round(port_vol(w_sharpe), 2),
                max_sharpe_sharpe=round(-neg_sharpe(w_sharpe), 3),
                min_variance_weights=_wdict(w_minvar),
                min_variance_return=round(port_ret(w_minvar), 2),
                min_variance_volatility=round(port_vol(w_minvar), 2),
                efficient_frontier=frontier,
                disponible=True,
                hrp_weights=hrp_w_dict,
                hrp_return=hrp_ret,
                hrp_volatility=hrp_vol,
                hrp_sharpe=hrp_sharpe_val,
                delta_aversion=bl.delta_aversion if bl else 0.0,
                tau=bl.tau if bl else 0.0,
                fuente_delta=bl.fuente_delta if bl else "",
            )

        except Exception as exc:
            logger.error(f"Error en optimización Markowitz: {exc}")
            return PortfolioOptimization(
                max_sharpe_weights=equal_w, max_sharpe_return=0.0,
                max_sharpe_volatility=0.0, max_sharpe_sharpe=0.0,
                min_variance_weights=equal_w, min_variance_return=0.0,
                min_variance_volatility=0.0, efficient_frontier=[], disponible=False,
            )

    # ------------------------------------------------------------------
    # Recomendación y alertas
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Hierarchical Risk Parity
    #
    # HRP se apoya únicamente en la matriz de covarianza (no usa retornos
    # esperados en absoluto), por lo que no participa de Black-Litterman —
    # no pasar `mu`/`bl` acá, es intencional.
    # ------------------------------------------------------------------

    def _hrp_weights(
        self, returns_mat: "np.ndarray", tickers: List[str], max_weight: float = 0.65
    ) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (López de Prado, 2016).

        Pasos:
        1. Clustering jerárquico sobre la matriz de correlación (distancia = √((1-ρ)/2)).
        2. Quasi-diagonalización: reordenar activos según el dendrograma.
        3. Bisección recursiva: asignar pesos inversamente proporcionales a la
           varianza de cada sub-cluster — sin invertir la matriz de covarianza.

        Ventaja frente a Markowitz: estable ante pequeños cambios en correlaciones
        y no sufre el problema de "error maximization" de la optimización MV.
        """
        corr = np.corrcoef(returns_mat.T)
        cov  = np.cov(returns_mat.T) * TRADING_DAYS * (100 ** 2)
        n = len(tickers)

        # Distancia de correlación y clustering
        dist = np.sqrt(np.maximum((1 - corr) / 2, 0))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")

        # Orden de hojas del dendrograma (quasi-diagonal)
        order = leaves_list(link)

        # Bisección recursiva
        weights = np.ones(n)

        def _cluster_var(idxs):
            sub_cov = cov[np.ix_(idxs, idxs)]
            inv_diag = 1.0 / np.maximum(np.diag(sub_cov), 1e-8)
            w = inv_diag / inv_diag.sum()
            return float(w @ sub_cov @ w)

        def _bisect(cluster):
            if len(cluster) == 1:
                return
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            var_l = _cluster_var(left)
            var_r = _cluster_var(right)
            alpha = 1 - var_l / (var_l + var_r + 1e-8)
            weights[left]  *= alpha
            weights[right] *= (1 - alpha)
            _bisect(left)
            _bisect(right)

        _bisect(list(order))
        weights /= weights.sum()
        raw = {tickers[i]: float(weights[i]) for i in range(n)}

        # Aplicar cota máxima por activo con redistribución iterativa
        w = dict(raw)
        cap = max(0.01, min(max_weight, 0.99))
        for _ in range(30):
            over = {t: v - cap for t, v in w.items() if v > cap + 1e-6}
            if not over:
                break
            excess = sum(over.values())
            for t in over:
                w[t] = cap
            free = {t: v for t, v in w.items() if v < cap - 1e-6}
            total_free = sum(free.values())
            if total_free <= 0:
                break
            for t in free:
                w[t] += excess * (free[t] / total_free)
        total = sum(w.values())
        return {t: round(v / total, 4) for t, v in w.items()}

    def _generar_recomendacion(
        self,
        activos: List[AssetAnalysis],
        metricas: PortfolioMetrics,
        opt: PortfolioOptimization,
    ) -> str:
        n = len(activos)
        compras = sum(1 for a in activos if a.recommendation.tipo == "compra")
        ventas = sum(1 for a in activos if a.recommendation.tipo == "venta")

        if compras > n * 0.6:
            base = "Portafolio con señales predominantemente alcistas."
        elif ventas > n * 0.6:
            base = "Portafolio con señales predominantemente bajistas."
        else:
            base = "Portafolio con señales mixtas."

        if metricas.sharpe_ratio > 1.0:
            q = "Ratio Sharpe atractivo (>1,0)."
        elif metricas.sharpe_ratio > 0.5:
            q = "Ratio Sharpe moderado."
        else:
            q = "Ratio Sharpe bajo — considerar rebalanceo."

        d = (
            "Los activos ofrecen diversificación efectiva."
            if metricas.diversification_ratio > 1.05
            else "Activos muy correlacionados — diversificación limitada."
        )

        opt_note = ""
        if opt.disponible and opt.max_sharpe_sharpe > metricas.sharpe_ratio + 0.1:
            opt_note = (
                f" El portafolio optimizado (Sharpe={opt.max_sharpe_sharpe:.2f}) "
                "mejoraría el perfil riesgo-retorno."
            )

        return (
            f"{base} {q} {d}"
            f" Retorno esperado: {metricas.expected_return:.1f}% | "
            f"Volatilidad: {metricas.volatility:.1f}% | "
            f"VaR 95%: {metricas.var_95:.1f}%.{opt_note}"
        )

    def _detectar_alertas(
        self, activos: List[AssetAnalysis], metricas: PortfolioMetrics
    ) -> List[dict]:
        alertas = []

        for a in activos:
            if a.weight > 0.40:
                alertas.append({
                    "tipo": "concentracion",
                    "nivel": "warning",
                    "mensaje": f"{a.ticker} representa {a.weight*100:.1f}% del portafolio (límite sugerido: 40%)",
                    "ticker": a.ticker,
                })

        corr = metricas.correlation_matrix
        tickers = list(corr.keys())
        for i, t1 in enumerate(tickers):
            for t2 in tickers[i + 1:]:
                val = corr.get(t1, {}).get(t2, 0.0)
                if val > 0.85:
                    alertas.append({
                        "tipo": "correlacion_alta",
                        "nivel": "warning",
                        "mensaje": f"{t1} y {t2} tienen correlación muy alta ({val:.2f})",
                        "ticker": f"{t1}/{t2}",
                    })

        if metricas.var_95 < -20:
            alertas.append({
                "tipo": "var_elevado",
                "nivel": "critical",
                "mensaje": f"VaR 95% del portafolio es muy elevado: {metricas.var_95:.1f}%",
                "ticker": "PORTAFOLIO",
            })

        if metricas.sharpe_ratio < 0:
            alertas.append({
                "tipo": "sharpe_negativo",
                "nivel": "warning",
                "mensaje": f"Sharpe negativo ({metricas.sharpe_ratio:.2f}): el portafolio no compensa el riesgo tomado",
                "ticker": "PORTAFOLIO",
            })

        return alertas
