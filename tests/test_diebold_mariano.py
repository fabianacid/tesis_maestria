"""
Test de Diebold-Mariano (DM) — Comparación estadística de pronósticos de dirección

Compara el ensemble ML contra dos benchmarks:
  - Buy & Hold : siempre predice subida (benchmark trivial)
  - SMA Crossover 20/50 : predice según cruce de medias móviles (benchmark técnico)

Metodología walk-forward:
  1. Ventana de entrenamiento: 504 días hábiles (~2 años)
  2. Reentrenamiento: cada STEP=63 días (~trimestral), como en BacktestAgent
  3. Señal ML (prob_subida) constante para los 63 días del trimestre siguiente
  4. Dirección real: precio(t + HORIZON) > precio(t), con HORIZON=3 días
  5. ~500 observaciones diarias por ticker → mayor poder estadístico que los ~8 períodos trimestrales

Función de pérdida:
  - 0-1 loss    : L_t = 1 si predicción incorrecta, 0 si correcta
  - Brier score : L_t = (prob_t - y_t)²  (penaliza probabilidades mal calibradas)

Estadístico:
  - d_t = L_benchmark_t − L_ML_t   (positivo → ML pierde menos → ML es mejor)
  - DM  = mean(d) / sqrt(S_NW / T)   con S_NW varianza HAC Newey-West
  - MDM (Harvey, Leybourne & Newbold 1997): corrección por muestra finita, usa t(T-1)
  - H0: E[d_t] = 0   H1: E[d_t] > 0  (ML estrictamente mejor, test unilateral)

Test complementario:
  - Sign test binomial: ¿accuracy ML > 50%? — no asume distribución de la pérdida

Referencias:
  - Diebold & Mariano (1995), Journal of Business & Economic Statistics, 13(3), 253-263
  - Harvey, Leybourne & Newbold (1997), International Journal of Forecasting, 13(2), 281-291
  - López de Prado (2018), Advances in Financial Machine Learning, Cap. 11
"""

import io
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as sps
import yfinance as yf

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Agrega la raíz del proyecto al path para importar backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.agents.model_agent import ModelAgent


# ─────────────────────────────────────────────────────────────────────────────
# Núcleo estadístico
# ─────────────────────────────────────────────────────────────────────────────

def _newey_west_variance(d: np.ndarray, h: int) -> float:
    """
    Estimador de varianza HAC Newey-West del diferencial de pérdidas d_t.

    Kernel de Bartlett con ancho de banda bw = max(h-1, floor(4*(T/100)^(2/9))).
    El ancho de banda escala con T y el horizonte de predicción h para capturar
    la autocorrelación inducida por predicciones solapadas.
    """
    T = len(d)
    bw = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    dc = d - d.mean()
    S = float(np.dot(dc, dc)) / T
    for k in range(1, bw + 1):
        gamma_k = float(np.dot(dc[k:], dc[:-k])) / T
        bartlett_weight = 1.0 - k / (bw + 1)
        S += 2.0 * bartlett_weight * gamma_k
    return max(S, 1e-10)


def diebold_mariano_test(
    loss_model: np.ndarray,
    loss_bench: np.ndarray,
    h: int = 3,
) -> Dict:
    """
    Test de Diebold-Mariano con corrección MDM de Harvey et al. (1997).

    Args:
        loss_model: array de pérdidas del modelo ML (longitud T)
        loss_bench: array de pérdidas del benchmark (longitud T)
        h:          horizonte de predicción en días (para el ancho de banda HAC)

    Returns:
        dict con estadísticos, p-values e interpretación

    Interpretación de mdm_stat:
        > 0 → ML tiene pérdida promedio menor → ML es MEJOR
        < 0 → benchmark supera al ML
    """
    d = loss_bench - loss_model
    T = len(d)

    var_d = _newey_west_variance(d, h)
    dm_stat = float(d.mean() / np.sqrt(var_d / T))

    # Corrección Harvey et al. (1997) para muestras finitas
    factor = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 0.0))
    mdm_stat = dm_stat * factor

    # p-value unilateral: H1 = E[d] > 0 (ML es mejor)
    p_one = float(1.0 - sps.t.cdf(mdm_stat, df=T - 1))
    # p-value bilateral (para referencia)
    p_two = float(2.0 * min(sps.t.cdf(mdm_stat, df=T - 1),
                             1.0 - sps.t.cdf(mdm_stat, df=T - 1)))

    return {
        "T": T,
        "mean_loss_model": round(float(loss_model.mean()), 5),
        "mean_loss_bench": round(float(loss_bench.mean()), 5),
        "mean_diff": round(float(d.mean()), 5),
        "dm_stat": round(dm_stat, 3),
        "mdm_stat": round(mdm_stat, 3),
        "p_value_one": round(p_one, 4),
        "p_value_two": round(p_two, 4),
        "sig_90": p_one < 0.10,
        "sig_95": p_one < 0.05,
        "sig_99": p_one < 0.01,
    }


def sign_test(y_ml: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    Sign test binomial (no paramétrico).

    H0: P(predicción correcta) = 0.5
    H1: P(predicción correcta) > 0.5

    Complementa al DM test: no asume ninguna distribución para las pérdidas.
    """
    T = len(y_true)
    correct = int((y_ml == y_true).sum())
    result = sps.binomtest(correct, T, p=0.5, alternative="greater")
    return {
        "T": T,
        "correct": correct,
        "accuracy": round(correct / T, 4),
        "p_value": round(float(result.pvalue), 4),
        "sig_90": result.pvalue < 0.10,
        "sig_95": result.pvalue < 0.05,
        "sig_99": result.pvalue < 0.01,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluador walk-forward
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEvaluator:
    """
    Replica el esquema de BacktestAgent pero registra predicciones y pérdidas
    día a día en lugar de correr el simulador de cartera.

    Por cada ventana trimestral:
      - Reentrena ModelAgent con los últimos TRAIN_WINDOW días
      - Obtiene prob_subida (constante para el trimestre)
      - Para cada día del trimestre siguiente registra:
          * predicción ML (prob_subida >= 0.5 → 1, else 0)
          * predicción B&H (siempre 1)
          * predicción SMA crossover (SMA20 > SMA50 → 1, else 0)
          * dirección real (precio en t+HORIZON > precio en t → 1)

    La autocorrelación dentro del trimestre (misma prob_ml para 63 días)
    es manejada por el estimador HAC Newey-West.
    """

    TRAIN_WINDOW = 504  # días hábiles (~2 años)
    STEP = 63           # reentrenamiento trimestral
    HORIZON = 3         # días hacia adelante a predecir
    SMA_SHORT = 20
    SMA_LONG = 50

    def __init__(self):
        self.model_agent = ModelAgent(ventana_entrenamiento=self.TRAIN_WINDOW)

    def _sma_signal(self, close: pd.Series, idx: int) -> int:
        """1 si SMA_short > SMA_long en la posición idx, else 0."""
        if idx < self.SMA_LONG:
            return 1  # sin historia suficiente → conservador (como B&H)
        s = float(close.iloc[idx - self.SMA_SHORT:idx].mean())
        l = float(close.iloc[idx - self.SMA_LONG:idx].mean())
        return 1 if s > l else 0

    def _classify_regime(
        self, close: pd.Series, idx: int, vol_threshold: float
    ) -> Tuple[str, str]:
        """
        Clasifica el régimen de mercado en el día idx.

        Retorna (tendencia, volatilidad):
          tendencia  : 'alcista' si precio > SMA200, 'bajista' si precio < SMA200
          volatilidad: 'alto_vol' si vol 20d anualizada > vol_threshold (p75 histórico)
                       'bajo_vol' en caso contrario
        """
        # Tendencia: precio actual vs SMA200
        if idx >= 200:
            sma200 = float(close.iloc[idx - 200:idx].mean())
            tendencia = "alcista" if float(close.iloc[idx]) > sma200 else "bajista"
        else:
            tendencia = "alcista"

        # Volatilidad realizada 20 días, anualizada
        if idx >= 21:
            rets = close.iloc[idx - 20:idx].pct_change().dropna()
            vol = float(rets.std() * np.sqrt(252)) if len(rets) >= 2 else 0.0
        else:
            vol = 0.0

        vol_reg = "alto_vol" if vol > vol_threshold else "bajo_vol"
        return tendencia, vol_reg

    def evaluate(self, ticker: str, years: int = 2) -> Optional[Dict]:
        """
        Ejecuta el walk-forward para un ticker y devuelve arrays de pérdidas.

        Args:
            ticker: símbolo del activo
            years:  años de datos de test (se descargan years+2 para cubrir el
                    entrenamiento inicial)
        """
        print(f"\n  [{ticker}] Descargando datos...")
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * (years + 2) + 60)

        raw = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if raw.empty:
            print(f"  [{ticker}] ERROR: Sin datos de mercado.")
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close"])

        min_rows = self.TRAIN_WINDOW + self.STEP + self.HORIZON + self.SMA_LONG
        if len(raw) < min_rows:
            print(f"  [{ticker}] ERROR: Solo {len(raw)} días (mínimo {min_rows}).")
            return None

        close = raw["Close"]
        n = len(raw)
        observations: List[Dict] = []
        n_retrains = 0

        # Umbral de volatilidad: percentil 75 de la vol 20d anualizada sobre todo el período
        all_rets = close.pct_change().dropna()
        rolling_vol = all_rets.rolling(20).std() * np.sqrt(252)
        vol_threshold = float(rolling_vol.quantile(0.75))

        for train_end in range(self.TRAIN_WINDOW, n - self.HORIZON, self.STEP):
            train_slice = raw.iloc[:train_end].copy()

            # Predicción ML para este trimestre
            try:
                result = self.model_agent.predecir(
                    precios=train_slice,
                    ticker=ticker,
                    forzar_actualizacion=True,
                )
                prob_ml = float(result.prob_subida) if result else 0.5
            except Exception:
                prob_ml = 0.5

            n_retrains += 1
            period_end = min(train_end + self.STEP, n - self.HORIZON)

            # Una observación por día del período test
            for t in range(train_end, period_end):
                p_now = float(close.iloc[t])
                p_future = float(close.iloc[t + self.HORIZON])
                if p_now == 0.0:
                    continue
                actual = 1 if p_future > p_now else 0
                tendencia, vol_reg = self._classify_regime(close, t, vol_threshold)
                observations.append({
                    "date": str(raw.index[t].date()),
                    "prob_ml": prob_ml,
                    "y_ml": 1 if prob_ml >= 0.5 else 0,
                    "y_bh": 1,
                    "y_sma": self._sma_signal(close, t),
                    "y_true": actual,
                    "trend_regime": tendencia,
                    "vol_regime": vol_reg,
                })

            print(
                f"  [{ticker}] Período {n_retrains:2d} "
                f"({str(raw.index[train_end - 1].date())}): "
                f"prob_ML={prob_ml:.3f}  obs_acum={len(observations)}",
                end="\r",
            )

        print()  # nueva línea tras el carriage-return

        if len(observations) < 30:
            print(f"  [{ticker}] Observaciones insuficientes ({len(observations)}).")
            return None

        df = pd.DataFrame(observations)
        y_true = df["y_true"].values.astype(float)
        y_ml = df["y_ml"].values.astype(float)
        y_bh = df["y_bh"].values.astype(float)
        y_sma = df["y_sma"].values.astype(float)
        prob_ml = df["prob_ml"].values.astype(float)

        # ── Pérdidas 0-1 ────────────────────────────────────────────────────
        l01_ml = (y_ml != y_true).astype(float)
        l01_bh = (y_bh != y_true).astype(float)
        l01_sma = (y_sma != y_true).astype(float)

        # ── Brier score ──────────────────────────────────────────────────────
        # ML usa probabilidades continuas; benchmarks usan predicciones duras (0/1)
        brier_ml = (prob_ml - y_true) ** 2
        brier_bh = (y_bh - y_true) ** 2
        brier_sma = (y_sma - y_true) ** 2

        print(f"  [{ticker}] {len(observations)} observaciones | {n_retrains} reentrenamientos")

        return {
            "ticker": ticker,
            "n_obs": len(observations),
            "n_retrains": n_retrains,
            "acc_ml": round(float((y_ml == y_true).mean()), 4),
            "acc_bh": round(float((y_bh == y_true).mean()), 4),
            "acc_sma": round(float((y_sma == y_true).mean()), 4),
            # Arrays de pérdida (no serializables directamente → se extraen abajo)
            "l01_ml": l01_ml,
            "l01_bh": l01_bh,
            "l01_sma": l01_sma,
            "brier_ml": brier_ml,
            "brier_bh": brier_bh,
            "brier_sma": brier_sma,
            "y_ml": y_ml.astype(int),
            "y_true": y_true.astype(int),
            "trend_regime": df["trend_regime"].values,
            "vol_regime": df["vol_regime"].values,
            "vol_threshold": vol_threshold,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Análisis por régimen de mercado
# ─────────────────────────────────────────────────────────────────────────────

def dm_by_regime(data: Dict) -> Dict:
    """
    Aplica DM test (pérdida 0-1) dentro de cada subgrupo de régimen de mercado.

    Regímenes evaluados:
      Por tendencia (SMA200):
        'alcista'  — precio > SMA200: mercado en tendencia positiva de largo plazo
        'bajista'  — precio < SMA200: mercado en tendencia negativa de largo plazo
      Por volatilidad (vol 20d anualizada vs percentil 75 histórico):
        'alto_vol' — vol realizada > p75: mercado nervioso / correctivo
        'bajo_vol' — vol realizada < p75: mercado tranquilo / tendencial

    Cada subgrupo puede tener distintas condiciones de predictibilidad:
    el ML debería ser más útil en mercados bajistas y volátiles donde
    Buy & Hold pierde poder.
    """
    tr = data["trend_regime"]
    vr = data["vol_regime"]
    regimes: Dict[str, Optional[Dict]] = {}

    def _regime_stats(mask: np.ndarray) -> Optional[Dict]:
        n = int(mask.sum())
        if n < 20:
            return None
        lml  = data["l01_ml"][mask]
        lbh  = data["l01_bh"][mask]
        lsma = data["l01_sma"][mask]
        yml  = data["y_ml"][mask].astype(int)
        ytr  = data["y_true"][mask].astype(int)
        return {
            "n": n,
            "acc_ml":  round(float((yml == ytr).mean()), 4),
            "acc_bh":  round(float(1.0 - lbh.mean()), 4),
            "acc_sma": round(float(1.0 - lsma.mean()), 4),
            "dm_vs_bh":  diebold_mariano_test(lml, lbh,  h=WalkForwardEvaluator.HORIZON),
            "dm_vs_sma": diebold_mariano_test(lml, lsma, h=WalkForwardEvaluator.HORIZON),
            "sign_test": sign_test(yml, ytr),
        }

    for label in ("alcista", "bajista", "alto_vol", "bajo_vol"):
        mask = (tr == label) if label in ("alcista", "bajista") else (vr == label)
        regimes[label] = _regime_stats(mask)

    return regimes


# ─────────────────────────────────────────────────────────────────────────────
# Salida por pantalla
# ─────────────────────────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "** "
    if p < 0.10:
        return "*  "
    return "   "


def _print_ticker_results(ticker: str, r: Dict) -> None:
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  {ticker}  |  {r['n_obs']} obs diarias  |  {r['n_retrains']} reentrenamientos")
    print(sep)
    print(
        f"  Accuracy:  ML = {r['accuracy']['ml']:.2%}  "
        f"B&H = {r['accuracy']['bh']:.2%}  "
        f"SMA = {r['accuracy']['sma']:.2%}"
    )
    print()
    print(f"  {'Test':<30} {'MDM':>8}  {'p (1-cola)':>11}  {'Sig':>5}")
    print(f"  {'─'*56}")

    for label, key in [
        ("DM vs B&H      (pérdida 0-1)", "dm_vs_bh_01"),
        ("DM vs SMA 20/50 (pérdida 0-1)", "dm_vs_sma_01"),
        ("DM vs B&H      (Brier score)", "dm_vs_bh_brier"),
        ("DM vs SMA 20/50 (Brier score)", "dm_vs_sma_brier"),
    ]:
        dm = r[key]
        print(
            f"  {label:<30} {dm['mdm_stat']:>+8.3f}  "
            f"{dm['p_value_one']:>11.4f}  "
            f"{_stars(dm['p_value_one']):>5}"
        )

    st = r["sign_test"]
    print(
        f"  {'Sign test binomial':<30} {'':>8}  "
        f"{st['p_value']:>11.4f}  "
        f"{_stars(st['p_value']):>5}"
        f"  (acc={st['accuracy']:.2%})"
    )
    print(f"\n  Leyenda: *** p<0.01  ** p<0.05  * p<0.10  (H1: ML es superior)")

    if r.get("by_regime"):
        _print_regime_results(r["by_regime"])


def _print_regime_results(regime_results: Dict) -> None:
    LABELS = {
        "alcista":  "Alcista  (precio > SMA200)   ",
        "bajista":  "Bajista  (precio < SMA200)   ",
        "alto_vol": "Alta vol (20d vol > p75)     ",
        "bajo_vol": "Baja vol (20d vol ≤ p75)     ",
    }
    SEP = "  " + "─" * 74
    print(f"\n  ── Análisis por régimen ─────────────────────────────────────────────")
    print(f"  {'Régimen':<30} {'N':>5}  {'AccML':>6} {'AccBH':>6}  {'DM vs B&H':^15}  {'DM vs SMA':^15}")
    print(f"  {'':30} {'':5}  {'':6} {'':6}  {'MDM':>6} {'p':>8}  {'MDM':>6} {'p':>8}")
    print(SEP)

    for key in ("alcista", "bajista", "alto_vol", "bajo_vol"):
        r = regime_results.get(key)
        if r is None:
            print(f"  {LABELS[key]:<30} {'< 20 obs — omitido':>40}")
            continue
        bh  = r["dm_vs_bh"]
        sma = r["dm_vs_sma"]
        print(
            f"  {LABELS[key]:<30} {r['n']:>5}  "
            f"{r['acc_ml']:>5.1%} {r['acc_bh']:>5.1%}  "
            f"{bh['mdm_stat']:>+6.2f} {bh['p_value_one']:>7.4f}{_stars(bh['p_value_one'])[0]}  "
            f"{sma['mdm_stat']:>+6.2f} {sma['p_value_one']:>7.4f}{_stars(sma['p_value_one'])[0]}"
        )


def print_aggregate_summary(results: Dict) -> None:
    if not results:
        return
    sep = "=" * 80
    print(f"\n{sep}")
    print("  RESUMEN AGREGADO — TEST DE DIEBOLD-MARIANO")
    print(sep)
    print(
        f"  {'Ticker':<8} {'AccML':>7} {'AccBH':>7} {'AccSMA':>7}"
        f"  {'DM vs B&H':^14}  {'DM vs SMA':^14}  {'Sign':^10}"
    )
    print(f"  {'':8} {'':7} {'':7} {'':7}  {'MDM':>6} {'p':>7}  {'MDM':>6} {'p':>7}  {'p':>7}")
    print(f"  {'─'*76}")

    for ticker, r in results.items():
        bh = r["dm_vs_bh_01"]
        sma = r["dm_vs_sma_01"]
        st = r["sign_test"]
        print(
            f"  {ticker:<8} "
            f"{r['accuracy']['ml']:>6.1%} "
            f"{r['accuracy']['bh']:>6.1%} "
            f"{r['accuracy']['sma']:>6.1%}  "
            f"{bh['mdm_stat']:>+6.2f} {bh['p_value_one']:>7.4f}{_stars(bh['p_value_one'])[0:2]}  "
            f"{sma['mdm_stat']:>+6.2f} {sma['p_value_one']:>7.4f}{_stars(sma['p_value_one'])[0:2]}  "
            f"{st['p_value']:>7.4f}{_stars(st['p_value'])[0:2]}"
        )

    n = len(results)
    n_bh = sum(1 for r in results.values() if r["dm_vs_bh_01"]["sig_95"])
    n_sma = sum(1 for r in results.values() if r["dm_vs_sma_01"]["sig_95"])
    n_sign = sum(1 for r in results.values() if r["sign_test"]["sig_95"])

    print(f"\n  Tickers con ML superior (p<0.05):")
    print(f"    vs Buy & Hold  : {n_bh}/{n}")
    print(f"    vs SMA 20/50   : {n_sma}/{n}")
    print(f"    Sign test      : {n_sign}/{n}")

    if n_bh + n_sma + n_sign == 0:
        conclusion = "Sin evidencia estadística de superioridad del ensemble ML."
    elif n_bh > n // 2 or n_sma > n // 2:
        conclusion = "Evidencia de superioridad ML en mayoría de tickers evaluados."
    else:
        conclusion = "Evidencia mixta — ML supera benchmarks en algunos activos."
    print(f"\n  Conclusión global: {conclusion}")

    # ── Agregado por régimen ─────────────────────────────────────────────────
    regime_keys = ("alcista", "bajista", "alto_vol", "bajo_vol")
    tickers_with_regime = [r for r in results.values() if r.get("by_regime")]
    if tickers_with_regime:
        print(f"\n  ── Superioridad ML por régimen (p<0.05, pérdida 0-1) ────────────────")
        print(f"  {'Régimen':<30}  {'vs B&H':^12}  {'vs SMA':^12}")
        print(f"  {'─'*58}")
        for reg in regime_keys:
            sig_bh  = sum(
                1 for r in tickers_with_regime
                if r["by_regime"].get(reg) and r["by_regime"][reg]["dm_vs_bh"]["sig_95"]
            )
            sig_sma = sum(
                1 for r in tickers_with_regime
                if r["by_regime"].get(reg) and r["by_regime"][reg]["dm_vs_sma"]["sig_95"]
            )
            total = sum(1 for r in tickers_with_regime if r["by_regime"].get(reg))
            label = {
                "alcista": "Alcista  (precio > SMA200)",
                "bajista": "Bajista  (precio < SMA200)",
                "alto_vol": "Alta vol (20d vol > p75) ",
                "bajo_vol": "Baja vol (20d vol ≤ p75) ",
            }[reg]
            print(f"  {label:<30}  {sig_bh}/{total} tickers    {sig_sma}/{total} tickers")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def run_dm_suite(tickers: List[str], years: int = 2) -> Dict:
    evaluator = WalkForwardEvaluator()
    all_results: Dict[str, Dict] = {}

    for ticker in tickers:
        t0 = time.time()
        data = evaluator.evaluate(ticker, years=years)
        elapsed = round(time.time() - t0, 1)

        if data is None:
            print(f"  [{ticker}] SKIP — datos insuficientes o error de descarga.")
            continue

        dm_bh_01 = diebold_mariano_test(data["l01_ml"], data["l01_bh"], h=WalkForwardEvaluator.HORIZON)
        dm_sma_01 = diebold_mariano_test(data["l01_ml"], data["l01_sma"], h=WalkForwardEvaluator.HORIZON)
        dm_bh_br = diebold_mariano_test(data["brier_ml"], data["brier_bh"], h=WalkForwardEvaluator.HORIZON)
        dm_sma_br = diebold_mariano_test(data["brier_ml"], data["brier_sma"], h=WalkForwardEvaluator.HORIZON)
        st = sign_test(data["y_ml"], data["y_true"])

        ticker_result = {
            "n_obs": data["n_obs"],
            "n_retrains": data["n_retrains"],
            "accuracy": {
                "ml": data["acc_ml"],
                "bh": data["acc_bh"],
                "sma": data["acc_sma"],
            },
            "dm_vs_bh_01": dm_bh_01,
            "dm_vs_sma_01": dm_sma_01,
            "dm_vs_bh_brier": dm_bh_br,
            "dm_vs_sma_brier": dm_sma_br,
            "sign_test": st,
            "by_regime": dm_by_regime(data),
            "elapsed_seconds": elapsed,
        }
        all_results[ticker] = ticker_result
        _print_ticker_results(ticker, ticker_result)

    return all_results


def _make_serializable(obj):
    """Convierte arrays numpy a listas para serialización JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    # ── Configuración ────────────────────────────────────────────────────────
    # Reducí TICKERS o YEARS para ejecuciones más rápidas durante desarrollo.
    # Para la tesis usa al menos 5 tickers y 3 años.
    TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "JPM"]
    YEARS = 2  # años de período test (se descargan YEARS+2 para el entrenamiento inicial)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   TEST DE DIEBOLD-MARIANO — Sistema Multiagente ML              ║
║   Ensemble ML  vs  Buy & Hold  vs  SMA Crossover (20/50)        ║
╠══════════════════════════════════════════════════════════════════╣
║   Tickers  : {', '.join(TICKERS):<51}║
║   Período  : {YEARS} años de test + 2 de entrenamiento inicial         ║
║   Señal ML : reentrenamiento trimestral (STEP=63 días)          ║
║   Pérdidas : 0-1 loss (dirección) + Brier score (calibración)  ║
║   Test     : MDM (Harvey et al. 1997) + Sign test binomial      ║
╚══════════════════════════════════════════════════════════════════╝
""")

    t_total = time.time()
    results = run_dm_suite(TICKERS, years=YEARS)

    if not results:
        print("\nNo se obtuvieron resultados. Verificá la conexión y los tickers.")
        return

    print_aggregate_summary(results)

    # ── Guardar resultados ───────────────────────────────────────────────────
    os.makedirs("test_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"test_results/dm_test_{timestamp}.json"

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(results), f, indent=2, ensure_ascii=False)

    total_min = (time.time() - t_total) / 60
    print(f"\n  Resultados guardados en: {outfile}")
    print(f"  Tiempo total: {total_min:.1f} min\n")


if __name__ == "__main__":
    main()
