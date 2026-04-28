"""
Backtesting del sistema multiagente de predicción bursátil — v3 Multi-Agente

Combina las señales de los 3 agentes disponibles sin lookahead bias:

  ModelAgent      → prob_subida (entrenado solo con datos hasta t)
  MarketAgent     → señal técnica calculada desde OHLCV histórico (SMA, RSI, MACD, ATR)
  SentimentAgent  → proxy de momentum de precio (las noticias históricas no están
                    disponibles en yfinance, por lo que se usa el retorno de 5 días
                    relativo como señal de sentimiento del mercado)
  RecommendationAgent → combina las señales anteriores en una recomendación unificada

Señal de COMPRA (consenso requerido):
  - prob_subida >= UMBRAL_COMPRA
  - señal técnica != "bajista"
  - recomendación no es venta
  - precio > SMA20 (uptrend)

Señal de VENTA (cualquiera activa):
  - prob_subida <= UMBRAL_VENTA
  - señal técnica == "bajista"
  - recomendación es venta
  - stop-loss activado
  - posición vencida sin señal de extensión

Uso:
    python backtesting.py
    python backtesting.py --tickers MSFT GOOGL --periodo 1y --paso 5
"""

import argparse
import json
import logging
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance no instalado. Ejecutá: pip install yfinance")
    sys.exit(1)

# Fix encoding para Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

logging.basicConfig(level=logging.ERROR)

from backend.agents.model_agent import ModelAgent
from backend.agents.recommendation_agent import RecommendationAgent

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
CAPITAL_INICIAL         = 10_000.0
COMISION_PCT            = 0.001       # 0.1% por operación
HORIZONTE_DIAS          = 3           # días de horizonte del modelo
MIN_DATOS_ENTRENAMIENTO = 120         # días mínimos antes de operar
UMBRAL_COMPRA           = 0.55        # prob_subida mínima (más bajo porque hay filtros extra)
UMBRAL_VENTA            = 0.45        # prob_subida máxima para salir
STOP_LOSS_PCT           = 0.04        # stop-loss 4%
SMA_VENTANA             = 20
PASO_REENTRENAMIENTO    = 5
PERIODO_HISTORICO       = "2y"
# ──────────────────────────────────────────────────────────────────────────────


# ─── DESCARGA DE DATOS ────────────────────────────────────────────────────────

def descargar_datos(ticker: str, periodo: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period=periodo)
        if df.empty:
            print(f"  [!] Sin datos para {ticker}")
            return None
        df = df.sort_index()
        print(f"  Datos: {len(df)} días  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"  [!] Error descargando {ticker}: {e}")
        return None


# ─── SEÑALES TÉCNICAS Y PROXY DE SENTIMIENTO ─────────────────────────────────

def calcular_señales_mercado(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula indicadores técnicos (proxy de MarketAgent) y sentimiento de
    momentum (proxy de SentimentAgent) desde datos OHLCV históricos.
    Sin lookahead: usa solo datos disponibles hasta la fecha actual.
    """
    close  = df["Close"]
    high   = df["High"]   if "High"   in df.columns else close
    low    = df["Low"]    if "Low"    in df.columns else close
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=close.index)

    precio = float(close.iloc[-1])

    # ── SMAs ──────────────────────────────────────────────────────────────────
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma20
    if pd.isna(sma20): sma20 = precio
    if pd.isna(sma50): sma50 = sma20

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - (100 / (1 + rs))).iloc[-1])
    if pd.isna(rsi): rsi = 50.0

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12       = close.ewm(span=12).mean()
    ema26       = close.ewm(span=26).mean()
    macd_line   = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    macd_hist   = float((macd_line - macd_signal).iloc[-1])
    macd_sig    = float(macd_signal.iloc[-1])
    if pd.isna(macd_hist): macd_hist = 0.0
    if pd.isna(macd_sig):  macd_sig  = 0.0

    # ── ATR ───────────────────────────────────────────────────────────────────
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    if pd.isna(atr): atr = float(close.std())

    # ── Volume ratio ──────────────────────────────────────────────────────────
    vol_sma20  = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio  = float(volume.iloc[-1]) / vol_sma20 if vol_sma20 > 0 else 1.0
    if pd.isna(vol_ratio): vol_ratio = 1.0

    # ── Proxy de sentimiento: momentum de precio a 5 días ─────────────────────
    ret_5 = float(close.pct_change(5).iloc[-1])  if len(close) >= 6  else 0.0
    if pd.isna(ret_5): ret_5 = 0.0
    sentiment_score = float(np.clip(ret_5 * 10, -1, 1))

    if sentiment_score > 0.1:
        sentimiento   = "positivo"
        conf_sent     = min(abs(sentiment_score), 0.8)
    elif sentiment_score < -0.1:
        sentimiento   = "negativo"
        conf_sent     = min(abs(sentiment_score), 0.8)
    else:
        sentimiento   = "neutral"
        conf_sent     = 0.5

    # ── Régimen de mercado ────────────────────────────────────────────────────
    if precio > sma20 > sma50:
        market_regime = "tendencia_alcista"
    elif precio < sma20 < sma50:
        market_regime = "tendencia_bajista"
    elif atr / precio > 0.03:
        market_regime = "alta_volatilidad"
    else:
        market_regime = "lateral"

    # ── Señal técnica compuesta ───────────────────────────────────────────────
    score = sum([
        precio > sma20,       # precio sobre tendencia de corto plazo
        sma20 > sma50,        # tendencia de corto > mediano plazo
        macd_hist > 0,        # MACD alcista
        30 < rsi < 70,        # RSI en zona sana (no sobrecomprado ni sobrevendido)
        ret_5 > 0,            # momentum positivo reciente
    ])

    if score >= 4:
        senal = "alcista"
    elif score <= 1:
        senal = "bajista"
    else:
        senal = "neutral"

    return {
        "senal":                senal,
        "market_regime":        market_regime,
        "rsi":                  rsi,
        "macd_signal":          macd_sig,
        "atr":                  atr,
        "vol_ratio":            vol_ratio,
        "sentimiento":          sentimiento,
        "sentiment_score":      sentiment_score,
        "confianza_sentimiento":conf_sent,
        "sma20":                sma20,
        "sma50":                sma50,
    }


# ─── SIMULACIÓN PRINCIPAL ─────────────────────────────────────────────────────

def ejecutar_backtesting(
    ticker: str,
    periodo: str              = PERIODO_HISTORICO,
    capital_inicial: float    = CAPITAL_INICIAL,
    umbral_compra: float      = UMBRAL_COMPRA,
    umbral_venta: float       = UMBRAL_VENTA,
    stop_loss_pct: float      = STOP_LOSS_PCT,
    paso_reentrenamiento: int = PASO_REENTRENAMIENTO,
    verbose: bool             = True,
) -> Optional[pd.DataFrame]:
    """
    Simula la estrategia multi-agente día a día sobre datos históricos reales.
    """
    df_completo = descargar_datos(ticker, periodo)
    if df_completo is None:
        return None

    total_dias     = len(df_completo)
    fin_simulacion = total_dias - HORIZONTE_DIAS
    if fin_simulacion <= MIN_DATOS_ENTRENAMIENTO + 10:
        print(f"  [!] Datos insuficientes ({total_dias} días)")
        return None

    model_agent          = ModelAgent(ventana_entrenamiento=252)
    recommendation_agent = RecommendationAgent()

    capital     = capital_inicial
    acciones    = 0.0
    precio_entrada = 0.0
    en_posicion = False
    dias_restantes_posicion = 0

    # Caché de señales (se actualizan cada PASO_REENTRENAMIENTO días)
    prob_subida_cache  = 0.5
    variacion_pct_cache = 0.0
    accuracy_cache     = 0.0
    señales_cache      = {"senal": "neutral", "market_regime": "lateral",
                          "rsi": 50.0, "macd_signal": 0.0, "atr": 1.0,
                          "vol_ratio": 1.0, "sentimiento": "neutral",
                          "sentiment_score": 0.0, "confianza_sentimiento": 0.5,
                          "sma20": 0.0, "sma50": 0.0}
    rec_tipo_cache     = "mantener"

    registros = []
    n_dias    = fin_simulacion - MIN_DATOS_ENTRENAMIENTO

    if verbose:
        print(f"  Simulando {n_dias} días de trading "
              f"(reentrenando cada {paso_reentrenamiento} días)...")

    for i, idx in enumerate(range(MIN_DATOS_ENTRENAMIENTO, fin_simulacion)):
        fecha_hoy  = df_completo.index[idx]
        precio_hoy = float(df_completo["Close"].iloc[idx])

        # ── Actualizar señales de todos los agentes ────────────────────────
        if i % paso_reentrenamiento == 0:
            df_hasta_hoy = df_completo.iloc[: idx + 1].copy()

            # 1. ModelAgent
            try:
                resultado = model_agent.predecir(
                    df_hasta_hoy, ticker, forzar_actualizacion=True
                )
                if resultado:
                    prob_subida_cache   = resultado.prob_subida
                    variacion_pct_cache = resultado.variacion_pct
                    accuracy_cache      = resultado.metricas_completas.accuracy
            except Exception:
                pass

            # 2. MarketAgent proxy + SentimentAgent proxy
            try:
                señales_cache = calcular_señales_mercado(df_hasta_hoy)
            except Exception:
                pass

            # 3. RecommendationAgent (combina señales 1 y 2)
            try:
                rec = recommendation_agent.generar_recomendacion(
                    ticker               = ticker,
                    senal_mercado        = señales_cache["senal"],
                    variacion_pct        = variacion_pct_cache,
                    sentimiento          = señales_cache["sentimiento"],
                    confianza_sentimiento= señales_cache["confianza_sentimiento"],
                    ultimo_precio        = precio_hoy,
                    volatilidad          = señales_cache["atr"],
                    market_regime        = señales_cache["market_regime"],
                    rsi                  = señales_cache["rsi"],
                    macd_signal          = señales_cache["macd_signal"],
                    volume_ratio         = señales_cache["vol_ratio"],
                    prediction_confidence= accuracy_cache,
                    sentiment_score      = señales_cache["sentiment_score"],
                )
                rec_tipo_cache = rec.tipo
            except Exception:
                pass

        prob_subida   = prob_subida_cache
        senal_tecnica = señales_cache["senal"]
        rec_tipo      = rec_tipo_cache
        sma20         = señales_cache["sma20"] if señales_cache["sma20"] > 0 else precio_hoy
        en_uptrend    = precio_hoy > sma20
        direccion     = "SUBIDA" if prob_subida > 0.5 else "BAJADA"

        # ── Precio real t+3 (solo para métricas, no para decisiones) ──────
        precio_t3      = float(df_completo["Close"].iloc[idx + HORIZONTE_DIAS])
        direccion_real = "SUBIDA" if precio_t3 > precio_hoy else "BAJADA"
        acertada       = (direccion == direccion_real)

        # ── Stop-loss ─────────────────────────────────────────────────────
        stop_loss_activado = (
            en_posicion
            and precio_entrada > 0
            and precio_hoy < precio_entrada * (1 - stop_loss_pct)
        )

        # ── Señal combinada de los 3 agentes ──────────────────────────────
        señal_compra = (
            prob_subida  >= umbral_compra
            and senal_tecnica != "bajista"
            and rec_tipo not in ["venta", "venta_fuerte", "venta_debil"]
            and en_uptrend
        )
        señal_venta = (
            prob_subida  <= umbral_venta
            or senal_tecnica == "bajista"
            or rec_tipo in ["venta", "venta_fuerte"]
        )

        # ── Lógica de trading ─────────────────────────────────────────────
        accion = "MANTENER"

        if dias_restantes_posicion > 0:
            dias_restantes_posicion -= 1

        if stop_loss_activado:
            valor_venta = acciones * precio_hoy
            comision    = valor_venta * COMISION_PCT
            capital     = valor_venta - comision
            acciones    = 0.0
            en_posicion = False
            dias_restantes_posicion = 0
            accion      = "VENTA_STOPLOSS"

        elif not en_posicion and señal_compra:
            comision       = capital * COMISION_PCT
            acciones       = (capital - comision) / precio_hoy
            precio_entrada = precio_hoy
            capital        = 0.0
            en_posicion    = True
            dias_restantes_posicion = HORIZONTE_DIAS
            accion         = "COMPRA"

        elif en_posicion and señal_venta:
            valor_venta = acciones * precio_hoy
            comision    = valor_venta * COMISION_PCT
            capital     = valor_venta - comision
            acciones    = 0.0
            en_posicion = False
            dias_restantes_posicion = 0
            accion      = "VENTA"

        elif en_posicion and dias_restantes_posicion == 0:
            if señal_compra:
                # Extender posición si el consenso sigue siendo alcista
                dias_restantes_posicion = HORIZONTE_DIAS
                accion = "EXTENDER"
            else:
                valor_venta = acciones * precio_hoy
                comision    = valor_venta * COMISION_PCT
                capital     = valor_venta - comision
                acciones    = 0.0
                en_posicion = False
                accion      = "VENTA"

        # ── Valor del portfolio ───────────────────────────────────────────
        valor_portfolio = capital + acciones * precio_hoy
        retorno_acum    = (valor_portfolio - capital_inicial) / capital_inicial * 100

        registros.append({
            "fecha":                 str(fecha_hoy.date()) if hasattr(fecha_hoy, "date") else str(fecha_hoy),
            "ticker":                ticker,
            "precio":                round(precio_hoy, 2),
            # Señales de cada agente
            "prob_subida":           round(prob_subida, 4),
            "senal_tecnica":         senal_tecnica,
            "sentimiento_proxy":     señales_cache["sentimiento"],
            "recomendacion":         rec_tipo,
            "rsi":                   round(señales_cache["rsi"], 1),
            # Resultado
            "direccion_predicha":    direccion,
            "precio_real_t3":        round(precio_t3, 2),
            "direccion_real":        direccion_real,
            "prediccion_correcta":   acertada,
            "accuracy_modelo":       round(accuracy_cache, 4),
            # Trading
            "accion":                accion,
            "stop_loss_activado":    stop_loss_activado,
            "en_posicion":           en_posicion,
            "valor_portfolio":       round(valor_portfolio, 2),
            "retorno_acumulado_pct": round(retorno_acum, 2),
        })

        if verbose and (i + 1) % 30 == 0:
            print(f"    [{i+1}/{n_dias}] {fecha_hoy.date()} | "
                  f"Modelo:{prob_subida:.2f}  Técnica:{senal_tecnica:<8}  "
                  f"Rec:{rec_tipo:<15}  ${valor_portfolio:,.0f} ({retorno_acum:+.1f}%)")

    # Cerrar posición abierta al final
    if en_posicion and acciones > 0:
        precio_cierre = float(df_completo["Close"].iloc[fin_simulacion - 1])
        valor_venta   = acciones * precio_cierre
        capital       = valor_venta - valor_venta * COMISION_PCT
        if registros:
            registros[-1]["valor_portfolio"]       = round(capital, 2)
            registros[-1]["retorno_acumulado_pct"] = round(
                (capital - capital_inicial) / capital_inicial * 100, 2
            )

    return pd.DataFrame(registros)


# ─── CÁLCULO DE MÉTRICAS ─────────────────────────────────────────────────────

def calcular_metricas(df: pd.DataFrame, capital_inicial: float = CAPITAL_INICIAL) -> dict:
    if df.empty:
        return {}

    capital_final = df["valor_portfolio"].iloc[-1]
    retorno_total = (capital_final - capital_inicial) / capital_inicial * 100

    precio_inicio = df["precio"].iloc[0]
    precio_fin    = df["precio"].iloc[-1]
    retorno_bh    = (precio_fin - precio_inicio) / precio_inicio * 100

    retornos_diarios = df["valor_portfolio"].pct_change().dropna()
    sharpe = (
        (retornos_diarios.mean() / retornos_diarios.std()) * np.sqrt(252)
        if retornos_diarios.std() > 0 else 0.0
    )

    running_max  = df["valor_portfolio"].cummax()
    max_drawdown = float(((df["valor_portfolio"] - running_max) / running_max * 100).min())

    total_pred  = len(df)
    correctas   = int(df["prediccion_correcta"].sum())
    pct_acierto = correctas / total_pred * 100 if total_pred else 0.0

    n_compras     = int((df["accion"] == "COMPRA").sum())
    n_ventas      = int((df["accion"] == "VENTA").sum())
    n_stoploss    = int((df["accion"] == "VENTA_STOPLOSS").sum())
    n_extensiones = int((df["accion"] == "EXTENDER").sum())

    # Distribución de recomendaciones usadas
    dist_rec = df["recomendacion"].value_counts().to_dict() if "recomendacion" in df.columns else {}

    return {
        "capital_inicial":            capital_inicial,
        "capital_final":              round(capital_final, 2),
        "retorno_total_pct":          round(retorno_total, 2),
        "retorno_buy_hold_pct":       round(retorno_bh, 2),
        "diferencia_vs_buyhold_pct":  round(retorno_total - retorno_bh, 2),
        "sharpe_ratio":               round(sharpe, 3),
        "max_drawdown_pct":           round(max_drawdown, 2),
        "pct_predicciones_correctas": round(pct_acierto, 2),
        "predicciones_correctas":     correctas,
        "total_predicciones":         total_pred,
        "n_compras":                  n_compras,
        "n_ventas":                   n_ventas,
        "n_stoploss":                 n_stoploss,
        "n_extensiones":              n_extensiones,
        "distribucion_recomendaciones": dist_rec,
        "dias_simulados":             len(df),
    }


# ─── SALIDA ───────────────────────────────────────────────────────────────────

def imprimir_resumen(metricas: dict, ticker: str) -> None:
    m = metricas
    print(f"\n{'='*60}")
    print(f"  BACKTESTING MULTI-AGENTE — {ticker}")
    print(f"{'='*60}")
    print(f"  Capital inicial:           ${m['capital_inicial']:>10,.2f}")
    print(f"  Capital final:             ${m['capital_final']:>10,.2f}")
    print(f"  Retorno total:             {m['retorno_total_pct']:>+9.2f} %")
    print(f"  Retorno Buy & Hold:        {m['retorno_buy_hold_pct']:>+9.2f} %")
    print(f"  Diferencia vs B&H:         {m['diferencia_vs_buyhold_pct']:>+9.2f} %")
    print(f"  Sharpe Ratio:              {m['sharpe_ratio']:>10.3f}")
    print(f"  Máximo Drawdown:           {m['max_drawdown_pct']:>+9.2f} %")
    print(f"  % Predicciones correctas:  {m['pct_predicciones_correctas']:>9.1f} %")
    print(f"  Compras / Ventas:          {m['n_compras']}/{m['n_ventas']}")
    print(f"  Stop-loss activados:       {m['n_stoploss']:>10}")
    print(f"  Extensiones de posición:   {m['n_extensiones']:>10}")
    if m.get("distribucion_recomendaciones"):
        print(f"  Recomendaciones usadas:")
        for k, v in sorted(m["distribucion_recomendaciones"].items(), key=lambda x: -x[1]):
            print(f"    {k:<22} {v:>4} días")
    print(f"  Días simulados:            {m['dias_simulados']:>10}")
    print(f"{'='*60}")


def guardar_resultados(df: pd.DataFrame, metricas: dict, ticker: str) -> tuple:
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    excel_file = output_dir / f"backtest_{ticker}_{ts}.xlsx"
    json_file  = output_dir / f"metricas_{ticker}_{ts}.json"

    df.to_excel(excel_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    return excel_file, json_file


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtesting multi-agente")
    parser.add_argument("--tickers",   nargs="+", default=["AAPL", "MSFT", "GOOGL"])
    parser.add_argument("--periodo",   default=PERIODO_HISTORICO,
                        help="1y, 2y, 3y (default: 2y)")
    parser.add_argument("--capital",   type=float, default=CAPITAL_INICIAL)
    parser.add_argument("--paso",      type=int,   default=PASO_REENTRENAMIENTO,
                        help="Reentrenar cada N días (default: 5)")
    parser.add_argument("--stop_loss", type=float, default=STOP_LOSS_PCT,
                        help="Stop-loss como fracción (default: 0.04 = 4%%)")
    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════╗
║   BACKTESTING MULTI-AGENTE — SISTEMA MULTIAGENTE      ║
╚═══════════════════════════════════════════════════════╝
""")
    print(f"  Agentes activos:  ModelAgent + MarketAgent(proxy) + "
          f"SentimentAgent(proxy) + RecommendationAgent")
    print(f"  Capital inicial:      ${args.capital:,.0f}")
    print(f"  Período:              {args.periodo}")
    print(f"  Umbral compra:        {UMBRAL_COMPRA}  (+ consenso de agentes)")
    print(f"  Umbral venta:         {UMBRAL_VENTA}   (o señal bajista de cualquier agente)")
    print(f"  Stop-loss:            {args.stop_loss*100:.1f}%")
    print(f"  Horizonte:            {HORIZONTE_DIAS} días (extensible)")
    print(f"  Comisión:             {COMISION_PCT*100:.1f}% por operación")
    print(f"  Paso reentrenamiento: cada {args.paso} días")

    todos = []

    for ticker in args.tickers:
        print(f"\n{'─'*60}")
        print(f"  Procesando {ticker} ...")
        print(f"{'─'*60}")

        df = ejecutar_backtesting(
            ticker,
            periodo              = args.periodo,
            capital_inicial      = args.capital,
            paso_reentrenamiento = args.paso,
            stop_loss_pct        = args.stop_loss,
            verbose              = True,
        )

        if df is None or df.empty:
            print(f"  Sin resultados para {ticker}")
            continue

        metricas            = calcular_metricas(df, args.capital)
        metricas["ticker"]  = ticker
        todos.append(metricas)

        imprimir_resumen(metricas, ticker)

        excel_file, json_file = guardar_resultados(df, metricas, ticker)
        print(f"\n  Archivos guardados:")
        print(f"    Excel: {excel_file}")
        print(f"    JSON:  {json_file}")

    if len(todos) > 1:
        print(f"\n{'='*74}")
        print(f"  COMPARATIVA ENTRE TICKERS")
        print(f"{'='*74}")
        print(f"  {'Ticker':<8}  {'Retorno':>9}  {'B&H':>9}  "
              f"{'vs B&H':>8}  {'Sharpe':>7}  {'Drawdown':>9}  {'Acierto':>8}")
        print(f"  {'-'*72}")
        for m in todos:
            print(f"  {m['ticker']:<8}  "
                  f"{m['retorno_total_pct']:>+8.2f}%  "
                  f"{m['retorno_buy_hold_pct']:>+8.2f}%  "
                  f"{m['diferencia_vs_buyhold_pct']:>+7.2f}%  "
                  f"{m['sharpe_ratio']:>7.3f}  "
                  f"{m['max_drawdown_pct']:>+8.2f}%  "
                  f"{m['pct_predicciones_correctas']:>7.1f}%")
        print(f"{'='*74}\n")

    print("  Backtesting completado.\n")


if __name__ == "__main__":
    main()
