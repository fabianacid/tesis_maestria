"""
Router de Perfil de Riesgo

Endpoints:
  POST /risk/profile
      Evalúa el cuestionario de 6 dimensiones y devuelve:
      - Perfil de riesgo (muy_conservador … muy_agresivo)
      - Score 0-100 con desglose por dimensión
      - Sectores/ETFs recomendados y pesos target
      - Presupuesto de VaR tolerado

  POST /risk/portfolio
      Usa los ETFs recomendados del perfil y lanza el PortfolioAgent
      completo (Markowitz + HRP + análisis multiagente) con los pesos
      sugeridos por el perfil.
"""
import asyncio
import logging
from functools import partial

from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    RiskProfileRequest,
    RiskProfileResponse,
    RiskPortfolioRequest,
    PortfolioResponse,
    PortfolioAssetSchema,
    PortfolioMetricsSchema,
    PortfolioOptimizationSchema,
    EfficientFrontierPointSchema,
)
from ..agents import (
    MarketAgent, ModelAgent, SentimentAgent,
    RecommendationAgent, AlertAgent, SECAgent, PortfolioAgent,
)
from ..agents.risk_profile_agent import (
    RiskProfileAgent, PerfilRiesgo,
    Q01Autopercepcion, Q02ConcursoTV, Q03VacacionPerdidaEmpleo,
    Q04Inversion20k, Q05ComodidadAcciones, Q06PalabraRiesgo,
    Q07BonosVsActivosDuros, Q08GananciaPerdidaPotencial,
    Q09GananciaSegura, Q10PerdidaSegura, Q11Herencia100k,
    Q12Asignacion20k, Q13MinaOro,
    _SECTORES_POR_PERFIL, _RISK_BUDGET, _DELTA_AVERSION_POR_PERFIL,
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Perfil de Riesgo"])

# ── Singletons (reutilizan caché de otros routers si ya están en memoria) ──
_risk_agent = RiskProfileAgent()

_market_agent    = MarketAgent(ventana_ma=20, periodo_historico="2y")
_model_agent     = ModelAgent(ventana_entrenamiento=504)
_sentiment_agent = SentimentAgent()
_rec_agent       = RecommendationAgent()
_alert_agent     = AlertAgent(
    umbral_warning=settings.ALERT_THRESHOLD_WARNING,
    umbral_critical=settings.ALERT_THRESHOLD_CRITICAL,
)
_sec_agent       = SECAgent()
_portfolio_agent = PortfolioAgent(
    market_agent=_market_agent,
    model_agent=_model_agent,
    sentiment_agent=_sentiment_agent,
    recommendation_agent=_rec_agent,
    sec_agent=_sec_agent,
    alert_agent=_alert_agent,
)

# Mapeo de strings a enums para validación ──────────────────────────────────
# 13 ítems del instrumento Grable & Lytton (1999)
_ENUM_MAP = {
    "q01": {e.value: e for e in Q01Autopercepcion},
    "q02": {e.value: e for e in Q02ConcursoTV},
    "q03": {e.value: e for e in Q03VacacionPerdidaEmpleo},
    "q04": {e.value: e for e in Q04Inversion20k},
    "q05": {e.value: e for e in Q05ComodidadAcciones},
    "q06": {e.value: e for e in Q06PalabraRiesgo},
    "q07": {e.value: e for e in Q07BonosVsActivosDuros},
    "q08": {e.value: e for e in Q08GananciaPerdidaPotencial},
    "q09": {e.value: e for e in Q09GananciaSegura},
    "q10": {e.value: e for e in Q10PerdidaSegura},
    "q11": {e.value: e for e in Q11Herencia100k},
    "q12": {e.value: e for e in Q12Asignacion20k},
    "q13": {e.value: e for e in Q13MinaOro},
}


def _parse_enums(body: RiskProfileRequest):
    errors = []
    resultado = {}
    for campo, mapping in _ENUM_MAP.items():
        valor = getattr(body, campo)
        if valor not in mapping:
            errors.append(f"'{campo}': valor '{valor}' no válido. Opciones: {list(mapping.keys())}")
        else:
            resultado[campo] = mapping[valor]
    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errores": errors},
        )
    return resultado


# ── Endpoint 1: evaluar perfil ─────────────────────────────────────────────

@router.post(
    "/profile",
    response_model=RiskProfileResponse,
    summary="Evaluar perfil de riesgo del inversor",
    description="""
Recibe las respuestas al cuestionario de tolerancia al riesgo validado
**Grable & Lytton (1999)** (13 ítems) y devuelve:
- **Perfil de riesgo** (muy_conservador, conservador, moderado, agresivo, muy_agresivo)
- **Score** con desglose e interpretación de cada ítem
- **Sectores recomendados** con ETF representativo y peso sugerido
- **Presupuesto de riesgo**: VaR 95% máximo tolerado y volatilidad anual objetivo
- **δ (delta_aversion)**: coeficiente de aversión al riesgo para Black-Litterman

### Cuestionario (ver `RiskProfileRequest` para las opciones válidas de cada ítem `q01`…`q13`)

Fuente: Grable, J., & Lytton, R. H. (1999). *Financial Risk Tolerance Revisited:
The Development of a Risk Assessment Instrument*. Financial Services Review, 8(3), 163-181.
    """,
)
async def evaluar_perfil(body: RiskProfileRequest):
    enums = _parse_enums(body)
    result = _risk_agent.evaluar(
        **enums,
        usar_seleccion_dinamica=body.usar_seleccion_dinamica,
        lookback=body.lookback,
        precio_maximo=body.precio_maximo,
        volumen_minimo_usd=body.volumen_minimo_usd,
    )

    return RiskProfileResponse(
        perfil=result.perfil.value,
        score=result.score,
        score_raw=result.score_raw,
        descripcion_perfil=result.descripcion_perfil,
        dimensiones=[
            {
                "dimension": d.dimension,
                "respuesta": d.respuesta,
                "puntos": d.puntos,
                "max_puntos": d.max_puntos,
                "interpretacion": d.interpretacion,
            }
            for d in result.dimensiones
        ],
        sectores_recomendados=[
            {
                "sector": s.sector,
                "descripcion": s.descripcion,
                "etf": s.etf,
                "peso_target": s.peso_target,
                "retorno_hist": s.retorno_hist,
                "volatilidad_hist": s.volatilidad_hist,
                "sharpe_hist": s.sharpe_hist,
                "ranking_score":      s.ranking_score,
                "seleccion":          s.seleccion,
                "tipo":               s.tipo,
                "señal_sentimiento":  s.señal_sentimiento,
                "señal_prediccion":   s.señal_prediccion,
                "señal_sec":          s.señal_sec,
            }
            for s in result.sectores_recomendados
        ],
        tickers_recomendados=result.tickers_recomendados,
        pesos_sugeridos=result.pesos_sugeridos,
        risk_budget={
            "var_95_max": result.risk_budget["var_95_max"],
            "vol_anual_max": result.risk_budget["vol_anual_max"],
            "max_peso_activo": result.risk_budget["max_peso_activo"],
        },
        delta_aversion=result.delta_aversion,
        advertencia=result.advertencia,
        seleccion_dinamica=result.seleccion_dinamica,
        periodo_analisis=result.periodo_analisis,
        universo_evaluado=result.universo_evaluado,
        metodologia=result.metodologia,
        fecha_analisis=result.fecha_analisis,
    )


# ── Endpoint 2: análisis de portafolio basado en perfil ───────────────────

@router.post(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="Portafolio optimizado para un perfil de riesgo",
    description="""
Dado un perfil de riesgo previamente evaluado, lanza el pipeline multiagente
completo sobre los ETFs sectoriales recomendados para ese perfil y devuelve
el análisis de portafolio completo: Markowitz (máx Sharpe + mín varianza),
HRP, métricas de riesgo-retorno y alertas.

Los pesos iniciales corresponden a los pesos target del perfil; la
optimización de Markowitz/HRP puede ajustarlos según condiciones actuales.

**Perfiles válidos:** `muy_conservador` · `conservador` · `moderado` · `agresivo` · `muy_agresivo`
    """,
)
async def portfolio_por_perfil(body: RiskPortfolioRequest):
    perfil_val = body.perfil.strip().lower()
    try:
        perfil = PerfilRiesgo(perfil_val)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Perfil '{body.perfil}' no válido. Opciones: {[p.value for p in PerfilRiesgo]}",
        )

    if body.tickers and body.pesos and len(body.tickers) == len(body.pesos):
        tickers = [t.upper().strip() for t in body.tickers]
        pesos   = body.pesos
    else:
        sectores = _SECTORES_POR_PERFIL[perfil]
        tickers  = [s.etf for s in sectores]
        pesos    = [s.peso_target for s in sectores]

    loop = asyncio.get_event_loop()
    try:
        max_w = _RISK_BUDGET[perfil]["max_peso_activo"]
        delta_perfil = _DELTA_AVERSION_POR_PERFIL[perfil]
        result = await loop.run_in_executor(
            None,
            partial(
                _portfolio_agent.analizar_portafolio,
                tickers=tickers,
                weights=pesos,
                forzar_actualizacion=body.forzar_actualizacion,
                max_weight=max_w,
                delta_aversion=delta_perfil,
            ),
        )
    except Exception as exc:
        logger.error(f"Error en portfolio por perfil '{perfil_val}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al analizar portafolio: {str(exc)}",
        )

    activos_schema = []
    for a in result.activos:
        rec = a.recommendation
        market = a.market
        sec = a.sec_data
        activos_schema.append(
            PortfolioAssetSchema(
                ticker=a.ticker,
                weight=round(a.weight, 4),
                price=a.price,
                expected_return=a.expected_return,
                volatility=a.volatility,
                recomendacion=rec.recomendacion if rec else "",
                tipo_recomendacion=rec.tipo if rec else "mantener",
                confianza=rec.confianza if rec else 0.5,
                senal_mercado=market.senal if market else "neutral",
                sentimiento=a.sentiment.sentimiento if a.sentiment else "neutral",
                fundamental_signal=sec.fundamental_signal if sec else "neutral",
                fundamental_score=getattr(sec, "fundamental_score", 0.0),
                variacion_pct=a.prediction.variacion_pct if a.prediction else 0.0,
                bl_prior=a.bl_prior,
                bl_view=a.bl_view,
                bl_posterior=a.bl_posterior,
                view_confidence=a.view_confidence,
            )
        )

    m = result.metricas
    o = result.optimizacion

    return PortfolioResponse(
        tickers=result.tickers,
        weights=result.weights,
        activos=activos_schema,
        metricas=PortfolioMetricsSchema(
            expected_return=m.expected_return,
            volatility=m.volatility,
            sharpe_ratio=m.sharpe_ratio,
            var_95=m.var_95,
            var_99=m.var_99,
            diversification_ratio=m.diversification_ratio,
            correlation_matrix=m.correlation_matrix,
            num_activos=m.num_activos,
            beta_portfolio=m.beta_portfolio,
        ),
        optimizacion=PortfolioOptimizationSchema(
            max_sharpe_weights=o.max_sharpe_weights,
            max_sharpe_return=o.max_sharpe_return,
            max_sharpe_volatility=o.max_sharpe_volatility,
            max_sharpe_sharpe=o.max_sharpe_sharpe,
            min_variance_weights=o.min_variance_weights,
            min_variance_return=o.min_variance_return,
            min_variance_volatility=o.min_variance_volatility,
            efficient_frontier=[
                EfficientFrontierPointSchema(
                    weights=fp.weights,
                    expected_return=fp.expected_return,
                    volatility=fp.volatility,
                    sharpe=fp.sharpe,
                )
                for fp in o.efficient_frontier
            ],
            disponible=o.disponible,
            hrp_weights=o.hrp_weights,
            hrp_return=o.hrp_return,
            hrp_volatility=o.hrp_volatility,
            hrp_sharpe=o.hrp_sharpe,
            delta_aversion=o.delta_aversion,
            tau=o.tau,
            fuente_delta=o.fuente_delta,
        ),
        recomendacion_portafolio=result.recomendacion_portafolio,
        alertas=result.alertas,
        fecha_analisis=result.fecha_analisis,
    )
