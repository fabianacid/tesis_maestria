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
    GrupoEdad, HorizonteInversion, EstabilidadIngresos,
    ToleranciaPerdidas, ExperienciaInversora, ObjetivoFinanciero,
    _SECTORES_POR_PERFIL, _RISK_BUDGET,
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
_ENUM_MAP = {
    "edad":        {e.value: e for e in GrupoEdad},
    "horizonte":   {e.value: e for e in HorizonteInversion},
    "ingresos":    {e.value: e for e in EstabilidadIngresos},
    "perdidas":    {e.value: e for e in ToleranciaPerdidas},
    "experiencia": {e.value: e for e in ExperienciaInversora},
    "objetivo":    {e.value: e for e in ObjetivoFinanciero},
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
Recibe las respuestas al cuestionario de 6 dimensiones y devuelve:
- **Perfil de riesgo** (muy_conservador, conservador, moderado, agresivo, muy_agresivo)
- **Score 0-100** con desglose e interpretación de cada dimensión
- **Sectores recomendados** con ETF representativo y peso sugerido
- **Presupuesto de riesgo**: VaR 95% máximo tolerado y volatilidad anual objetivo

### Cuestionario

| Campo | Valores válidos |
|---|---|
| `edad` | `menor_35` · `36_45` · `46_55` · `56_65` · `mayor_65` |
| `horizonte` | `menos_1_anio` · `1_3_anios` · `3_5_anios` · `5_10_anios` · `mas_10_anios` |
| `ingresos` | `muy_inestable` · `inestable` · `moderada` · `estable` · `muy_estable` |
| `perdidas` | `vender_todo` · `vender_mayoria` · `mantener` · `comprar_poco` · `comprar_mucho` |
| `experiencia` | `ninguna` · `basica` · `intermedia` · `avanzada` · `experto` |
| `objetivo` | `preservacion` · `ingreso` · `crecimiento_ingreso` · `crecimiento` · `especulacion` |
    """,
)
async def evaluar_perfil(body: RiskProfileRequest):
    enums = _parse_enums(body)
    result = _risk_agent.evaluar(
        **enums,
        usar_seleccion_dinamica=body.usar_seleccion_dinamica,
        lookback=body.lookback,
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
        advertencia=result.advertencia,
        seleccion_dinamica=result.seleccion_dinamica,
        periodo_analisis=result.periodo_analisis,
        universo_evaluado=result.universo_evaluado,
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
        result = await loop.run_in_executor(
            None,
            partial(
                _portfolio_agent.analizar_portafolio,
                tickers,
                pesos,
                body.forzar_actualizacion,
                max_w,
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
        ),
        recomendacion_portafolio=result.recomendacion_portafolio,
        alertas=result.alertas,
        fecha_analisis=result.fecha_analisis,
    )
