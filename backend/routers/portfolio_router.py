"""
Router de Portafolio

Endpoint principal:
  POST /portfolio/analyze — análisis completo de un portafolio de activos

El pipeline para cada activo es idéntico al de /predict/{ticker}
(MarketAgent → ModelAgent → SentimentAgent → RecommendationAgent + SECAgent),
y se ejecuta en paralelo. Luego se calculan métricas de portafolio
y optimización de Markowitz.
"""
import asyncio
import logging
from datetime import datetime
from functools import partial
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db, Usuario
from ..schemas import (
    PortfolioRequest,
    PortfolioResponse,
    PortfolioMetricsSchema,
    PortfolioOptimizationSchema,
    EfficientFrontierPointSchema,
    PortfolioAssetSchema,
)
from ..auth import get_optional_current_user
from ..config import settings
from ..agents import (
    MarketAgent, ModelAgent, SentimentAgent,
    RecommendationAgent, AlertAgent,
    SECAgent, PortfolioAgent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portafolio"])

# Singletons compartidos con predict_router para aprovechar caché
market_agent = MarketAgent(ventana_ma=20, periodo_historico="2y")
model_agent = ModelAgent(ventana_entrenamiento=504)
sentiment_agent = SentimentAgent()
recommendation_agent = RecommendationAgent()
alert_agent = AlertAgent(
    umbral_warning=settings.ALERT_THRESHOLD_WARNING,
    umbral_critical=settings.ALERT_THRESHOLD_CRITICAL,
)
sec_agent = SECAgent()
portfolio_agent = PortfolioAgent(
    market_agent=market_agent,
    model_agent=model_agent,
    sentiment_agent=sentiment_agent,
    recommendation_agent=recommendation_agent,
    sec_agent=sec_agent,
    alert_agent=alert_agent,
)


@router.post(
    "/analyze",
    response_model=PortfolioResponse,
    summary="Análisis completo de portafolio",
    description="""
    Analiza un portafolio de activos financieros.

    Para cada activo ejecuta el pipeline multiagente completo
    (datos de mercado, predicción ML, sentimiento, fundamentales SEC,
    recomendación). Luego calcula métricas de portafolio y
    optimización de Markowitz (máximo Sharpe y mínima varianza).

    Los pesos se normalizan automáticamente a suma=1.
    Se requieren mínimo 2 activos.
    """,
)
async def analyze_portfolio(
    body: PortfolioRequest,
    db: Session = Depends(get_db),
    current_user: Optional[Usuario] = Depends(get_optional_current_user),
):
    if len(body.tickers) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Se necesitan al menos 2 activos para el análisis de portafolio",
        )
    if len(body.tickers) != len(body.weights):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El número de tickers y pesos debe ser igual",
        )
    if any(w < 0 for w in body.weights):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Los pesos deben ser no negativos",
        )
    if len(body.tickers) > 15:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El portafolio no puede tener más de 15 activos",
        )

    tickers = [t.upper().strip() for t in body.tickers]
    logger.info(f"Iniciando análisis de portafolio: {tickers}")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                portfolio_agent.analizar_portafolio,
                tickers,
                body.weights,
                body.forzar_actualizacion,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error en análisis de portafolio {tickers}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando el portafolio: {str(exc)}",
        )

    # Serializar activos
    activos_schema = []
    for a in result.activos:
        activos_schema.append(PortfolioAssetSchema(
            ticker=a.ticker,
            weight=round(a.weight, 4),
            price=round(a.price, 4),
            expected_return=a.expected_return,
            volatility=a.volatility,
            recomendacion=a.recommendation.recomendacion,
            tipo_recomendacion=a.recommendation.tipo,
            confianza=a.recommendation.confianza,
            senal_mercado=a.market.senal,
            sentimiento=a.sentiment.sentimiento,
            fundamental_signal=a.sec_data.fundamental_signal,
            fundamental_score=a.sec_data.fundamental_score,
            variacion_pct=a.prediction.variacion_pct if a.prediction else 0.0,
        ))

    # Serializar métricas
    metricas_schema = PortfolioMetricsSchema(
        expected_return=result.metricas.expected_return,
        volatility=result.metricas.volatility,
        sharpe_ratio=result.metricas.sharpe_ratio,
        var_95=result.metricas.var_95,
        var_99=result.metricas.var_99,
        diversification_ratio=result.metricas.diversification_ratio,
        correlation_matrix=result.metricas.correlation_matrix,
        num_activos=result.metricas.num_activos,
        beta_portfolio=result.metricas.beta_portfolio,
    )

    # Serializar optimización
    frontier_schema = [
        EfficientFrontierPointSchema(
            weights=p.weights,
            expected_return=p.expected_return,
            volatility=p.volatility,
            sharpe=p.sharpe,
        )
        for p in result.optimizacion.efficient_frontier
    ]
    opt_schema = PortfolioOptimizationSchema(
        max_sharpe_weights=result.optimizacion.max_sharpe_weights,
        max_sharpe_return=result.optimizacion.max_sharpe_return,
        max_sharpe_volatility=result.optimizacion.max_sharpe_volatility,
        max_sharpe_sharpe=result.optimizacion.max_sharpe_sharpe,
        min_variance_weights=result.optimizacion.min_variance_weights,
        min_variance_return=result.optimizacion.min_variance_return,
        min_variance_volatility=result.optimizacion.min_variance_volatility,
        efficient_frontier=frontier_schema,
        disponible=result.optimizacion.disponible,
        hrp_weights=result.optimizacion.hrp_weights,
        hrp_return=result.optimizacion.hrp_return,
        hrp_volatility=result.optimizacion.hrp_volatility,
        hrp_sharpe=result.optimizacion.hrp_sharpe,
    )

    logger.info(f"Análisis de portafolio completado: {tickers}")
    return PortfolioResponse(
        tickers=result.tickers,
        weights=result.weights,
        activos=activos_schema,
        metricas=metricas_schema,
        optimizacion=opt_schema,
        recomendacion_portafolio=result.recomendacion_portafolio,
        alertas=result.alertas,
        fecha_analisis=result.fecha_analisis,
    )
