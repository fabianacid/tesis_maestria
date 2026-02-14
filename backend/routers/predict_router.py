"""
Router de Predicción

Este módulo implementa el endpoint principal del sistema multiagente:
- GET /predict/{ticker}: Ejecuta el pipeline completo de análisis

El pipeline orquesta los cinco agentes del sistema:
1. MarketAgent: Obtiene datos de mercado
2. ModelAgent: Genera predicción
3. SentimentAgent: Analiza sentimiento
4. RecommendationAgent: Genera recomendación
5. AlertAgent: Evalúa umbrales

El resultado es una respuesta unificada con toda la información
del análisis, incluyendo explicaciones para cumplir con XAI.
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db, Usuario, MetricaModelo
from ..schemas import (
    PredictionResponse,
    MarketDataResponse,
    SentimentResponse,
    RecommendationResponse,
    AlertRealtimeResponse
)
from ..auth import get_current_active_user, get_optional_current_user
from ..config import settings
from ..agents import (
    MarketAgent,
    ModelAgent,
    SentimentAgent,
    RecommendationAgent,
    AlertAgent
)

# Configuración de logging
logger = logging.getLogger(__name__)

# Crear router con prefijo y tags
router = APIRouter(
    prefix="/predict",
    tags=["Predicción"]
)

# Instanciar agentes (singleton pattern)
# Usar 1 año de datos para tener suficientes features para el ensemble ML
market_agent = MarketAgent(ventana_ma=20, periodo_historico="1y")
model_agent = ModelAgent(ventana_entrenamiento=252)  # 1 año completo para mejor precisión
sentiment_agent = SentimentAgent()
recommendation_agent = RecommendationAgent()
alert_agent = AlertAgent(
    umbral_warning=settings.ALERT_THRESHOLD_WARNING,
    umbral_critical=settings.ALERT_THRESHOLD_CRITICAL
)


@router.get(
    "/{ticker}",
    response_model=PredictionResponse,
    summary="Análisis completo de activo",
    description="""
    Ejecuta el pipeline completo del sistema multiagente para un activo financiero.

    El análisis incluye:
    - Datos de mercado actualizados
    - Predicción de precio
    - Análisis de sentimiento
    - Recomendación integrada
    - Evaluación de alertas
    """
)
async def predict_ticker(
    ticker: str,
    forzar_actualizacion: bool = Query(
        False,
        description="Forzar actualización de datos ignorando caché"
    ),
    umbral_warning: float = Query(
        None,
        description="Umbral personalizado para alertas de advertencia (%)",
        ge=0.1,
        le=20.0
    ),
    umbral_critical: float = Query(
        None,
        description="Umbral personalizado para alertas críticas (%)",
        ge=0.5,
        le=30.0
    ),
    db: Session = Depends(get_db),
    current_user: Optional[Usuario] = Depends(get_optional_current_user)
):
    """
    Ejecuta análisis completo para un ticker.

    Este endpoint orquesta todos los agentes del sistema multiagente:

    1. **MarketAgent**: Obtiene datos de yfinance, calcula indicadores
    2. **ModelAgent**: Entrena modelo y genera predicción
    3. **SentimentAgent**: Analiza sentimiento del mercado
    4. **RecommendationAgent**: Integra señales y genera recomendación
    5. **AlertAgent**: Evalúa umbrales y genera alertas si corresponde

    Args:
        ticker: Símbolo del activo (ej: AAPL, GOOGL, MSFT)
        forzar_actualizacion: Ignorar caché de datos
        db: Sesión de base de datos
        current_user: Usuario autenticado (opcional)

    Returns:
        PredictionResponse: Respuesta completa con todos los análisis

    Raises:
        HTTPException 400: Si el ticker es inválido
        HTTPException 500: Si hay error en el procesamiento
    """
    ticker = ticker.upper().strip()

    # Validar ticker
    if not ticker or len(ticker) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ticker inválido"
        )

    logger.info(f"Iniciando análisis para {ticker}")

    try:
        # ========================================
        # PASO 1: Agente de Mercado
        # ========================================
        market_data = market_agent.obtener_datos(
            ticker,
            forzar_actualizacion=forzar_actualizacion
        )

        if market_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticker '{ticker}' no encontrado en Yahoo Finance. Verifica que el símbolo sea correcto y que el activo esté listado."
            )

        logger.info(f"[{ticker}] MarketAgent: Datos obtenidos correctamente")

        # ========================================
        # PASO 2: Agente de Modelo
        # ========================================
        prediction = model_agent.predecir(
            precios=market_data.precios,
            ticker=ticker
        )

        if prediction is None:
            # Usar valores por defecto si falla la predicción
            precio_predicho = market_data.ultimo_precio
            variacion_pct = 0.0
            rmse = mape = mae = r2 = 0.0
            accuracy = precision = recall = f1 = auc = 0.0
            parametros = {}
            modelos_usados = {}
            logger.warning(f"[{ticker}] ModelAgent: Predicción no disponible")
        else:
            precio_predicho = prediction.precio_predicho
            variacion_pct = prediction.variacion_pct
            rmse = prediction.rmse
            mape = prediction.mape
            mae = prediction.mae
            r2 = prediction.metricas_completas.r2
            # Métricas de clasificación
            accuracy = prediction.metricas_completas.accuracy
            precision = prediction.metricas_completas.precision
            recall = prediction.metricas_completas.recall
            f1 = prediction.metricas_completas.f1
            auc = prediction.metricas_completas.auc
            parametros = prediction.parametros
            # Incluir predicciones de cada modelo
            modelos_usados = {
                "predicciones": prediction.predicciones_modelos,
                "pesos": prediction.pesos_ensemble
            }
            logger.info(f"[{ticker}] ModelAgent: Predicción generada")

            # Persistir métricas si hay usuario autenticado
            if current_user:
                _guardar_metricas(db, current_user.id, ticker, prediction)

        # ========================================
        # PASO 3: Agente de Sentimiento
        # ========================================
        sentiment = sentiment_agent.analizar(ticker)
        logger.info(f"[{ticker}] SentimentAgent: Sentimiento={sentiment.sentimiento}")

        # ========================================
        # PASO 4: Agente de Recomendación
        # ========================================
        # Pasar TODOS los datos del análisis técnico al agente de recomendación
        recommendation = recommendation_agent.generar_recomendacion(
            ticker=ticker,
            senal_mercado=market_data.senal,
            variacion_pct=variacion_pct,
            sentimiento=sentiment.sentimiento,
            confianza_sentimiento=sentiment.confianza,
            ultimo_precio=market_data.ultimo_precio,
            # Pasar datos técnicos reales del MarketAgent
            volatilidad=market_data.indicators.atr if market_data.indicators.atr > 0 else 2.0,
            market_regime=market_data.signal_analysis.market_regime.value if hasattr(market_data.signal_analysis.market_regime, 'value') else str(market_data.signal_analysis.market_regime),
            rsi=market_data.indicators.rsi,
            macd_signal=market_data.indicators.macd_signal,
            # Calcular volume_ratio basado en MFI o usar valor por defecto
            volume_ratio=market_data.indicators.mfi / 50.0 if market_data.indicators.mfi > 0 else 1.0,
            prediction_confidence=prediction.confianza if prediction else 0.5,
            sentiment_score=sentiment.score if hasattr(sentiment, 'score') else 0.0
        )
        logger.info(f"[{ticker}] RecommendationAgent: {recommendation.tipo}")

        # ========================================
        # PASO 5: Agente de Alertas
        # ========================================
        alert_result = alert_agent.evaluar(
            ticker=ticker,
            variacion_pct=variacion_pct,
            precio_actual=market_data.ultimo_precio,
            precio_predicho=precio_predicho,
            contexto={
                "senal_mercado": market_data.senal,
                "sentimiento": sentiment.sentimiento,
                "recomendacion": recommendation.tipo
            },
            # Pasar umbrales personalizados si fueron proporcionados
            umbral_warning_custom=umbral_warning,
            umbral_critical_custom=umbral_critical
        )

        # Persistir alerta si corresponde y hay usuario
        if alert_result.debe_alertar and current_user:
            alert_agent.persistir_alerta(db, alert_result, current_user.id)
            logger.info(f"[{ticker}] AlertAgent: Alerta persistida")
        else:
            logger.info(f"[{ticker}] AlertAgent: Sin alerta")

        # ========================================
        # Construir respuesta unificada
        # ========================================
        response = PredictionResponse(
            ticker=ticker,
            fecha_analisis=datetime.now(),
            mercado=MarketDataResponse(
                ticker=ticker,
                ultimo_precio=market_data.ultimo_precio,
                precio_anterior=market_data.precio_anterior,
                variacion_diaria=market_data.variacion_diaria,
                media_movil_20=market_data.media_movil_20,
                senal=market_data.senal,
                fecha_actualizacion=market_data.fecha_actualizacion
            ),
            prediccion={
                "precio_predicho": precio_predicho,
                "variacion_pct": variacion_pct,
                "modelo": "ensemble_classification",
                "metricas": {
                    # Métricas de clasificación
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc": auc,
                    # Legacy para compatibilidad
                    "rmse": rmse,
                    "mape": mape,
                    "mae": mae,
                    "r2": r2
                },
                "parametros": parametros,
                "modelos_detalle": modelos_usados
            },
            sentimiento=SentimentResponse(
                ticker=ticker,
                sentimiento=sentiment.sentimiento,
                confianza=sentiment.confianza,
                score=sentiment.score,
                fuente=sentiment.fuente,
                explicacion_simple=sentiment.explicacion_simple,
                que_significa=sentiment.que_significa,
                como_se_calcula=sentiment.como_se_calcula,
                icono=sentiment.icono
            ),
            recomendacion=RecommendationResponse(
                ticker=ticker,
                recomendacion=recommendation.recomendacion,
                tipo=recommendation.tipo,
                confianza=recommendation.confianza,
                razon=recommendation.razon,
                factores=recommendation.factores,
                accion_sugerida=recommendation.accion_sugerida,
                explicacion_simple=recommendation.explicacion_simple,
                nivel_riesgo_simple=recommendation.nivel_riesgo_simple,
                porque_esta_recomendacion=recommendation.porque_esta_recomendacion,
                icono=recommendation.icono
            ),
            alerta=AlertRealtimeResponse(
                ticker=ticker,
                tiene_alerta=alert_result.debe_alertar,
                nivel=alert_result.nivel.value,
                mensaje=alert_result.mensaje,
                variacion_pct=alert_result.variacion_pct,
                umbral_superado=alert_result.umbral_superado
            )
        )

        logger.info(f"Análisis completado para {ticker}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en análisis de {ticker}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando análisis para {ticker}"
        )


@router.get(
    "/{ticker}/market",
    response_model=MarketDataResponse,
    summary="Solo datos de mercado",
    description="Obtiene únicamente los datos de mercado sin predicción."
)
async def get_market_data(
    ticker: str,
    forzar_actualizacion: bool = False
):
    """
    Obtiene solo datos de mercado para un ticker.

    Endpoint más ligero que solo ejecuta el MarketAgent.

    Args:
        ticker: Símbolo del activo
        forzar_actualizacion: Ignorar caché

    Returns:
        MarketDataResponse: Datos de mercado
    """
    ticker = ticker.upper().strip()

    market_data = market_agent.obtener_datos(
        ticker,
        forzar_actualizacion=forzar_actualizacion
    )

    if market_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{ticker}' no encontrado en Yahoo Finance. Verifica que el símbolo sea correcto y que el activo esté listado."
        )

    return MarketDataResponse(
        ticker=ticker,
        ultimo_precio=market_data.ultimo_precio,
        precio_anterior=market_data.precio_anterior,
        variacion_diaria=market_data.variacion_diaria,
        media_movil_20=market_data.media_movil_20,
        senal=market_data.senal,
        fecha_actualizacion=market_data.fecha_actualizacion
    )


@router.get(
    "/{ticker}/sentiment",
    response_model=SentimentResponse,
    summary="Solo análisis de sentimiento",
    description="Obtiene únicamente el análisis de sentimiento."
)
async def get_sentiment(ticker: str):
    """
    Obtiene solo análisis de sentimiento para un ticker.

    Args:
        ticker: Símbolo del activo

    Returns:
        SentimentResponse: Análisis de sentimiento
    """
    ticker = ticker.upper().strip()

    sentiment = sentiment_agent.analizar(ticker)

    return SentimentResponse(
        ticker=ticker,
        sentimiento=sentiment.sentimiento,
        confianza=sentiment.confianza,
        score=sentiment.score,
        fuente=sentiment.fuente,
        explicacion_simple=sentiment.explicacion_simple,
        que_significa=sentiment.que_significa,
        icono=sentiment.icono
    )


def _guardar_metricas(
    db: Session,
    usuario_id: int,
    ticker: str,
    prediction
) -> None:
    """
    Guarda métricas del modelo en la base de datos.

    Args:
        db: Sesión de base de datos
        usuario_id: ID del usuario
        ticker: Símbolo del activo
        prediction: Resultado de predicción
    """
    try:
        metrica = MetricaModelo(
            usuario_id=usuario_id,
            ticker=ticker,
            modelo=prediction.modelo,
            rmse=prediction.rmse,
            mape=prediction.mape,
            mae=prediction.mae
        )

        db.add(metrica)
        db.commit()
        logger.info(f"Métricas guardadas para {ticker}")

    except Exception as e:
        db.rollback()
        logger.error(f"Error guardando métricas: {str(e)}")
