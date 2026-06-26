"""
Schemas Pydantic para Validación de Datos

Este módulo define los modelos de datos utilizados para
validar las entradas y salidas de la API REST, garantizando
la integridad y formato correcto de la información.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr


# ============================================================
# Schemas de Usuario y Autenticación
# ============================================================

class UserCreate(BaseModel):
    """Schema para creación de usuario."""
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: str = Field(..., min_length=6)
    rol: str = Field(default="analista")


class UserResponse(BaseModel):
    """Schema de respuesta para usuario (sin password)."""
    id: int
    username: str
    email: Optional[str] = None
    rol: str
    fecha_creacion: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema para token JWT de acceso."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema para datos contenidos en el token."""
    username: Optional[str] = None
    user_id: Optional[int] = None


class ForgotPasswordRequest(BaseModel):
    """Schema para solicitud de recuperación de contraseña."""
    email: EmailStr = Field(..., description="Email del usuario registrado")


class ResetPasswordRequest(BaseModel):
    """Schema para reseteo de contraseña con token."""
    token: str = Field(..., min_length=32, description="Token de reseteo recibido por email")
    new_password: str = Field(..., min_length=6, description="Nueva contraseña")


class ChangePasswordRequest(BaseModel):
    """Schema para cambio de contraseña autenticado."""
    current_password: str = Field(..., description="Contraseña actual")
    new_password: str = Field(..., min_length=6, description="Nueva contraseña")


class MessageResponse(BaseModel):
    """Schema para respuestas simples con mensaje."""
    message: str
    detail: Optional[str] = None


# ============================================================
# Schemas de Datos de Mercado
# ============================================================

class MarketDataResponse(BaseModel):
    """Schema de respuesta con datos de mercado."""
    ticker: str
    ultimo_precio: float
    precio_anterior: float
    variacion_diaria: float
    media_movil_20: float
    senal: str  # "alcista", "bajista", "neutral"
    fecha_actualizacion: datetime
    precios_recientes: List[float] = []
    fechas_recientes: List[str] = []


# ============================================================
# Schemas de Predicción
# ============================================================

class PredictionRequest(BaseModel):
    """Schema para solicitud de predicción."""
    ticker: str = Field(..., min_length=1, max_length=10)
    modelo: str = Field(default="regresion_lineal")
    ventana: int = Field(default=30, ge=10, le=365)


# ============================================================
# Schemas de Sentimiento
# ============================================================

class SentimentResponse(BaseModel):
    """Schema de respuesta del análisis de sentimiento."""
    ticker: str
    sentimiento: str  # "positivo", "negativo", "neutral"
    confianza: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=-1.0, le=1.0)
    fuente: str = "modelo_base"
    # Explicaciones para inversor minorista
    explicacion_simple: str = ""
    que_significa: str = ""
    como_se_calcula: str = ""
    icono: str = ""


# ============================================================
# Schemas de Recomendación
# ============================================================

class RecommendationResponse(BaseModel):
    """Schema de respuesta de recomendación."""
    ticker: str
    recomendacion: str
    tipo: str
    razon: str
    confianza: float
    factores: dict
    # Explicaciones para inversor minorista
    accion_sugerida: str = ""
    explicacion_simple: str = ""
    nivel_riesgo_simple: str = ""
    porque_esta_recomendacion: str = ""
    icono: str = ""


# ============================================================
# Schemas de Alertas
# ============================================================

class AlertRealtimeResponse(BaseModel):
    """Schema de respuesta para alertas en tiempo real (endpoint de predicción)."""
    ticker: str
    tiene_alerta: bool
    nivel: str
    mensaje: str
    variacion_pct: float
    umbral_superado: float


class AlertCreate(BaseModel):
    """Schema para creación de alerta."""
    ticker: str
    severidad: str = Field(..., pattern="^(info|advertencia|critica)$")
    mensaje: str
    senal: Optional[float] = None
    recomendacion: Optional[str] = None
    variacion_pct: Optional[float] = None


class AlertResponse(BaseModel):
    """Schema de respuesta para alertas."""
    id: int
    usuario_id: Optional[int] = None
    ticker: str
    tipo: str  # info, warning, critical
    mensaje: str
    variacion_pct: Optional[float] = None
    precio_actual: Optional[float] = None
    precio_predicho: Optional[float] = None
    leida: bool = False
    fecha_creacion: datetime

    class Config:
        from_attributes = True


class AlertListResponse(BaseModel):
    """Schema para lista de alertas."""
    total: int
    alertas: List[AlertResponse]
    pagina_actual: int = 1
    total_paginas: int = 1


# ============================================================
# Schemas de Métricas
# ============================================================

class MetricaResponse(BaseModel):
    """Schema de respuesta para métricas del modelo."""
    id: int
    ticker: str
    modelo: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc: Optional[float] = None
    fecha: datetime

    class Config:
        from_attributes = True


# ============================================================
# Schemas Generales
# ============================================================

class StatusResponse(BaseModel):
    """Schema para respuesta de estado de la aplicación."""
    status: str
    app_name: str
    version: str
    message: str


class ErrorResponse(BaseModel):
    """Schema para respuestas de error."""
    detail: str
    error_code: Optional[str] = None


class PredictionResponse(BaseModel):
    """Schema de respuesta completa de predicción."""
    ticker: str
    fecha_analisis: datetime
    mercado: MarketDataResponse
    prediccion: dict
    sentimiento: SentimentResponse
    recomendacion: RecommendationResponse
    alerta: AlertRealtimeResponse
    sec_data: Optional[dict] = None


# ============================================================
# Schemas SEC / Fundamental
# ============================================================

class SECFilingSchema(BaseModel):
    form_type: str
    filing_date: str
    description: str
    accession_number: str


class FinancialRatiosSchema(BaseModel):
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    market_cap: Optional[float] = None
    health_score: float = 5.0
    health_label: str = "neutral"


class BalanceSummarySchema(BaseModel):
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    revenue_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None


class SECDataResponse(BaseModel):
    ticker: str
    company_name: str
    ratios: FinancialRatiosSchema
    balance: BalanceSummarySchema
    recent_filings: List[SECFilingSchema]
    fundamental_signal: str
    fundamental_score: float
    resumen: str
    fecha_actualizacion: datetime
    disponible: bool = True
    error_msg: str = ""


# ============================================================
# Schemas de Portafolio
# ============================================================

class PortfolioRequest(BaseModel):
    """Request para análisis de portafolio."""
    tickers: List[str] = Field(..., min_length=2, description="Lista de tickers (mínimo 2)")
    weights: List[float] = Field(..., min_length=2, description="Pesos de cada activo (se normalizan)")
    forzar_actualizacion: bool = Field(default=False)


class PortfolioMetricsSchema(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    diversification_ratio: float
    correlation_matrix: Dict[str, Dict[str, float]]
    num_activos: int
    beta_portfolio: Optional[float] = None


class EfficientFrontierPointSchema(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float


class PortfolioOptimizationSchema(BaseModel):
    max_sharpe_weights: Dict[str, float]
    max_sharpe_return: float
    max_sharpe_volatility: float
    max_sharpe_sharpe: float
    min_variance_weights: Dict[str, float]
    min_variance_return: float
    min_variance_volatility: float
    efficient_frontier: List[EfficientFrontierPointSchema]
    disponible: bool
    hrp_weights: Dict[str, float] = {}
    hrp_return: float = 0.0
    hrp_volatility: float = 0.0
    hrp_sharpe: float = 0.0


class PortfolioAssetSchema(BaseModel):
    ticker: str
    weight: float
    price: float
    expected_return: float
    volatility: float
    recomendacion: str
    tipo_recomendacion: str
    confianza: float
    senal_mercado: str
    sentimiento: str
    fundamental_signal: str
    fundamental_score: float
    variacion_pct: float


class PortfolioResponse(BaseModel):
    """Schema de respuesta completa del análisis de portafolio."""
    tickers: List[str]
    weights: Dict[str, float]
    activos: List[PortfolioAssetSchema]
    metricas: PortfolioMetricsSchema
    optimizacion: PortfolioOptimizationSchema
    recomendacion_portafolio: str
    alertas: List[Dict[str, Any]]
    fecha_analisis: datetime


# ============================================================
# Schemas de Perfil de Riesgo
# ============================================================

class RiskProfileRequest(BaseModel):
    """Cuestionario de perfil de riesgo del inversor (6 dimensiones)."""
    edad: str = Field(
        ...,
        description="Grupo etario: menor_35 | 36_45 | 46_55 | 56_65 | mayor_65",
    )
    horizonte: str = Field(
        ...,
        description="Horizonte de inversión: menos_1_anio | 1_3_anios | 3_5_anios | 5_10_anios | mas_10_anios",
    )
    ingresos: str = Field(
        ...,
        description="Estabilidad de ingresos: muy_inestable | inestable | moderada | estable | muy_estable",
    )
    perdidas: str = Field(
        ...,
        description="Reacción ante caída del 20%: vender_todo | vender_mayoria | mantener | comprar_poco | comprar_mucho",
    )
    experiencia: str = Field(
        ...,
        description="Experiencia inversora: ninguna | basica | intermedia | avanzada | experto",
    )
    objetivo: str = Field(
        ...,
        description="Objetivo financiero: preservacion | ingreso | crecimiento_ingreso | crecimiento | especulacion",
    )
    usar_seleccion_dinamica: bool = Field(
        default=True,
        description="Si True, selecciona los mejores ETFs del universo usando datos históricos reales.",
    )
    lookback: str = Field(
        default="1y",
        description="Período histórico para selección dinámica: 6mo | 1y | 2y",
    )


class DimensionScoreSchema(BaseModel):
    dimension: str
    respuesta: str
    puntos: int
    max_puntos: int
    interpretacion: str


class SectorRecomendadoSchema(BaseModel):
    sector: str
    descripcion: str
    etf: str
    peso_target: float
    retorno_hist: Optional[float] = None
    volatilidad_hist: Optional[float] = None
    sharpe_hist: Optional[float] = None
    ranking_score: Optional[float] = None
    seleccion: str = "predefinida"


class RiskBudgetSchema(BaseModel):
    var_95_max: float
    vol_anual_max: float
    max_peso_activo: float


class RiskProfileResponse(BaseModel):
    """Resultado completo del análisis de perfil de riesgo."""
    perfil: str
    score: float
    score_raw: int
    descripcion_perfil: str
    dimensiones: List[DimensionScoreSchema]
    sectores_recomendados: List[SectorRecomendadoSchema]
    tickers_recomendados: List[str]
    pesos_sugeridos: List[float]
    risk_budget: RiskBudgetSchema
    advertencia: str
    seleccion_dinamica: bool = False
    periodo_analisis: str = ""
    universo_evaluado: int = 0


class RiskPortfolioRequest(BaseModel):
    """Solicita el análisis de portafolio basado en un perfil de riesgo previamente calculado."""
    perfil: str = Field(
        ...,
        description="Perfil de riesgo: muy_conservador | conservador | moderado | agresivo | muy_agresivo",
    )
    forzar_actualizacion: bool = Field(default=False)
    tickers: Optional[List[str]] = Field(
        default=None,
        description="Tickers a analizar. Si se omite, usa los predefinidos del perfil.",
    )
    pesos: Optional[List[float]] = Field(
        default=None,
        description="Pesos correspondientes a tickers. Si se omite, usa los predefinidos.",
    )
