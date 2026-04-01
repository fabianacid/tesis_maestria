"""
Schemas Pydantic para Validación de Datos

Este módulo define los modelos de datos utilizados para
validar las entradas y salidas de la API REST, garantizando
la integridad y formato correcto de la información.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List
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
