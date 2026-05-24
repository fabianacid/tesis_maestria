"""
Aplicación Principal FastAPI

Este módulo configura e inicializa la aplicación FastAPI del
Sistema Multiagente de Análisis y Optimización de Portafolios Financieros.

Características:
- Configuración de CORS para acceso desde el dashboard
- Registro de routers para cada módulo funcional
- Middleware de logging para trazabilidad
- Endpoints de salud y estado del sistema
- Inicialización de base de datos

La aplicación expone:
- /auth/*: Endpoints de autenticación
- /predict/*: Endpoints del sistema multiagente
- /alerts/*: Gestión de alertas
- /portfolio/*: Análisis y optimización de portafolios
- /health: Estado del sistema
- /docs: Documentación Swagger UI
"""
import logging
import logging.handlers
import os
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database import init_db
from .routers import auth_router, predict_router, alerts_router, portfolio_router, backtest_router

# Configuración de logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.TimedRotatingFileHandler(
            os.path.join(LOG_DIR, "app.log"),
            when="midnight",
            backupCount=7,
            encoding="utf-8"
        )
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación.

    - Startup: Inicializa base de datos
    - Shutdown: Limpieza de recursos
    """
    # Startup
    logger.info("Iniciando aplicación...")
    init_db()
    logger.info("Base de datos inicializada")
    yield
    # Shutdown
    logger.info("Cerrando aplicación...")


# Crear instancia de FastAPI
app = FastAPI(
    title="Sistema Multiagente de Análisis y Optimización de Portafolios Financieros",
    description="""
    API REST del prototipo de sistema inteligente de seguimiento
    y alertas para activos financieros.

    ## Características

    * **Autenticación JWT** - Registro y login seguro
    * **Análisis Multiagente** - Pipeline de 5 agentes especializados
    * **Predicción** - Modelo de regresión lineal con métricas
    * **Alertas** - Sistema de umbrales configurables
    * **Explicabilidad** - Recomendaciones con justificación

    ## Agentes del Sistema

    1. **MarketAgent**: Obtención de datos de mercado
    2. **ModelAgent**: Predicción de precios
    3. **SentimentAgent**: Análisis de sentimiento
    4. **RecommendationAgent**: Generación de recomendaciones
    5. **AlertAgent**: Evaluación de umbrales

    ## Tecnologías

    - FastAPI + Pydantic
    - SQLAlchemy + SQLite
    - scikit-learn
    - yfinance
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware de logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware para logging de todas las requests.
    """
    start_time = datetime.now()

    # Procesar request
    response = await call_next(request)

    # Calcular tiempo de procesamiento
    process_time = (datetime.now() - start_time).total_seconds()

    # Log de la request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    return response


# Manejador global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Manejador global de excepciones no controladas.
    """
    logger.error(f"Error no controlado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Error interno del servidor",
            "detail": "Ha ocurrido un error inesperado"
        }
    )


# Registrar routers
app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(alerts_router)
app.include_router(portfolio_router)
app.include_router(backtest_router)


# Endpoints de estado
@app.get(
    "/",
    tags=["Estado"],
    summary="Raíz de la API",
    description="Endpoint de bienvenida con información básica."
)
async def root():
    """
    Endpoint raíz de la API.
    """
    return {
        "mensaje": "Sistema Multiagente de Análisis y Optimización de Portafolios Financieros",
        "version": "1.0.0",
        "docs": "/docs",
        "estado": "operativo"
    }


@app.get(
    "/health",
    tags=["Estado"],
    summary="Estado del sistema",
    description="Verifica el estado de salud de la aplicación."
)
async def health_check():
    """
    Endpoint de verificación de salud.

    Útil para monitoreo y balanceadores de carga.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "operational",
            "database": "operational",
            "agents": "operational"
        }
    }


@app.get(
    "/config",
    tags=["Estado"],
    summary="Configuración pública",
    description="Retorna configuración no sensible del sistema."
)
async def get_config():
    """
    Retorna configuración pública del sistema.

    No incluye información sensible como claves secretas.
    """
    return {
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "alert_thresholds": {
            "warning": settings.ALERT_THRESHOLD_WARNING,
            "critical": settings.ALERT_THRESHOLD_CRITICAL
        },
        "token_expiration_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES
    }


# Punto de entrada para desarrollo
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
