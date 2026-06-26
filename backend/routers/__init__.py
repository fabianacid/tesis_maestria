"""
Routers de la API REST
"""
from .auth_router import router as auth_router
from .predict_router import router as predict_router
from .alerts_router import router as alerts_router
from .portfolio_router import router as portfolio_router
from .backtest_router import router as backtest_router
from .risk_router import router as risk_router

__all__ = ["auth_router", "predict_router", "alerts_router", "portfolio_router", "backtest_router", "risk_router"]
