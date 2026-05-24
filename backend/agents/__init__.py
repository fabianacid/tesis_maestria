"""
Sistema Multiagente para Análisis Financiero

Agentes especializados:
- MarketAgent: Datos de mercado e indicadores técnicos
- ModelAgent: Predicción ML (ensemble)
- SentimentAgent: Análisis NLP de sentimiento
- RecommendationAgent: Decisión multi-factor
- AlertAgent: Evaluación de umbrales
- SECAgent: Datos fundamentales y filings SEC EDGAR
- PortfolioAgent: Análisis y optimización de portafolio
- BacktestAgent: Backtesting walk-forward con Backtrader
"""
from .market_agent import MarketAgent, MarketData
from .model_agent import ModelAgent, PredictionResult
from .sentiment_agent import SentimentAgent, SentimentResult
from .recommendation_agent import RecommendationAgent, RecommendationResult
from .alert_agent import AlertAgent, AlertResult, NivelAlerta
from .sec_agent import SECAgent, SECData, FinancialRatios, BalanceSummary, SECFiling
from .portfolio_agent import PortfolioAgent, PortfolioResult, PortfolioMetrics, PortfolioOptimization, AssetAnalysis
from .backtest_agent import BacktestAgent, BacktestResult, BacktestMetrics

__all__ = [
    "MarketAgent", "ModelAgent", "SentimentAgent",
    "RecommendationAgent", "AlertAgent",
    "SECAgent", "PortfolioAgent", "BacktestAgent",
    "MarketData", "PredictionResult", "SentimentResult",
    "RecommendationResult", "AlertResult", "NivelAlerta",
    "SECData", "FinancialRatios", "BalanceSummary", "SECFiling",
    "PortfolioResult", "PortfolioMetrics", "PortfolioOptimization", "AssetAnalysis",
    "BacktestResult", "BacktestMetrics",
]
