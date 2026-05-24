"""
Router de Backtesting — POST /backtest/{ticker}
"""
import logging
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..agents import ModelAgent
from ..agents.backtest_agent import BacktestAgent, BacktestResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtesting"])

_model_agent = ModelAgent()
_backtest_agent = BacktestAgent(model_agent=_model_agent)
_executor = ThreadPoolExecutor(max_workers=2)


def _serialize(result: BacktestResult) -> dict:
    def m(obj):
        return {k: v for k, v in obj.__dict__.items()}
    return {
        "ticker": result.ticker,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_cash": result.initial_cash,
        "final_value": result.final_value,
        "benchmark_final": result.benchmark_final,
        "signals_generated": result.signals_generated,
        "buy_signals": result.buy_signals,
        "sell_signals": result.sell_signals,
        "metrics": m(result.metrics),
        "benchmark_metrics": m(result.benchmark_metrics),
        "trades": result.trades,
        "equity_curve": result.equity_curve,
    }


@router.post(
    "/{ticker}",
    summary="Backtest walk-forward con señales ML",
    description=(
        "Ejecuta un backtest walk-forward para el ticker indicado. "
        "Reentrena el ensemble de ML cada trimestre (63 días) con una ventana de "
        "504 días y simula las operaciones con Backtrader. "
        "Devuelve métricas (CAGR, Sharpe, Sortino, max drawdown, alpha, beta) "
        "y la curva de equity vs benchmark buy & hold."
    ),
)
async def run_backtest(
    ticker: str,
    years: int = Query(3, ge=1, le=5, description="Período de backtest en años (1-5)"),
    initial_cash: float = Query(10000.0, ge=1000.0, description="Capital inicial"),
):
    ticker = ticker.upper().strip()
    logger.info(f"[BacktestRouter] {ticker} | {years} años | capital={initial_cash}")

    loop = get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: _backtest_agent.run_backtest(ticker, years=years, initial_cash=initial_cash),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[BacktestRouter] Error en {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Error en backtest: {str(e)}")

    return JSONResponse(content=_serialize(result))
