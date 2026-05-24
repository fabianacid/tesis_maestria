"""
BacktestAgent — Walk-forward backtesting con señales ML usando Backtrader.

Estrategia: cada STEP días (trimestral) reentrena el ensemble con la ventana
de TRAIN_WINDOW días anteriores y asigna la señal resultante (compra/venta/hold)
a los STEP días siguientes. Simula un gestor que revisa su modelo trimestralmente.

Comisión: 0.1% por operación. Tasa libre de riesgo: 4.5% anual.
"""
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import backtrader as bt
import backtrader.analyzers as btanalyzers

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────
@dataclass
class BacktestMetrics:
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    num_trades: int
    win_rate: float
    avg_trade_pct: float
    alpha: float
    beta: float


@dataclass
class BacktestResult:
    ticker: str
    start_date: str
    end_date: str
    initial_cash: float
    final_value: float
    benchmark_final: float
    metrics: BacktestMetrics
    benchmark_metrics: BacktestMetrics
    equity_curve: List[Dict]
    trades: List[Dict]
    signals_generated: int
    buy_signals: int
    sell_signals: int


# ─────────────────────────────────────────────
# Estrategia Backtrader
# ─────────────────────────────────────────────
class MLSignalStrategy(bt.Strategy):
    """Compra/vende según señales ML pre-computadas (1=compra, -1=vende, 0=hold)."""

    params = (
        ("signals", {}),
        ("stake_pct", 0.95),
    )

    def __init__(self):
        self.order = None
        self.trade_log = []
        self._entry_price = None
        self._entry_date = None

    def next(self):
        if self.order:
            return

        date_str = str(self.datas[0].datetime.date(0))
        signal = self.params.signals.get(date_str, 0)

        if signal == 1 and not self.position:
            cash = self.broker.getcash()
            price = self.data.close[0]
            size = int(cash * self.params.stake_pct / price)
            if size > 0:
                self.order = self.buy(size=size)
                self._entry_price = price
                self._entry_date = date_str

        elif signal == -1 and self.position:
            exit_price = self.data.close[0]
            pnl_pct = (exit_price / self._entry_price - 1) * 100 if self._entry_price else 0
            self.trade_log.append({
                "date_open": self._entry_date,
                "date_close": date_str,
                "entry_price": round(self._entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "result": "ganancia" if pnl_pct > 0 else "pérdida",
            })
            self.order = self.sell(size=self.position.size)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None


# ─────────────────────────────────────────────
# Agente
# ─────────────────────────────────────────────
class BacktestAgent:
    TRAIN_WINDOW = 504
    STEP = 63
    BUY_THRESHOLD = 0.54
    SELL_THRESHOLD = 0.46
    COMMISSION = 0.001
    RISK_FREE = 0.045

    def __init__(self, model_agent):
        self.model_agent = model_agent

    def _generate_signals(self, ticker: str, data: pd.DataFrame) -> dict:
        signals = {}
        n = len(data)
        if n < self.TRAIN_WINDOW + self.STEP:
            return signals

        total_steps = (n - self.TRAIN_WINDOW) // self.STEP
        logger.info(f"[BacktestAgent] {ticker}: {total_steps} períodos de reentrenamiento")

        for i, start in enumerate(range(0, n - self.TRAIN_WINDOW - self.STEP + 1, self.STEP)):
            end_train = start + self.TRAIN_WINDOW
            train_slice = data.iloc[start:end_train].copy()
            try:
                result = self.model_agent.predecir(
                    precios=train_slice,
                    ticker=ticker,
                    forzar_actualizacion=True,
                )
                if result is None:
                    continue
                prob = result.prob_subida
                sig = 1 if prob >= self.BUY_THRESHOLD else (-1 if prob <= self.SELL_THRESHOLD else 0)
                for d in data.index[end_train: end_train + self.STEP]:
                    signals[str(d.date())] = sig
                logger.info(
                    f"[BacktestAgent] {ticker} período {i+1}/{total_steps}: "
                    f"prob={prob:.3f} → {'COMPRA' if sig==1 else 'VENTA' if sig==-1 else 'HOLD'}"
                )
            except Exception as e:
                logger.warning(f"[BacktestAgent] {ticker} período {i+1}: {e}")

        return signals

    def _compute_metrics(self, equity: pd.Series, benchmark: pd.Series, trades: list) -> BacktestMetrics:
        returns = equity.pct_change().dropna()
        bench_ret = benchmark.pct_change().dropna()
        n_years = max(len(equity) / 252, 0.01)

        total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100

        rf_daily = self.RISK_FREE / 252
        excess = returns - rf_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
        downside = excess[excess < 0].std()
        sortino = float(excess.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0

        roll_max = equity.cummax()
        max_dd = float(((equity - roll_max) / roll_max).min() * 100)
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        wins = [t for t in trades if t.get("pnl_pct", 0) > 0]
        win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
        avg_trade = float(np.mean([t["pnl_pct"] for t in trades])) if trades else 0.0

        common = returns.index.intersection(bench_ret.index)
        if len(common) > 10:
            r, b = returns.loc[common].values, bench_ret.loc[common].values
            cov = np.cov(r, b)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0
            bench_ann = float(bench_ret.mean() * 252 * 100)
            alpha = cagr - (self.RISK_FREE * 100 + beta * (bench_ann - self.RISK_FREE * 100))
        else:
            beta, alpha = 1.0, 0.0

        return BacktestMetrics(
            total_return=round(total_ret, 2),
            cagr=round(cagr, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown=round(max_dd, 2),
            calmar_ratio=round(calmar, 3),
            num_trades=len(trades),
            win_rate=round(win_rate, 1),
            avg_trade_pct=round(avg_trade, 2),
            alpha=round(alpha, 2),
            beta=round(beta, 3),
        )

    def run_backtest(self, ticker: str, years: int = 3, initial_cash: float = 10_000.0) -> BacktestResult:
        end_date = datetime.today()
        start_data = end_date - timedelta(days=365 * (years + 2) + 30)
        start_bt = end_date - timedelta(days=365 * years)

        logger.info(f"[BacktestAgent] Descargando datos de {ticker}...")
        raw = yf.download(
            ticker,
            start=start_data.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if raw.empty:
            raise ValueError(f"No se encontraron datos para {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close"])

        if len(raw) < self.TRAIN_WINDOW + self.STEP:
            raise ValueError(f"Datos insuficientes: {len(raw)} días (mínimo {self.TRAIN_WINDOW + self.STEP})")

        logger.info(f"[BacktestAgent] Generando señales walk-forward para {ticker}...")
        all_signals = self._generate_signals(ticker, raw)

        bt_data = raw[raw.index >= start_bt.strftime("%Y-%m-%d")].copy()
        if bt_data.empty:
            raise ValueError("Sin datos en el período de backtest")

        bt_signals = {k: v for k, v in all_signals.items() if k >= start_bt.strftime("%Y-%m-%d")}
        buy_count = sum(1 for v in bt_signals.values() if v == 1)
        sell_count = sum(1 for v in bt_signals.values() if v == -1)

        # ── Backtrader ────────────────────────────────────────────────────
        feed = bt.feeds.PandasData(
            dataname=bt_data,
            datetime=None,
            open="Open", high="High", low="Low", close="Close",
            volume="Volume", openinterest=-1,
        )

        cerebro = bt.Cerebro(stdstats=False)
        cerebro.adddata(feed)
        cerebro.addstrategy(MLSignalStrategy, signals=bt_signals, stake_pct=0.95)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=self.COMMISSION)
        cerebro.addanalyzer(btanalyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days)

        results = cerebro.run()
        strat = results[0]
        final_value = cerebro.broker.getvalue()

        # ── Equity curve ─────────────────────────────────────────────────
        time_returns = strat.analyzers.time_return.get_analysis()
        eq_dates = sorted(time_returns.keys())
        eq_values = [initial_cash]
        for d in eq_dates:
            eq_values.append(eq_values[-1] * (1 + time_returns[d]))
        eq_series = pd.Series(eq_values[1:], index=pd.DatetimeIndex(eq_dates))

        bench_price = bt_data["Close"]
        bench_series = (bench_price / bench_price.iloc[0]) * initial_cash

        common_idx = eq_series.index.intersection(bench_series.index)
        if common_idx.empty:
            common_idx = bench_series.index
            eq_series = bench_series.copy()

        eq_aligned = eq_series.reindex(common_idx).ffill()
        bench_aligned = bench_series.reindex(common_idx).ffill()

        equity_curve = [
            {
                "date": str(d.date()),
                "value": round(float(eq_aligned.get(d, initial_cash)), 2),
                "benchmark": round(float(bench_aligned.get(d, initial_cash)), 2),
            }
            for d in common_idx
        ]

        trade_log = strat.trade_log
        metrics = self._compute_metrics(eq_aligned, bench_aligned, trade_log)
        bench_trade = [{"pnl_pct": float((bench_aligned.iloc[-1] / initial_cash - 1) * 100)}]
        bench_metrics = self._compute_metrics(bench_aligned, bench_aligned, bench_trade)

        logger.info(
            f"[BacktestAgent] {ticker}: CAGR={metrics.cagr:.1f}% "
            f"Sharpe={metrics.sharpe_ratio:.2f} MaxDD={metrics.max_drawdown:.1f}% "
            f"Trades={metrics.num_trades} WinRate={metrics.win_rate:.0f}%"
        )

        return BacktestResult(
            ticker=ticker,
            start_date=str(bt_data.index[0].date()),
            end_date=str(bt_data.index[-1].date()),
            initial_cash=initial_cash,
            final_value=round(final_value, 2),
            benchmark_final=round(float(bench_series.iloc[-1]), 2),
            metrics=metrics,
            benchmark_metrics=bench_metrics,
            equity_curve=equity_curve,
            trades=trade_log,
            signals_generated=len(bt_signals),
            buy_signals=buy_count,
            sell_signals=sell_count,
        )
