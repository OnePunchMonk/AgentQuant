# src/backtest/simple_backtest.py
from typing import Dict, Any
import numpy as np
import pandas as pd

def max_drawdown_from_equity(equity: pd.Series) -> float:
    equity = equity.dropna()
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max
    return float(drawdown.max())

def ensure_equity_from_returns(maybe_series: pd.Series) -> pd.Series:
    s = maybe_series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    # heuristic: if values mostly small (|mean| < 0.5) treat as returns
    if (s.abs().mean() < 0.5) and (s.max() < 10):
        return (1 + s).cumprod()
    return s

def basic_momentum_backtest(ohlcv_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple deterministic dual-moving-average momentum backtest.
    - ohlcv_df: DataFrame indexed by date with at least a 'Close' column
    - params: {'fast_window': int, 'slow_window': int}
    Returns a dict with keys:
      - 'equity_curve' -> pd.Series
      - 'max_drawdown' -> float (decimal, e.g. 0.12)
      - 'sharpe' -> float (annualized)
      - 'total_return' -> float (decimal)
      - 'num_trades' -> int
      - 'params' -> params (echoed)
    """
    fast = int(params.get('fast_window', 21))
    slow = int(params.get('slow_window', 63))
    close_col_candidates = [c for c in ohlcv_df.columns if 'close' in str(c).lower()]
    if not close_col_candidates:
        raise ValueError("ohlcv_df must contain a Close column (or column name containing 'close').")
    close = ohlcv_df[close_col_candidates[0]].astype(float).dropna()

    # signals: 1 when fast > slow else 0; shift so we trade next bar open (approx)
    sma_fast = close.rolling(window=fast, min_periods=1).mean()
    sma_slow = close.rolling(window=slow, min_periods=1).mean()
    signal = (sma_fast > sma_slow).astype(int)
    # entry/exit points count
    trades = signal.diff().fillna(0).abs().sum() / 2.0  # rough count of round-trip changes

    # compute daily returns of buy-and-hold then mask by signal to get strategy returns
    # approximate strategy return: when signal==1 we get daily close returns, else zero (cash)
    daily_returns = close.pct_change().fillna(0)
    strat_returns = daily_returns * signal.shift(1).fillna(0)  # act on previous signal

    equity = (1 + strat_returns).cumprod()
    if equity.empty:
        # fallback: treat buy-and-hold
        equity = (1 + daily_returns).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)
    # annualized Sharpe (assumes ~252 trading days)
    mean_ret = strat_returns.mean() * 252
    std_ret = strat_returns.std() * np.sqrt(252)
    sharpe = float(mean_ret / std_ret) if std_ret and std_ret != 0 else float("nan")

    max_dd = max_drawdown_from_equity(equity)

    return {
        'equity_curve': equity,
        'max_drawdown': float(max_dd),
        'sharpe': sharpe,
        'total_return': total_return,
        'num_trades': int(trades),
        'params': params
    }
