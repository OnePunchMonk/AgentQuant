"""Shared return and evaluation math.

Every evaluator in the project (the production backtest runner and the
benchmark arms) goes through the functions here, so there is exactly one
definition of lag, cost and metrics.

Conventions
-----------
* A position is decided at the close of bar ``t`` and earns the return of bar
  ``t + 1`` (``signal.shift(1)``). Without the lag a rule would trade on a
  price it has not seen.
* Cost is charged on the bar the position changes, proportional to the absolute
  change (a flip from -1 to +1 is two units).
* ``evaluate_window`` computes the signal on the *full history it is given* and
  scores only the bars from ``eval_start`` on. Moving-average strategies need
  history before the window they are scored on; scoring a short slice on its
  own silently changes the strategy being measured.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.metrics import PerformanceMetrics


def strategy_returns(
    close: pd.Series, signal: pd.Series, cost_per_unit: float
) -> Tuple[pd.Series, pd.Series]:
    """Net strategy returns and per-bar turnover for a position series.

    Args:
        close: price series.
        signal: position decided at each close (typically in {-1, 0, 1}).
        cost_per_unit: cost per unit of turnover as a fraction (5 bps = 0.0005).

    Returns:
        (net_returns, turnover), both indexed like ``close``.
    """
    signal = signal.reindex(close.index).fillna(0.0).astype(float)
    turnover = signal.diff().abs().fillna(0.0)
    gross = close.pct_change().fillna(0.0) * signal.shift(1).fillna(0.0)
    return gross - turnover * cost_per_unit, turnover


def window_metrics(net_returns: pd.Series, turnover: pd.Series) -> Optional[Dict[str, float]]:
    """Metrics for one scored window, or None if it is too short to score."""
    if len(net_returns) < 5:
        return None
    equity = np.concatenate([[1.0], (1.0 + net_returns.fillna(0.0)).cumprod().to_numpy()])
    equity_series = pd.Series(equity)
    return {
        "observations": float(len(net_returns)),
        "sharpe": PerformanceMetrics.sharpe(net_returns),
        "net_return": float(equity[-1] - 1.0),
        "max_drawdown": PerformanceMetrics.max_drawdown(equity_series),
        "turnover": float(turnover.sum()),
    }


def required_warmup_bars(params: Dict[str, Any], default: int = 0) -> int:
    """History a strategy needs: the longest window-like parameter.

    The runner used to look only at ``slow_window`` or ``window``, so a
    trend-following config (``long_window``) was checked against a 252-bar
    default and rejected on any window shorter than that. Any parameter whose
    name contains "window" counts.
    """
    windows = [
        int(v) for k, v in params.items()
        if "window" in str(k) and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return max(windows) if windows else default


def evaluate_window(
    df: pd.DataFrame,
    strategy_name: str,
    params: Dict[str, Any],
    cost_per_unit: float,
    eval_start: Optional[Any] = None,
) -> Optional[Dict[str, float]]:
    """Score ``strategy_name(params)`` on ``df``, counting bars from ``eval_start``.

    ``df`` should include whatever history precedes ``eval_start`` (and must not
    include anything after the decision point). Returns None when the scored
    window has fewer than 5 bars.
    """
    from src.strategies.base import _get_close
    from src.strategies.strategy_registry import STRATEGY_REGISTRY

    if df is None or df.empty:
        return None
    close = _get_close(df).astype(float)
    signal = STRATEGY_REGISTRY[strategy_name].generate_signal(df, params)
    net, turnover = strategy_returns(close, signal, cost_per_unit)
    keep = np.ones(len(net), dtype=bool)
    if eval_start is not None:
        keep = np.asarray(net.index >= pd.Timestamp(eval_start))
    if len(keep) and keep[0]:
        # The very first row has no prior price, so its return is an artificial 0.
        # Dropping it makes whole-frame scoring match the runner (equity.pct_change()).
        keep[0] = False
    return window_metrics(net[keep], turnover[keep])


def common_warmup_bars(grid: list, max_fraction: float = 0.4, n_bars: Optional[int] = None) -> int:
    """Bars of history every config in ``grid`` needs before its signal is defined.

    Used to pick one shared scoring start for all candidates, so a long-window
    config is not penalized by flat warmup bars that a short-window config
    does not have. Returns 0 when the grid has no window parameters, or when
    ``n_bars`` is given and the required warmup would exceed ``max_fraction`` of
    it (the caller should then keep the old whole-window scoring).
    """
    longest = max((required_warmup_bars(entry) for entry in grid), default=0)
    if longest == 0:
        return 0
    # One extra bar: the position that earns the first scored return is the signal
    # from the bar before, and shifted-window rules (breakout) or pct_change-based
    # rules (volatility) need a bar more than their window to be defined.
    longest += 1
    if n_bars is not None and longest > max_fraction * n_bars:
        return 0
    return longest
