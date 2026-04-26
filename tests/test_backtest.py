"""
Backtest runner tests — updated to use the new unified interface.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.runner import run_backtest


def _make_df(n: int = 300, trend: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.005, "Low": close * 0.995,
        "Close": close, "Volume": 1_000_000
    }, index=idx)


def test_momentum_backtest_returns_result():
    df = _make_df()
    result = run_backtest({"SPY": df}, ["SPY"], "momentum", {"fast_window": 10, "slow_window": 30})
    assert result is not None
    assert "equity_curve" in result
    assert "metrics" in result


def test_invalid_strategy_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="not found"):
        run_backtest({"SPY": df}, ["SPY"], "nonexistent_strat", {})


def test_zero_signal_flat_equity():
    """Volatility strategy with very high threshold → near-zero signal → flat equity."""
    from src.backtest.metrics import PerformanceMetrics
    df = _make_df()
    # vol_threshold so high that daily vol never exceeds it → flat equity
    result = run_backtest({"SPY": df}, ["SPY"], "volatility",
                          {"window": 21, "vol_threshold": 99.0})
    assert result is not None
    eq = result["equity_curve"]
    metrics = PerformanceMetrics.from_equity(eq)
    assert abs(metrics["total_return"]) < 0.01


def test_metrics_keys_present():
    df = _make_df(trend=0.002)
    result = run_backtest({"SPY": df}, ["SPY"], "momentum", {"fast_window": 10, "slow_window": 40})
    assert result is not None
    for key in ("total_return", "max_drawdown", "sharpe_ratio"):
        assert key in result["metrics"], f"Missing metric key: {key}"