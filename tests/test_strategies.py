"""Tests for strategy registry and signal generation."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.strategy_registry import STRATEGY_REGISTRY, get_strategy_function


def _make_df(n: int = 300, trend: float = 0.001, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high = close * 1.005
    low = close * 0.995
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": 1e6}, index=idx)


@pytest.mark.parametrize("strategy_name", list(STRATEGY_REGISTRY.keys()))
def test_strategy_registered(strategy_name):
    """All strategies should be in the registry."""
    assert strategy_name in STRATEGY_REGISTRY


@pytest.mark.parametrize("strategy_name,params", [
    ("momentum", {"fast_window": 10, "slow_window": 30}),
    ("mean_reversion", {"window": 20, "num_std": 2.0}),
    ("volatility", {"window": 21, "vol_threshold": 0.20}),
    ("trend_following", {"short_window": 10, "medium_window": 30, "long_window": 90}),
    ("breakout", {"window": 20, "threshold_pct": 0.02}),
])
def test_strategy_produces_valid_signal(strategy_name, params):
    """Every strategy should produce a Series of {-1, 0, 1}."""
    df = _make_df()
    strategy = STRATEGY_REGISTRY[strategy_name]
    signal = strategy.generate_signal(df, params)
    assert isinstance(signal, pd.Series)
    assert len(signal) == len(df)
    assert signal.isin([-1, 0, 1]).all(), f"Signal contains values outside {{-1,0,1}}: {signal.unique()}"


def test_get_strategy_function_backward_compat():
    """Shim should return a callable."""
    fn = get_strategy_function("momentum")
    assert callable(fn)


def test_invalid_strategy_raises():
    from src.backtest.runner import run_backtest
    from src.exceptions import StrategyNotFoundError
    df = _make_df(n=300)
    with pytest.raises(StrategyNotFoundError, match="not found"):
        run_backtest({"SPY": df}, ["SPY"], "nonexistent_strategy", {})


def test_momentum_profitable_on_trend():
    """Momentum on strong uptrend should produce positive Sharpe."""
    from src.backtest.metrics import PerformanceMetrics
    from src.backtest.runner import run_backtest

    df = _make_df(n=400, trend=0.002)  # strong uptrend
    result = run_backtest({"SPY": df}, ["SPY"], "momentum", {"fast_window": 10, "slow_window": 30})
    assert result is not None
    assert "equity_curve" in result
    metrics = PerformanceMetrics.from_equity(result["equity_curve"])
    assert metrics["sharpe"] > 0.0
