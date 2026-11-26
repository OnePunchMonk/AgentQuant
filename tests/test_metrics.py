"""Tests for PerformanceMetrics — single source of truth validation."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import PerformanceMetrics


def _make_equity(n: int = 252, drift: float = 0.001, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    return pd.Series((1 + rets).cumprod() * 100_000)


def test_sharpe_flat_returns():
    """Flat returns should give Sharpe of 0."""
    returns = pd.Series([0.0] * 252)
    assert PerformanceMetrics.sharpe(returns) == 0.0


def test_sharpe_known_value():
    """Known daily return should give known Sharpe."""
    # 1% daily return, 0% std → infinity, but let's use a realistic case
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 252))
    sharpe = PerformanceMetrics.sharpe(rets)
    # Expected around sqrt(252) * (0.001/0.01) = ~1.59
    assert 0.5 < sharpe < 4.0


def test_max_drawdown_no_drawdown():
    """Monotonically increasing equity curve has zero drawdown."""
    equity = pd.Series(np.linspace(100, 200, 252))
    dd = PerformanceMetrics.max_drawdown(equity)
    assert dd == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known():
    """Equity drops from 100 to 50 — max drawdown should be 0.50."""
    equity = pd.Series([100, 90, 80, 70, 60, 50, 60, 70, 80])
    dd = PerformanceMetrics.max_drawdown(equity)
    assert dd == pytest.approx(0.50, abs=1e-6)


def test_from_equity_returns_all_keys():
    equity = _make_equity()
    metrics = PerformanceMetrics.from_equity(equity)
    for key in ("total_return", "sharpe", "max_drawdown", "calmar", "sortino"):
        assert key in metrics


def test_zero_signal_flat_equity():
    """All-zero signals should produce flat equity curve."""
    close = pd.Series(np.linspace(100, 110, 252))
    daily_ret = close.pct_change().fillna(0)
    signal = pd.Series(0, index=close.index)
    strat_ret = daily_ret * signal.shift(1).fillna(0)
    equity = (1 + strat_ret).cumprod() * 100_000
    tr = PerformanceMetrics.total_return(equity)
    assert tr == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("block_size", [1, 7, 20, 500, 0])
@pytest.mark.parametrize("constant", [None, 0.0, 0.01])
def test_bootstrap_matches_pandas_reference(monkeypatch, block_size, constant):
    """Keep overlapping blocks, trimming, ddof and zero-variance behavior unchanged."""
    values = (
        np.random.default_rng(7).normal(0, 0.01, 53)
        if constant is None else np.full(53, constant)
    )
    returns = pd.Series(values)
    returns.iloc[4] = np.nan
    values = returns.dropna().to_numpy()
    size = max(1, min(block_size, len(values)))
    reference_rng = np.random.default_rng(42)
    sharpes = []
    for _ in range(31):
        starts = reference_rng.integers(
            0, len(values) - size + 1, size=int(np.ceil(len(values) / size))
        )
        sample = np.concatenate([values[s:s + size] for s in starts])[:len(values)]
        sharpes.append(PerformanceMetrics.sharpe(pd.Series(sample)))
    rng = np.random.default_rng(42)
    monkeypatch.setattr(np.random, "default_rng", lambda: rng)
    assert PerformanceMetrics.bootstrap_sharpe(returns, n=31, pct=17, block_size=block_size) == pytest.approx(
        np.percentile(sharpes, 17), abs=1e-12
    )


def test_bootstrap_short_history():
    assert PerformanceMetrics.bootstrap_sharpe(pd.Series([np.nan] + [0.01] * 9)) == 0.0
