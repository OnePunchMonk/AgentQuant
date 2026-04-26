"""Tests for extended feature engine."""

import numpy as np
import pandas as pd
import pytest

from src.features.engine import compute_features


def _make_ohlcv(n=350, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    high = close * 1.005
    low = close * 0.995
    idx = pd.date_range("2020-01-01", periods=n)
    df = pd.DataFrame({
        "Open": close, "High": high, "Low": low,
        "Close": close, "Volume": 1_000_000
    }, index=idx)
    vix = pd.DataFrame({"Close": rng.uniform(15, 35, n)}, index=idx)
    return {"SPY": df, "^VIX": vix}


def test_expected_columns_present():
    data = _make_ohlcv()
    features = compute_features(data, "SPY", "^VIX")
    required = [
        "volatility_5d", "volatility_21d", "volatility_63d",
        "momentum_21d", "momentum_63d", "momentum_252d",
        "sma_21", "sma_50", "sma_63", "sma_200",
        "rsi_14", "macd", "macd_signal",
        "bb_upper", "bb_lower", "bb_width",
        "atr_14", "drawdown_from_peak", "vix_close",
    ]
    for col in required:
        assert col in features.columns, f"Missing column: {col}"


def test_rsi_bounded():
    """RSI must be in [0, 100]."""
    data = _make_ohlcv()
    features = compute_features(data, "SPY", "^VIX")
    assert features["rsi_14"].min() >= 0.0
    assert features["rsi_14"].max() <= 100.0


def test_momentum_21d_value():
    """momentum_21d at row i should equal (close[i] / close[i-21]) - 1."""
    data = _make_ohlcv(n=350)
    features = compute_features(data, "SPY", "^VIX")
    # Compute expected against the features index (post-dropna)
    close_aligned = data["SPY"]["Close"].reindex(features.index)
    expected = (close_aligned.iloc[-1] / close_aligned.iloc[-22]) - 1
    assert np.isclose(features["momentum_21d"].iloc[-1], expected, atol=1e-8)


def test_no_vix_runs_without_error():
    data = _make_ohlcv()
    del data["^VIX"]
    features = compute_features(data, "SPY", "^VIX")
    assert "vix_close" not in features.columns
    assert "rsi_14" in features.columns


def test_no_nan_after_dropna():
    data = _make_ohlcv(n=300)
    features = compute_features(data, "SPY", "^VIX")
    assert not features.isnull().any().any(), "Features DataFrame contains NaN values"