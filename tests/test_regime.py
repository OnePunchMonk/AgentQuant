"""Tests for regime detection."""

import numpy as np
import pandas as pd
import pytest

from src.features.regime import detect_regime, detect_regime_full


def _make_features(n=300, vix_level=20.0, momentum_63d=0.06, has_vix=True):
    idx = pd.date_range("2020-01-01", periods=n)
    close = pd.Series(np.linspace(100, 100 * (1 + momentum_63d), n), index=idx)
    df = pd.DataFrame({
        "Close": close,
        "momentum_21d": np.full(n, momentum_63d * 0.5),
        "momentum_63d": np.full(n, momentum_63d),
        "momentum_252d": np.full(n, momentum_63d * 2),
        "volatility_21d": np.full(n, 0.15),
        "sma_200": close * 0.95,
        "sma_50": close * 0.98,
    }, index=idx)
    if has_vix:
        # Fill with historical VIX: first 200 bars at 15, then spike to vix_level
        vix_vals = np.full(n, 15.0)
        vix_vals[-50:] = vix_level
        df["vix_close"] = vix_vals
    return df


def test_low_vol_bull():
    features = _make_features(vix_level=12.0, momentum_63d=0.10)
    regime = detect_regime(features)
    assert "Bull" in regime
    assert "Low" in regime or "Mid" in regime


def test_crisis_bear():
    """High VIX + negative momentum should be Crisis-Bear."""
    features = _make_features(vix_level=55.0, momentum_63d=-0.15)
    regime = detect_regime(features)
    assert "Bear" in regime or "Neutral" in regime
    assert "Crisis" in regime or "HighVol" in regime


def test_regime_confidence_in_full():
    features = _make_features(vix_level=12.0, momentum_63d=0.12)
    signals = detect_regime_full(features)
    assert 0.0 <= signals.regime_confidence <= 1.0
    assert signals.regime_label != "Unknown"


def test_no_vix_falls_back_gracefully():
    features = _make_features(has_vix=False)
    regime = detect_regime(features)
    assert isinstance(regime, str)
    assert len(regime) > 0


def test_vix_spike_detected_as_high_vol():
    """When VIX is in the 90th percentile, vol_regime should be high or crisis."""
    features = _make_features(vix_level=50.0, momentum_63d=0.0)
    signals = detect_regime_full(features)
    assert signals.vol_regime in ("high", "crisis")
