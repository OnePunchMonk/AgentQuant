"""Tests: features."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.agent.context_builder import build_context
from src.agent.episode_splits import synthetic_ohlcv
from src.features.engine import compute_features
from src.features.regime import detect_regime, detect_regime_full

warnings.filterwarnings("ignore", category=FutureWarning)



SIM = synthetic_ohlcv(3, n_days=1500)["SIM"]


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


@pytest.mark.parametrize("n", [176, 220, 300])
def test_short_windows_still_produce_features_and_a_regime(n):
    features = compute_features({"SIM": SIM.iloc[:n]}, "SIM")
    assert len(features) > 0
    assert detect_regime(features) != "Unknown"


def test_long_horizon_features_are_unknown_not_dropped_and_not_nan_in_the_context():
    features = compute_features({"SIM": SIM.iloc[:176]}, "SIM")
    assert features["momentum_252d"].isna().all()  # there is no 252-day history
    context = build_context(features)
    assert context.momentum_252d == 0.0
    assert "nan" not in context.to_prompt_string().lower()


def test_long_windows_are_unchanged():
    features = compute_features({"SIM": SIM.iloc[:1000]}, "SIM")
    last = features.iloc[-1]
    assert not last.isna().any()
    assert detect_regime(features) == detect_regime(features.dropna())


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


def test_no_nan_in_required_features():
    """Every feature a consumer needs is NaN-free. The long-horizon ones (252-day momentum,
    200-day SMA) are allowed to be NaN where the window is too short to define them; dropping
    those rows used to empty the frame for windows of about 252 bars or fewer."""
    from src.features.engine import _LONG_HORIZON_FEATURES

    data = _make_ohlcv(n=300)
    features = compute_features(data, "SPY", "^VIX")
    required = [c for c in features.columns if c not in _LONG_HORIZON_FEATURES]
    assert len(features) > 0
    assert not features[required].isnull().any().any(), "a required feature contains NaN"


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
