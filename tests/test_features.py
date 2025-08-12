import pytest
import pandas as pd
import numpy as np
from src.features.engine import compute_features

@pytest.fixture
def sample_ohlcv_data():
    """Creates a sample OHLCV DataFrame for testing."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
    close_prices = 100 + np.cumsum(np.random.randn(100))
    data = {
        'Open': close_prices - 0.5,
        'High': close_prices + 1,
        'Low': close_prices - 1,
        'Close': close_prices,
        'Volume': np.random.randint(1000, 5000, size=100)
    }
    df = pd.DataFrame(data, index=dates)
    
    vix_dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100))
    vix_data = pd.DataFrame({'Close': 20 + np.random.randn(100)}, index=vix_dates)
    
    return {'SPY': df, '^VIX': vix_data}

def test_compute_features_columns(sample_ohlcv_data):
    """Tests if the feature computation creates the expected columns."""
    features = compute_features(sample_ohlcv_data, 'SPY', '^VIX')
    expected_cols = [
        'volatility_21d', 'momentum_63d', 'sma_63', 'price_vs_sma63', 'vix_close'
    ]
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    for col in expected_cols:
        assert col in features.columns

def test_compute_features_no_vix(sample_ohlcv_data):
    """Tests that feature computation runs without VIX data."""
    ohlcv_no_vix = {'SPY': sample_ohlcv_data['SPY']}
    features = compute_features(ohlcv_no_vix, 'SPY', '^VIX')
    assert 'vix_close' not in features.columns
    assert 'momentum_63d' in features.columns # Ensure other features were still made

def test_compute_features_values(sample_ohlcv_data):
    """Tests a specific calculated value for correctness."""
    df = sample_ohlcv_data['SPY']
    # Manually calculate one value for a sanity check
    # pct_change over 21 periods for the last row
    expected_mom = (df['Close'].iloc[-1] / df['Close'].iloc[-1-21]) - 1
    
    features = compute_features(sample_ohlcv_data, 'SPY', '^VIX')
    
    # Compare with a tolerance for floating point errors
    assert np.isclose(features['momentum_21d'].iloc[-1], expected_mom)