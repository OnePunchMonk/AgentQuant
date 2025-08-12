import pytest
import pandas as pd
import numpy as np
from src.backtest.runner import run_backtest

@pytest.fixture
def sample_trending_data():
    """Creates a sample OHLCV DataFrame with a clear upward trend."""
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=200))
    # Strong upward trend with some noise
    close_prices = 100 + np.arange(200) * 0.5 + np.random.randn(200) * 2
    df = pd.DataFrame({'Close': close_prices}, index=dates)
    return df

def test_run_backtest_momentum(sample_trending_data):
    """Tests that a momentum strategy is profitable on clearly trending data."""
    params = {'fast_window': 10, 'slow_window': 30}
    
    stats = run_backtest(
        ohlcv_df=sample_trending_data,
        asset_ticker='TREND',
        strategy_name='momentum',
        params=params
    )
    
    assert isinstance(stats, pd.Series)
    assert stats['Total Return [%]'] > 10.0 # Should be profitable
    assert stats['Sharpe Ratio'] > 1.0 # Should be a good Sharpe
    assert stats['Num Trades'] > 0 # Should have made trades

def test_run_backtest_invalid_strategy(sample_trending_data):
    """Tests that the runner raises an error for a non-existent strategy."""
    params = {'fast_window': 10, 'slow_window': 30}
    with pytest.raises(ValueError, match="Strategy 'non_existent_strat' not found"):
        run_backtest(
            ohlcv_df=sample_trending_data,
            asset_ticker='TREND',
            strategy_name='non_existent_strat',
            params=params
        )