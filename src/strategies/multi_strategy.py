"""
Multi-Strategy Implementation Module
====================================

This module provides a comprehensive collection of quantitative trading strategies
implemented in a unified framework. Each strategy follows consistent interfaces
for signal generation, position calculation, and performance evaluation.

The module supports multiple strategy types including momentum, mean reversion,
volatility targeting, trend following, breakout, and regime-based strategies.
All strategies are designed to work with vectorized pandas operations for
optimal performance on large datasets.

Key Features:
- Unified strategy interface with consistent API
- Robust parameter handling and validation
- Support for multi-asset strategies
- Market regime awareness and adaptation
- Comprehensive signal generation logic

Dependencies:
- pandas: Time series data manipulation
- numpy: Numerical computations and array operations
- typing: Type hints for better code documentation

Author: AgentQuant Development Team
License: MIT
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


def _get_col(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Robust column extraction utility for handling various DataFrame formats.
    
    This utility function attempts to extract a column from a DataFrame using
    case-insensitive matching against a list of candidate column names. It's
    designed to handle edge cases like tuple column names, MultiIndex columns,
    and various data types that might appear in financial datasets.
    
    Args:
        df (pd.DataFrame): Input DataFrame to search for columns
        candidates (List[str]): List of potential column names to search for
        
    Returns:
        pd.Series: Numeric series with NaN values removed, or empty series if no match
        
    Note:
        - Performs case-insensitive matching using string conversion
        - Handles tuple column names by converting to string representation
        - Coerces data to numeric format and drops NaN values
        - Provides fallback mechanism for edge cases
    """
    # Build a case-insensitive mapping from stringified column names to original
    try:
        col_map = {str(c).lower(): c for c in df.columns}
    except Exception:
        # In rare cases df.columns may not be iterable as expected; fallback
        col_map = {}
    for name in candidates:
        ln = str(name).lower()
        if ln in col_map:
            return pd.to_numeric(df[col_map[ln]], errors='coerce').dropna()
    # Fallback: first column that coerces to numeric with at least one non-null
    for c in df.columns:
        try:
            s = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            # If c is unusual (e.g., tuple key), indexing still works; continue
            try:
                s = pd.to_numeric(df.loc[:, c], errors='coerce')
            except Exception:
                continue
        if getattr(s, 'notna', lambda: pd.Series([]))().any():
            return s.dropna()
    # If nothing found, raise with the first candidate requested
    raise KeyError(candidates[0])


def _get_close(df: pd.DataFrame) -> pd.Series:
    return _get_col(df, ['Close', 'Adj Close', 'adjclose', 'price'])


def _get_high(df: pd.DataFrame) -> pd.Series:
    try:
        return _get_col(df, ['High'])
    except KeyError:
        # approximate using close if high missing
        return _get_close(df)


def _get_low(df: pd.DataFrame) -> pd.Series:
    try:
        return _get_col(df, ['Low'])
    except KeyError:
        return _get_close(df)


def calculate_momentum_signal(data: pd.DataFrame, fast_window: int, slow_window: int) -> pd.Series:
    """
    Calculate momentum signal based on moving average crossover.
    
    Args:
        data: DataFrame with OHLCV data
        fast_window: Fast moving average window
        slow_window: Slow moving average window
        
    Returns:
        Series with momentum signals (1 for buy, -1 for sell, 0 for neutral)
    """
    # Calculate moving averages
    close = _get_close(data)
    fast_ma = close.rolling(window=fast_window).mean()
    slow_ma = close.rolling(window=slow_window).mean()
    
    # Generate signals
    signal = pd.Series(0, index=data.index)
    signal[fast_ma > slow_ma] = 1
    signal[fast_ma < slow_ma] = -1
    
    return signal


def calculate_mean_reversion_signal(data: pd.DataFrame, window: int, num_std: float) -> pd.Series:
    """
    Calculate mean reversion signal based on Bollinger Bands.
    
    Args:
        data: DataFrame with OHLCV data
        window: Lookback window for calculating mean and standard deviation
        num_std: Number of standard deviations for upper and lower bands
        
    Returns:
        Series with mean reversion signals (1 for buy, -1 for sell, 0 for neutral)
    """
    # Calculate middle band (SMA)
    close = _get_close(data)
    middle_band = close.rolling(window=window).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # Generate signals
    signal = pd.Series(0, index=data.index)
    signal[close > upper_band] = -1  # Sell when price is above upper band
    signal[close < lower_band] = 1   # Buy when price is below lower band
    
    return signal


def calculate_volatility_signal(data: pd.DataFrame, window: int, vol_threshold: float) -> pd.Series:
    """
    Calculate volatility-based signal.
    
    Args:
        data: DataFrame with OHLCV data
        window: Lookback window for calculating volatility
        vol_threshold: Volatility threshold for signal generation
        
    Returns:
        Series with volatility signals (1 for buy, 0 for neutral)
    """
    # Calculate daily returns
    close = _get_close(data)
    returns = close.pct_change()
    
    # Calculate rolling volatility
    volatility = returns.rolling(window=window).std()
    
    # Generate signals
    signal = pd.Series(0, index=data.index)
    signal[volatility < vol_threshold] = 1  # Buy when volatility is low
    
    return signal


def calculate_trend_following_signal(
    data: pd.DataFrame, 
    short_window: int, 
    medium_window: int, 
    long_window: int
) -> pd.Series:
    """
    Calculate trend following signal based on multiple moving averages.
    
    Args:
        data: DataFrame with OHLCV data
        short_window: Short-term moving average window
        medium_window: Medium-term moving average window
        long_window: Long-term moving average window
        
    Returns:
        Series with trend following signals (1 for buy, -1 for sell, 0 for neutral)
    """
    # Calculate moving averages
    close = _get_close(data)
    short_ma = close.rolling(window=short_window).mean()
    medium_ma = close.rolling(window=medium_window).mean()
    long_ma = close.rolling(window=long_window).mean()
    
    # Generate signals
    signal = pd.Series(0, index=data.index)
    
    # Strong uptrend
    uptrend = (short_ma > medium_ma) & (medium_ma > long_ma)
    signal[uptrend] = 1
    
    # Strong downtrend
    downtrend = (short_ma < medium_ma) & (medium_ma < long_ma)
    signal[downtrend] = -1
    
    return signal


def calculate_breakout_signal(
    data: pd.DataFrame,
    window: int,
    threshold_pct: float
) -> pd.Series:
    """
    Calculate breakout signal based on price breaking above/below range.
    
    Args:
        data: DataFrame with OHLCV data
        window: Lookback window for calculating range
        threshold_pct: Threshold percentage for breakout confirmation
        
    Returns:
        Series with breakout signals (1 for buy, -1 for sell, 0 for neutral)
    """
    # Calculate rolling high and low
    high = _get_high(data)
    low = _get_low(data)
    close = _get_close(data)
    # Align indexes
    idx = close.index
    high = high.reindex(idx).ffill()
    low = low.reindex(idx).ffill()
    rolling_high = high.rolling(window=window).max()
    rolling_low = low.rolling(window=window).min()
    
    # Calculate threshold values
    upper_threshold = rolling_high * (1 + threshold_pct)
    lower_threshold = rolling_low * (1 - threshold_pct)
    
    # Generate signals
    signal = pd.Series(0, index=data.index)
    signal[close > upper_threshold] = 1    # Buy on upside breakout
    signal[close < lower_threshold] = -1   # Sell on downside breakout
    
    return signal


def calculate_regime_based_signal(
    data: pd.DataFrame, 
    regime_data: Any, 
    momentum_params: dict,
    mean_reversion_params: dict
) -> pd.Series:
    """
    Calculate regime-based signal that switches between momentum and mean reversion.
    
    Args:
        data: DataFrame with OHLCV data
        regime_data: Dictionary with regime information
        momentum_params: Parameters for momentum strategy
        mean_reversion_params: Parameters for mean reversion strategy
        
    Returns:
        Series with regime-based signals
    """
    # Debug: Print regime_data type and value
    print(f"DEBUG regime_based_signal: regime_data type={type(regime_data)}, value={regime_data}")
    
    # Accept either a string (e.g., from detect_regime) or a dict with a 'name' field
    if isinstance(regime_data, str):
        regime_type = regime_data.lower()
    elif isinstance(regime_data, dict):
        name = regime_data.get("name", "")
        if isinstance(name, (tuple, list)):
            name = str(name[0]) if len(name) > 0 else ""
        elif not isinstance(name, str):
            name = str(name)
        regime_type = name.lower()
    elif isinstance(regime_data, (tuple, list)):
        # Handle tuple/list case - convert to string
        regime_type = str(regime_data[0]).lower() if len(regime_data) > 0 else ""
    else:
        regime_type = str(regime_data).lower() if regime_data is not None else ""
    
    if "bull" in regime_type or "uptrend" in regime_type:
        # In bull market or uptrend, use momentum strategy
        return calculate_momentum_signal(
            data, 
            momentum_params.get("fast_window", 20), 
            momentum_params.get("slow_window", 50)
        )
    elif "bear" in regime_type or "downtrend" in regime_type:
        # In bear market or downtrend, use mean reversion strategy
        return calculate_mean_reversion_signal(
            data, 
            mean_reversion_params.get("window", 20), 
            mean_reversion_params.get("num_std", 2.0)
        )
    elif "volatility" in regime_type or "high_vol" in regime_type:
        # In high volatility regime, use volatility strategy
        return calculate_volatility_signal(
            data,
            momentum_params.get("window", 20),
            momentum_params.get("vol_threshold", 0.02)
        )
    else:
        # Default to momentum strategy
        return calculate_momentum_signal(
            data, 
            momentum_params.get("fast_window", 20), 
            momentum_params.get("slow_window", 50)
        )


def calculate_portfolio_weights(
    asset_tickers: List[str],
    data: Dict[str, pd.DataFrame],
    signals: Dict[str, pd.Series],
    allocation_weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Calculate portfolio weights based on signals and allocation weights.
    
    Args:
        asset_tickers: List of asset tickers
        data: Dictionary of DataFrames with OHLCV data for each asset
        signals: Dictionary of signal Series for each asset
        allocation_weights: Optional dictionary with allocation weights for each asset
        
    Returns:
        DataFrame with portfolio weights over time
    """
    # Combine all signals into a DataFrame
    all_signals = pd.DataFrame({ticker: signals[ticker] for ticker in asset_tickers})
    
    # Handle missing values
    all_signals = all_signals.fillna(0)
    
    # Initialize weights DataFrame
    weights = pd.DataFrame(0, index=all_signals.index, columns=asset_tickers)
    
    # If allocation weights are provided, use them
    if allocation_weights is not None:
        for ticker in asset_tickers:
            weight = allocation_weights.get(ticker, 1.0 / len(asset_tickers))
            weights[ticker] = all_signals[ticker] * weight
    else:
        # Equal weighting
        for ticker in asset_tickers:
            weights[ticker] = all_signals[ticker] / len(asset_tickers)
    
    # Normalize weights to ensure they sum to 1 (or -1 for short positions)
    for idx in weights.index:
        row_sum = weights.loc[idx].abs().sum()
        if row_sum > 0:
            weights.loc[idx] = weights.loc[idx] / row_sum
    
    return weights


def run_multi_asset_strategy(
    data: Dict[str, pd.DataFrame],
    asset_tickers: List[str],
    strategy_type: str,
    params: Dict[str, Any],
    allocation_weights: Optional[Dict[str, float]] = None,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run a multi-asset strategy backtest.
    
    Args:
        data: Dictionary of DataFrames with OHLCV data for each asset
        asset_tickers: List of asset tickers to include
        strategy_type: Type of strategy to run
        params: Strategy parameters
        allocation_weights: Optional allocation weights for each asset
        initial_capital: Initial capital for the backtest
        
    Returns:
        Dictionary with backtest results
    """
    # Validate inputs
    if not asset_tickers:
        raise ValueError("No assets specified")
    
    for ticker in asset_tickers:
        if ticker not in data:
            raise ValueError(f"Data for {ticker} not found")
    
    # Generate signals for each asset
    signals = {}
    for ticker in asset_tickers:
        asset_data = data[ticker]
        
        if strategy_type == "momentum":
            signals[ticker] = calculate_momentum_signal(
                asset_data, 
                params.get("fast_window", 20), 
                params.get("slow_window", 50)
            )
        elif strategy_type == "mean_reversion":
            signals[ticker] = calculate_mean_reversion_signal(
                asset_data, 
                params.get("window", 20), 
                params.get("num_std", 2.0)
            )
        elif strategy_type == "volatility":
            signals[ticker] = calculate_volatility_signal(
                asset_data,
                params.get("window", 20),
                params.get("vol_threshold", 0.02)
            )
        elif strategy_type == "trend_following":
            signals[ticker] = calculate_trend_following_signal(
                asset_data,
                params.get("short_window", 10),
                params.get("medium_window", 30),
                params.get("long_window", 90)
            )
        elif strategy_type == "breakout":
            signals[ticker] = calculate_breakout_signal(
                asset_data,
                params.get("window", 20),
                params.get("threshold_pct", 0.02)
            )
        elif strategy_type == "regime_based":
            # This is a placeholder - in a real implementation, 
            # we would need to pass regime data
            regime_data = {"name": "bull"}  # Default to bull regime
            signals[ticker] = calculate_regime_based_signal(
                asset_data,
                regime_data,
                {"fast_window": params.get("fast_window", 20), "slow_window": params.get("slow_window", 50)},
                {"window": params.get("mr_window", 20), "num_std": params.get("mr_num_std", 2.0)}
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Calculate portfolio weights
    weights = calculate_portfolio_weights(asset_tickers, data, signals, allocation_weights)
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(0.0, index=weights.index)
    
    for ticker in asset_tickers:
        asset_returns = _get_close(data[ticker]).pct_change()
        portfolio_returns += weights[ticker].shift(1) * asset_returns
    
    # Calculate equity curve
    equity_curve = (1 + portfolio_returns).cumprod() * initial_capital
    
    # Calculate performance metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
    
    # Calculate drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_drawdown = drawdown.min()
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
    
    # Return results
    return {
        "equity_curve": equity_curve,
        "weights": weights,
        "signals": signals,
        "metrics": {
            "Total Return [%]": total_return * 100,
            "Annual Return [%]": annual_return * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown [%]": max_drawdown * 100,
            "Num Trades": (weights.diff() != 0).sum().sum()
        }
    }
