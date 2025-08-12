"""
This module defines the expected schemas for pandas DataFrames.
It serves as documentation and can be used for data validation in the future,
for example with a library like `pandera`.
"""

# --- Raw Data Schemas ---

OHLCV_SCHEMA = {
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Volume": "int64",
    # Index is a pandas DatetimeIndex
}

FRED_SCHEMA = {
    "SERIES_ID": "float64",
    # Index is a pandas DatetimeIndex
}


# --- Processed Data Schemas ---

FEATURE_SCHEMA = {
    # Inherits OHLCV_SCHEMA columns
    "volatility_21d": "float64",
    "volatility_63d": "float64",
    "momentum_21d": "float64",
    "momentum_63d": "float64",
    "momentum_252d": "float64",
    "sma_21": "float64",
    "sma_63": "float64",
    "price_vs_sma63": "float64",
    "vix_close": "float64", # Optional
    # Index is a pandas DatetimeIndex
}


# --- Output Schemas ---

BACKTEST_RESULTS_SCHEMA = {
    # From vectorbt
    "Total Return [%]": "float64",
    "Max Drawdown [%]": "float64",
    "Sharpe Ratio": "float64",
    "Num Trades": "int64",
    # Custom added
    "label": "object (string)",
    "params": "object (string repr of dict)",
}