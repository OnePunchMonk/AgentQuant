"""
Synthetic Datasets
==================

Small, deterministic datasets used by `peek demo`, the README examples, and
the test suite. One is intentionally leaky, one is intentionally clean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _synthetic_series(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    returns = rng.normal(loc=0.0003, scale=0.01, size=n)
    price = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"date": dates, "price": price})


def make_leaky_dataset(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """
    A dataset with two classic, real-world leaks baked in:

    - `centered_ma_5`: a rolling mean centered on each row (uses t-2..t+2),
      which secretly looks 2 days into the future.
    - `future_return_leak`: the next day's return, accidentally left in the
      feature set used to predict `target` (itself a shifted return).
    """
    df = _synthetic_series(n, seed)
    df["return"] = df["price"].pct_change().fillna(0.0)

    df["target"] = df["return"].shift(-1)

    df["trailing_ma_5"] = df["price"].rolling(5, min_periods=1).mean()
    df["centered_ma_5"] = df["price"].rolling(5, center=True, min_periods=1).mean()
    df["future_return_leak"] = df["return"].shift(-1)
    df["rsi_14"] = _causal_rsi(df["price"], window=14)

    df = df.iloc[:-1].reset_index(drop=True)  # drop last row (NaN target)
    return df


def make_clean_dataset(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """The same generative process, with only causal (trailing) features."""
    df = _synthetic_series(n, seed)
    df["return"] = df["price"].pct_change().fillna(0.0)

    df["target"] = df["return"].shift(-1)

    df["trailing_ma_5"] = df["price"].rolling(5, min_periods=1).mean()
    df["trailing_ma_20"] = df["price"].rolling(20, min_periods=1).mean()
    df["rsi_14"] = _causal_rsi(df["price"], window=14)

    df = df.iloc[:-1].reset_index(drop=True)
    return df


def _causal_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def leaky_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Feature function mirroring `make_leaky_dataset`'s leaky columns."""
    price = df["price"]
    ret = price.pct_change().fillna(0.0)
    out = pd.DataFrame(index=df.index)
    out["trailing_ma_5"] = price.rolling(5, min_periods=1).mean()
    out["centered_ma_5"] = price.rolling(5, center=True, min_periods=1).mean()
    out["future_return_leak"] = ret.shift(-1)
    out["rsi_14"] = _causal_rsi(price, window=14)
    return out


def clean_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    """Feature function mirroring `make_clean_dataset`'s causal columns."""
    price = df["price"]
    out = pd.DataFrame(index=df.index)
    out["trailing_ma_5"] = price.rolling(5, min_periods=1).mean()
    out["trailing_ma_20"] = price.rolling(20, min_periods=1).mean()
    out["rsi_14"] = _causal_rsi(price, window=14)
    return out
