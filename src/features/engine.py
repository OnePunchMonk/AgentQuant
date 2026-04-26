"""
Feature Engine — Extended Feature Computation
==============================================

Computes RSI, MACD, Bollinger Bands, ATR, multi-horizon realized vol,
stationarity checks, and drawdown features in addition to base indicators.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column extraction utilities
# ---------------------------------------------------------------------------

def _find_field_series(df: pd.DataFrame, field: str) -> pd.Series:
    """
    Robustly find a single Series in df for `field` (e.g. 'Close').
    Handles MultiIndex, case-insensitive match, and substring fallback.
    """
    field_l = field.lower()

    def _to_series(candidate):
        s = df[candidate]
        if isinstance(s, pd.DataFrame):
            logger.warning("Multiple columns found for %r; using first.", candidate)
            s = s.iloc[:, 0]
        return s.rename(field)

    cols = df.columns

    if isinstance(cols, pd.MultiIndex):
        matches = [c for c in cols if any(str(x).lower() == field_l for x in c)]
        if matches:
            return _to_series(matches[0])
        for lvl in range(cols.nlevels):
            for col in cols:
                if str(col[lvl]).lower() == field_l:
                    return _to_series(col)
        sub = [c for c in cols if any(field_l in str(x).lower() for x in c)]
        if sub:
            return _to_series(sub[0])
    else:
        if field in cols:
            return _to_series(field)
        exact_ci = [c for c in cols if str(c).lower() == field_l]
        if exact_ci:
            return _to_series(exact_ci[0])
        sub = [c for c in cols if field_l in str(c).lower()]
        if sub:
            logger.warning("Substring match for field %r -> %s", field, sub[0])
            return _to_series(sub[0])

    raise KeyError(f"Could not find field '{field}' in DataFrame. Columns sample: {list(cols[:20])}")


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "rsi_14"
    return rsi


def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    macd_line.name = "macd"
    signal_line.name = "macd_signal"
    histogram.name = "macd_hist"
    return macd_line, signal_line, histogram


def _compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Compute Bollinger Bands."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = (mid + num_std * std).rename("bb_upper")
    lower = (mid - num_std * std).rename("bb_lower")
    width = ((upper - lower) / mid).rename("bb_width")
    pct_b = ((close - lower) / (upper - lower + 1e-12)).rename("bb_pct_b")
    return upper, lower, width, pct_b


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    atr.name = "atr_14"
    return atr


def _compute_drawdown_from_peak(close: pd.Series, window: int = 252) -> pd.Series:
    """Rolling drawdown from peak over trailing `window` bars."""
    roll_max = close.rolling(window, min_periods=1).max()
    dd = (close / roll_max) - 1.0
    dd.name = "drawdown_from_peak"
    return dd


def check_stationarity(series: pd.Series, significance: float = 0.05) -> bool:
    """
    ADF test for stationarity. Returns True if stationary, False otherwise.
    Logs a warning for non-stationary series.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series.dropna(), autolag="AIC")
        p_value = result[1]
        is_stationary = p_value < significance
        if not is_stationary:
            logger.warning(
                "Feature '%s' may be non-stationary (ADF p=%.4f). "
                "Consider differencing before using as LLM input.",
                series.name or "unknown", p_value
            )
        return is_stationary
    except ImportError:
        logger.debug("statsmodels not installed; skipping stationarity check.")
        return True
    except Exception as e:
        logger.debug("Stationarity check failed: %s", e)
        return True


# ---------------------------------------------------------------------------
# Main feature computation function
# ---------------------------------------------------------------------------

def compute_features(
    ohlcv_data: Dict[str, pd.DataFrame],
    ref_asset_ticker: str = "SPY",
    vix_ticker: str = "^VIX",
) -> pd.DataFrame:
    """
    Compute extended features for `ref_asset_ticker`.

    Features computed:
      - Realized vol: 5d, 21d, 63d (annualized)
      - Momentum: 21d, 63d, 252d
      - SMAs: 21, 50, 63, 200
      - Price vs SMA: 63, 200
      - RSI (14)
      - MACD (12, 26, 9)
      - Bollinger Bands (20, 2σ)
      - ATR (14)
      - Drawdown from 52-week high
      - VIX close (with forward-fill)

    Returns a DataFrame with single-level string columns.
    """
    if ref_asset_ticker not in ohlcv_data or ohlcv_data[ref_asset_ticker] is None:
        raise ValueError(f"Reference asset '{ref_asset_ticker}' not found in OHLCV data.")
    if ohlcv_data[ref_asset_ticker].empty:
        raise ValueError(f"Reference asset '{ref_asset_ticker}' has empty data.")

    raw_df = ohlcv_data[ref_asset_ticker].copy()

    # 1) Extract Close series robustly
    close_s = _find_field_series(raw_df, "Close")

    # Try to get High/Low for ATR
    try:
        high_s = _find_field_series(raw_df, "High")
    except KeyError:
        high_s = close_s.copy()
    try:
        low_s = _find_field_series(raw_df, "Low")
    except KeyError:
        low_s = close_s.copy()

    # 2) Flatten base DataFrame to single-level columns
    base_df = raw_df.copy()
    if isinstance(base_df.columns, pd.MultiIndex):
        base_df.columns = ["_".join(map(str, col)).strip() for col in base_df.columns]
    if "Close" not in base_df.columns:
        base_df = base_df.assign(Close=close_s)

    # 3) Compute features
    features: Dict[str, pd.Series] = {}

    # Realized vol (annualized)
    daily_ret = close_s.pct_change()
    features["volatility_5d"] = (daily_ret.rolling(5).std() * np.sqrt(252)).rename("volatility_5d")
    features["volatility_21d"] = (daily_ret.rolling(21).std() * np.sqrt(252)).rename("volatility_21d")
    features["volatility_63d"] = (daily_ret.rolling(63).std() * np.sqrt(252)).rename("volatility_63d")

    # Momentum
    features["momentum_21d"] = close_s.pct_change(21).rename("momentum_21d")
    features["momentum_63d"] = close_s.pct_change(63).rename("momentum_63d")
    features["momentum_252d"] = close_s.pct_change(252).rename("momentum_252d")

    # SMAs
    features["sma_21"] = close_s.rolling(21).mean().rename("sma_21")
    features["sma_50"] = close_s.rolling(50).mean().rename("sma_50")
    features["sma_63"] = close_s.rolling(63).mean().rename("sma_63")
    features["sma_200"] = close_s.rolling(200).mean().rename("sma_200")

    # Price vs SMA
    features["price_vs_sma63"] = (close_s / features["sma_63"] - 1).rename("price_vs_sma63")
    features["price_vs_sma200"] = (close_s / features["sma_200"] - 1).rename("price_vs_sma200")

    # RSI
    features["rsi_14"] = _compute_rsi(close_s, 14)

    # MACD
    macd, macd_sig, macd_hist = _compute_macd(close_s)
    features["macd"] = macd
    features["macd_signal"] = macd_sig
    features["macd_hist"] = macd_hist

    # Bollinger Bands
    bb_upper, bb_lower, bb_width, bb_pct_b = _compute_bollinger(close_s)
    features["bb_upper"] = bb_upper
    features["bb_lower"] = bb_lower
    features["bb_width"] = bb_width
    features["bb_pct_b"] = bb_pct_b

    # ATR
    features["atr_14"] = _compute_atr(high_s, low_s, close_s)

    # Drawdown from 52-week high
    features["drawdown_from_peak"] = _compute_drawdown_from_peak(close_s)

    # 4) Assemble final DataFrame
    final_df = base_df.copy()
    for name, series in features.items():
        final_df[name] = series

    # 5) Attach VIX close
    if vix_ticker in ohlcv_data and ohlcv_data[vix_ticker] is not None and not ohlcv_data[vix_ticker].empty:
        vix_raw = ohlcv_data[vix_ticker].copy()
        try:
            vix_close = _find_field_series(vix_raw, "Close").rename("vix_close")
        except KeyError:
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = ["_".join(map(str, col)).strip() for col in vix_raw.columns]
            close_cols = [c for c in vix_raw.columns if "close" in str(c).lower()]
            vix_close = vix_raw[close_cols[0]].rename("vix_close") if close_cols else None

        if vix_close is not None:
            final_df["vix_close"] = vix_close
            final_df["vix_close"] = final_df["vix_close"].ffill()

    # 6) Drop NaN rows from rolling windows
    final_df = final_df.dropna()

    # 7) Stationarity check on key momentum features (informational)
    for feat in ["momentum_63d", "volatility_21d", "rsi_14"]:
        if feat in final_df.columns:
            check_stationarity(final_df[feat])

    logger.info("Features computed: %d rows, %d columns", len(final_df), len(final_df.columns))
    return final_df
