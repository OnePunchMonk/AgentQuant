"""
Strategy Base Class — Unified Interface
========================================

All strategies implement generate_signal() returning pd.Series of {-1, 0, 1}.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def _get_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Extract a column from df by trying candidate names (case-insensitive)."""
    col_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        ln = str(name).lower()
        if ln in col_map:
            return pd.to_numeric(df[col_map[ln]], errors="coerce").dropna()
    # Fallback: first numeric column
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            return s.dropna()
    raise KeyError(f"Could not find any of {candidates} in {list(df.columns)[:10]}")


def _get_close(df: pd.DataFrame) -> pd.Series:
    return _get_col(df, "close", "adj close", "adjclose", "price")


def _get_high(df: pd.DataFrame) -> pd.Series:
    try:
        return _get_col(df, "high")
    except KeyError:
        return _get_close(df)


def _get_low(df: pd.DataFrame) -> pd.Series:
    try:
        return _get_col(df, "low")
    except KeyError:
        return _get_close(df)


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate a trading signal.

        Returns:
            pd.Series indexed like df.index, values in {-1, 0, 1}.
            Long = 1, Flat = 0, Short = -1.
        """
        ...

    @property
    @abstractmethod
    def param_schema(self) -> Dict[str, Any]:
        """JSON-serialisable schema for valid parameters."""
        ...


class MomentumStrategy(Strategy):
    """Dual moving-average crossover strategy."""

    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close = _get_close(df)
        fast = int(params.get("fast_window", 21))
        slow = int(params.get("slow_window", 63))
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1
        return signal.reindex(df.index).fillna(0)

    @property
    def param_schema(self) -> Dict[str, Any]:
        return {
            "fast_window": {"type": "int", "min": 5, "max": 100},
            "slow_window": {"type": "int", "min": 20, "max": 300},
            "constraint": "fast_window < slow_window",
        }


class MeanReversionStrategy(Strategy):
    """Bollinger Band mean reversion strategy."""

    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close = _get_close(df)
        window = int(params.get("window", 20))
        num_std = float(params.get("num_std", 2.0))
        mid = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[close < lower] = 1    # oversold → buy
        signal[close > upper] = -1   # overbought → sell
        return signal.reindex(df.index).fillna(0)

    @property
    def param_schema(self) -> Dict[str, Any]:
        return {
            "window": {"type": "int", "min": 5, "max": 100},
            "num_std": {"type": "float", "min": 0.5, "max": 4.0},
        }


class VolatilityStrategy(Strategy):
    """Go long when realized volatility is below threshold."""

    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close = _get_close(df)
        window = int(params.get("window", 21))
        threshold = float(params.get("vol_threshold", 0.20))
        daily_vol = close.pct_change().rolling(window).std() * (252 ** 0.5)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[daily_vol < threshold] = 1
        return signal.reindex(df.index).fillna(0)

    @property
    def param_schema(self) -> Dict[str, Any]:
        return {
            "window": {"type": "int", "min": 5, "max": 126},
            "vol_threshold": {"type": "float", "min": 0.05, "max": 0.60},
        }


class TrendFollowingStrategy(Strategy):
    """Triple moving-average trend-following strategy."""

    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close = _get_close(df)
        sw = int(params.get("short_window", 10))
        mw = int(params.get("medium_window", 50))
        lw = int(params.get("long_window", 100))
        short_ma = close.rolling(sw).mean()
        medium_ma = close.rolling(mw).mean()
        long_ma = close.rolling(lw).mean()
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[(short_ma > medium_ma) & (medium_ma > long_ma)] = 1
        signal[(short_ma < medium_ma) & (medium_ma < long_ma)] = -1
        return signal.reindex(df.index).fillna(0)

    @property
    def param_schema(self) -> Dict[str, Any]:
        return {
            "short_window": {"type": "int", "min": 5, "max": 50},
            "medium_window": {"type": "int", "min": 20, "max": 100},
            "long_window": {"type": "int", "min": 50, "max": 300},
        }


class BreakoutStrategy(Strategy):
    """Price breakout above rolling high / below rolling low."""

    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close = _get_close(df)
        high = _get_high(df).reindex(close.index).ffill()
        low = _get_low(df).reindex(close.index).ffill()
        window = int(params.get("window", 20))
        threshold = float(params.get("threshold_pct", 0.02))
        roll_high = high.rolling(window).max()
        roll_low = low.rolling(window).min()
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[close > roll_high * (1 + threshold)] = 1
        signal[close < roll_low * (1 - threshold)] = -1
        return signal.reindex(df.index).fillna(0)

    @property
    def param_schema(self) -> Dict[str, Any]:
        return {
            "window": {"type": "int", "min": 5, "max": 252},
            "threshold_pct": {"type": "float", "min": 0.001, "max": 0.10},
        }


class RegimeBasedStrategy(Strategy):
    """Switches between momentum and mean reversion based on market regime."""

    def generate_signal(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        regime = str(params.get("regime_data", "neutral")).lower()
        mom_params = params.get("momentum_params", {"fast_window": 21, "slow_window": 63})
        mr_params = params.get("mean_reversion_params", {"window": 20, "num_std": 2.0})

        if "bull" in regime or "uptrend" in regime:
            return MomentumStrategy().generate_signal(df, mom_params)
        elif "bear" in regime or "downtrend" in regime:
            return MeanReversionStrategy().generate_signal(df, mr_params)
        elif "crisis" in regime or "highvol" in regime:
            return VolatilityStrategy().generate_signal(df, {"window": 21, "vol_threshold": 0.30})
        else:
            return MomentumStrategy().generate_signal(df, mom_params)

    @property
    def param_schema(self) -> Dict[str, Any]:
        return {
            "regime_data": {"type": "str"},
            "momentum_params": {"type": "dict"},
            "mean_reversion_params": {"type": "dict"},
        }
