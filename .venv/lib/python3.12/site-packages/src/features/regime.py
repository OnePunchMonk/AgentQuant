"""
Regime Detection — Percentile-Based Market Regime Classification
================================================================

Uses VIX percentile (relative to trailing 252d) instead of absolute
thresholds. Optionally uses HMM for probabilistic regime switching.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class RegimeSignals:
    """Full set of regime signals for the latest observation."""
    # Trend
    above_200sma: bool = False
    above_50sma: bool = False
    price_vs_200sma_pct: float = 0.0
    # Volatility
    vix_level: float = 20.0
    vix_percentile_252d: float = 50.0
    realized_vol_21d: float = 0.0
    vol_regime: str = "mid"   # "low" | "mid" | "high" | "crisis"
    # Momentum
    momentum_21d: float = 0.0
    momentum_63d: float = 0.0
    momentum_252d: float = 0.0
    # Drawdown
    drawdown_from_52w_high: float = 0.0
    # Derived
    regime_label: str = "Unknown"
    regime_confidence: float = 0.5


def detect_regime(features_df: pd.DataFrame) -> str:
    """
    Detects the current market regime label.

    Uses VIX percentile (relative, not absolute) and multi-horizon momentum.

    Returns:
        str: Regime label e.g. "LowVol-Bull", "Crisis-Bear".
    """
    signals = detect_regime_full(features_df)
    return signals.regime_label


def detect_regime_full(features_df: pd.DataFrame) -> RegimeSignals:
    """
    Full regime detection returning all signals.

    Returns:
        RegimeSignals dataclass with all computed signals.
    """
    if features_df.empty:
        return RegimeSignals(regime_label="Unknown")

    latest = features_df.iloc[-1]
    signals = RegimeSignals()

    # --- VIX (percentile-based) ---
    vix = latest.get("vix_close", 20.0)
    signals.vix_level = float(vix) if not pd.isna(vix) else 20.0

    if "vix_close" in features_df.columns:
        vix_history = features_df["vix_close"].dropna().tail(252)
        if len(vix_history) > 10:
            signals.vix_percentile_252d = float(
                scipy_stats.percentileofscore(vix_history, signals.vix_level)
            )
        else:
            signals.vix_percentile_252d = 50.0

    # Vol regime buckets from percentile
    vp = signals.vix_percentile_252d
    if vp > 85:
        signals.vol_regime = "crisis"
        vol_label = "Crisis"
    elif vp > 65:
        signals.vol_regime = "high"
        vol_label = "HighVol"
    elif vp > 35:
        signals.vol_regime = "mid"
        vol_label = "MidVol"
    else:
        signals.vol_regime = "low"
        vol_label = "LowVol"

    # Confidence: distance from 50th percentile
    vol_confidence = 2.0 * abs(vp / 100.0 - 0.5)

    # --- Momentum ---
    signals.momentum_21d = float(latest.get("momentum_21d", 0.0) or 0.0)
    signals.momentum_63d = float(latest.get("momentum_63d", 0.0) or 0.0)
    signals.momentum_252d = float(latest.get("momentum_252d", 0.0) or 0.0)

    mom = signals.momentum_63d
    if mom > 0.05:
        trend_label = "Bull"
    elif mom < -0.05:
        trend_label = "Bear"
    else:
        trend_label = "Neutral"

    mom_confidence = min(abs(mom) / 0.10, 1.0)

    # --- Realized vol ---
    signals.realized_vol_21d = float(latest.get("volatility_21d", 0.0) or 0.0)

    # --- Trend (SMA) ---
    close = latest.get("Close", None)
    sma200 = latest.get("sma_200", None)
    sma50 = latest.get("sma_50", None)

    if close is not None and sma200 is not None and not pd.isna(sma200) and float(sma200) > 0:
        pct = float(close) / float(sma200) - 1.0
        signals.price_vs_200sma_pct = pct
        signals.above_200sma = pct > 0
    if close is not None and sma50 is not None and not pd.isna(sma50) and float(sma50) > 0:
        signals.above_50sma = float(close) > float(sma50)

    # --- Drawdown from 52w high ---
    if "Close" in features_df.columns:
        close_series = features_df["Close"].dropna().tail(252)
        if len(close_series) > 1:
            peak = close_series.max()
            latest_close = close_series.iloc[-1]
            signals.drawdown_from_52w_high = (latest_close / peak) - 1.0

    # --- Optional HMM regime detection ---
    hmm_label = _try_hmm_regime(features_df)
    if hmm_label:
        logger.debug("HMM regime suggestion: %s", hmm_label)

    # --- Final label and confidence ---
    signals.regime_label = f"{vol_label}-{trend_label}"
    signals.regime_confidence = (vol_confidence + mom_confidence) / 2.0

    logger.info(
        "Regime detected: %s (VIX=%.1f at %.0fth pct, mom63d=%.1f%%, confidence=%.0f%%)",
        signals.regime_label, signals.vix_level, signals.vix_percentile_252d,
        signals.momentum_63d * 100, signals.regime_confidence * 100,
    )

    return signals


def _try_hmm_regime(features_df: pd.DataFrame, n_states: int = 3) -> Optional[str]:
    """
    Attempt HMM-based regime detection. Returns regime label or None if unavailable.
    hmmlearn is optional — falls back to rule-based if not installed.
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        return None

    try:
        # Features for HMM: returns and realized vol
        if "Close" not in features_df.columns or len(features_df) < 60:
            return None

        close = features_df["Close"].dropna()
        returns = close.pct_change().dropna()
        vol = returns.rolling(21).std().dropna()
        aligned = pd.concat([returns, vol], axis=1).dropna()
        if len(aligned) < 60:
            return None

        X = aligned.values
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full",
                                  n_iter=100, random_state=42)
        model.fit(X)
        states = model.predict(X)
        current_state = int(states[-1])

        # Label states by volatility (mean of second feature = realized vol)
        state_vols = {}
        for s in range(n_states):
            mask = states == s
            if mask.sum() > 0:
                state_vols[s] = float(X[mask, 1].mean())

        sorted_states = sorted(state_vols, key=state_vols.get)
        labels = {sorted_states[0]: "LowVol", sorted_states[-1]: "HighVol"}
        if len(sorted_states) > 2:
            labels[sorted_states[1]] = "MidVol"

        return labels.get(current_state, "Unknown")
    except Exception as e:
        logger.debug("HMM regime detection failed: %s", e)
        return None