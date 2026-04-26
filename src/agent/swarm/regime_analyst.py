"""
Regime Analyst Agent — Enhanced Regime Analysis with Macro Context
===================================================================

Upgrades the single analyze_node into a dedicated agent that:
1. Runs the standard VIX-percentile + momentum regime detector
2. Adds a structured narrative (regime story, not just a label)
3. Optionally injects FRED macro context (yield curve, credit spreads)
4. Computes cross-asset correlation regime

This agent's output feeds EVERY specialist with richer context.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.agent.context_builder import RegimeContext, build_context
from src.agent.swarm.state import SwarmState
from src.features.engine import compute_features
from src.features.regime import detect_regime_full
from src.utils.config import config

logger = logging.getLogger(__name__)


def _build_narrative(signals, context: RegimeContext) -> str:
    """
    Build a rich natural-language regime description from quantitative signals.
    This is injected into each specialist's prompt as structured context.
    """
    parts = []

    # Volatility description
    vol_phrases = {
        "crisis": "📛 CRISIS: Extreme volatility (VIX top decile). High probability of fat-tail losses. Favour capital preservation over alpha generation.",
        "high": "⚠️ HIGH VOL: Elevated volatility (VIX above 65th percentile). Momentum strategies lose edge; mean reversion more reliable.",
        "mid": "🟡 MID VOL: Moderate volatility. Both momentum and mean reversion viable. Execution costs matter more.",
        "low": "🟢 LOW VOL: Calm conditions (VIX below 35th percentile). Trend-following and long-horizon momentum historically strongest.",
    }
    parts.append(vol_phrases.get(signals.vol_regime, f"Vol regime: {signals.vol_regime}"))

    # Trend description
    mom_pct = signals.momentum_63d * 100
    if signals.trend_regime == "Bull":
        parts.append(f"📈 TREND: Strong uptrend ({mom_pct:.1f}% 3M return). Above SMA-200: {context.price_vs_sma200 > 0}.")
    elif signals.trend_regime == "Bear":
        parts.append(f"📉 TREND: Downtrend ({mom_pct:.1f}% 3M return). Caution on long-only momentum.")
    else:
        parts.append(f"↔️ TREND: Neutral ({mom_pct:.1f}% 3M return). Range-bound conditions likely.")

    # Regime confidence
    conf_pct = signals.regime_confidence * 100
    if conf_pct > 75:
        parts.append(f"✅ CONFIDENCE: High ({conf_pct:.0f}%) — regime signal is clear and consistent.")
    elif conf_pct > 40:
        parts.append(f"🔶 CONFIDENCE: Moderate ({conf_pct:.0f}%) — regime on the boundary, apply wider parameter ranges.")
    else:
        parts.append(f"❓ CONFIDENCE: Low ({conf_pct:.0f}%) — mixed signals, diversify strategy types.")

    # VIX absolute level
    if signals.vix_level > 0:
        parts.append(f"VIX: {signals.vix_level:.1f} (at {signals.vix_percentile_252d:.0f}th pct of trailing 252d).")

    # Multi-horizon momentum alignment
    if hasattr(context, "momentum_21d") and hasattr(context, "momentum_252d"):
        short = context.momentum_21d * 100
        long_ = context.momentum_252d * 100
        if short > 0 and long_ > 0:
            parts.append(f"Momentum alignment: BULLISH (1M={short:.1f}%, 12M={long_:.1f}%).")
        elif short < 0 and long_ < 0:
            parts.append(f"Momentum alignment: BEARISH (1M={short:.1f}%, 12M={long_:.1f}%).")
        else:
            parts.append(f"Momentum alignment: DIVERGING (1M={short:.1f}%, 12M={long_:.1f}%) — transition risk.")

    return "\n".join(parts)


def _get_macro_context(ohlcv_data: Dict[str, pd.DataFrame]) -> str:
    """
    Attempt to pull macro context from FRED data (if available in ohlcv_data).
    Returns a formatted string or empty string if not available.
    """
    macro_lines = []

    # Check for yield curve proxy: TLT (long bonds) vs SHY (short bonds)
    if "TLT" in ohlcv_data and "SHY" in ohlcv_data:
        try:
            tlt_ret = ohlcv_data["TLT"]["Close"].pct_change(63).iloc[-1]
            macro_lines.append(f"Long-bond 3M return: {tlt_ret * 100:.1f}% (proxy for duration risk)")
        except Exception:
            pass

    # Check for credit spreads via HYG vs LQD
    if "HYG" in ohlcv_data and "LQD" in ohlcv_data:
        try:
            hyg = ohlcv_data["HYG"]["Close"].iloc[-1] / ohlcv_data["HYG"]["Close"].iloc[-63]
            lqd = ohlcv_data["LQD"]["Close"].iloc[-1] / ohlcv_data["LQD"]["Close"].iloc[-63]
            spread_proxy = (lqd - hyg)  # positive = credit stress
            macro_lines.append(f"Credit stress proxy (LQD-HYG 3M spread): {spread_proxy:.3f}")
        except Exception:
            pass

    return "\n".join(macro_lines) if macro_lines else "Macro context: not available (add TLT/SHY/HYG/LQD to universe for macro signals)."


def run_regime_analyst(state: SwarmState) -> SwarmState:
    """
    Regime Analyst Agent node.

    Reads: ohlcv_data, assets
    Writes: features_df, regime_context, regime_narrative, macro_summary
    """
    logger.info("=== [REGIME ANALYST] Building market context ===")

    ohlcv_data = state["ohlcv_data"]
    asset = state.get("assets", [config.reference_asset])[0]

    # Compute features
    features_df = compute_features(ohlcv_data, asset, config.vix_ticker)

    # Full regime signals (percentile-based)
    signals = detect_regime_full(features_df)

    # Build RegimeContext for downstream specialists
    context = build_context(features_df)
    context.regime_label = signals.regime_label

    # Build rich narrative
    narrative = _build_narrative(signals, context)

    # Macro context (opportunistic)
    macro_summary = _get_macro_context(ohlcv_data)

    log_msg = (
        f"Regime: {signals.regime_label} | "
        f"VIX={signals.vix_level:.1f} ({signals.vix_percentile_252d:.0f}th pct) | "
        f"Momentum={signals.momentum_63d * 100:.1f}% | "
        f"Confidence={signals.regime_confidence:.0%}"
    )
    logger.info(log_msg)

    state["features_df"] = features_df
    state["regime_context"] = context
    state["regime_narrative"] = narrative
    state["macro_summary"] = macro_summary
    state.setdefault("run_log", []).append(f"[Regime Analyst] {log_msg}")

    return state
