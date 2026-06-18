"""Regime analyst agent for the multi-agent swarm."""

import logging
from typing import Dict

import pandas as pd

from src.agent.context_builder import RegimeContext, build_context
from src.agent.swarm.state import SwarmState
from src.features.engine import compute_features
from src.features.regime import detect_regime_full
from src.utils.config import config

logger = logging.getLogger(__name__)


def _build_narrative(signals, context: RegimeContext) -> str:
    parts = []
    vol_text = {
        "crisis": "CRISIS VOL: volatility sits in the upper tail; prefer capital preservation and shorter confirmation windows.",
        "high": "HIGH VOL: elevated volatility; require stricter drawdown controls and avoid fragile momentum settings.",
        "mid": "MID VOL: balanced conditions; compare momentum and mean reversion directly.",
        "low": "LOW VOL: calm conditions; trend following and long-horizon momentum deserve priority.",
    }
    parts.append(vol_text.get(signals.vol_regime, f"Volatility regime: {signals.vol_regime}."))

    momentum_pct = signals.momentum_63d * 100
    if signals.momentum_63d > 0.03:
        parts.append(f"TREND: positive 3M momentum ({momentum_pct:.1f}%).")
    elif signals.momentum_63d < -0.03:
        parts.append(f"TREND: negative 3M momentum ({momentum_pct:.1f}%).")
    else:
        parts.append(f"TREND: neutral 3M momentum ({momentum_pct:.1f}%).")

    parts.append(
        f"VIX: {signals.vix_level:.1f}, trailing percentile {signals.vix_percentile_252d:.0f}, "
        f"regime confidence {signals.regime_confidence:.0%}."
    )
    parts.append(
        "Momentum alignment: "
        f"1M={context.momentum_21d * 100:.1f}%, "
        f"3M={context.momentum_63d * 100:.1f}%, "
        f"12M={context.momentum_252d * 100:.1f}%."
    )
    return "\n".join(parts)


def _get_macro_context(ohlcv_data: Dict[str, pd.DataFrame]) -> str:
    lines = []
    if "TLT" in ohlcv_data:
        try:
            tlt_return = ohlcv_data["TLT"]["Close"].pct_change(63).iloc[-1]
            lines.append(f"Long-bond 3M return proxy: {tlt_return * 100:.1f}%.")
        except Exception:
            pass
    if "HYG" in ohlcv_data and "LQD" in ohlcv_data:
        try:
            hyg = ohlcv_data["HYG"]["Close"].iloc[-1] / ohlcv_data["HYG"]["Close"].iloc[-63]
            lqd = ohlcv_data["LQD"]["Close"].iloc[-1] / ohlcv_data["LQD"]["Close"].iloc[-63]
            lines.append(f"Credit stress proxy (LQD-HYG 3M spread): {lqd - hyg:.3f}.")
        except Exception:
            pass
    return "\n".join(lines) if lines else "Macro context unavailable from current universe."


def run_regime_analyst(state: SwarmState) -> SwarmState:
    """Build the market context consumed by all downstream agents."""
    ohlcv_data = state["ohlcv_data"]
    asset = state.get("assets", [config.reference_asset])[0]
    features_df = compute_features(ohlcv_data, asset, config.vix_ticker)
    signals = detect_regime_full(features_df)
    context = build_context(features_df)
    context.regime_label = signals.regime_label
    context.regime_confidence = signals.regime_confidence

    narrative = _build_narrative(signals, context)
    state["features_df"] = features_df
    state["regime_context"] = context
    state["regime_narrative"] = narrative
    state["macro_summary"] = _get_macro_context(ohlcv_data)
    state.setdefault("run_log", []).append(
        f"[Regime Analyst] {signals.regime_label} | VIX={signals.vix_level:.1f} | "
        f"confidence={signals.regime_confidence:.0%}"
    )
    logger.info(state["run_log"][-1])
    return state
