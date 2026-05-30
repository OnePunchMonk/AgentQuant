"""
Context Builder — Structured Regime Context for LLM Prompts
============================================================

Builds a rich RegimeContext dataclass from features DataFrame,
providing structured information the LLM can reason about.
"""

import logging
from dataclasses import dataclass

import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class RegimeContext:
    """Structured market context for LLM consumption."""

    regime_label: str = "Unknown"
    vix_level: float = 20.0
    vix_percentile: float = 50.0  # percentile vs trailing 252d
    momentum_21d: float = 0.0
    momentum_63d: float = 0.0
    momentum_252d: float = 0.0
    realized_vol_5d: float = 0.0
    realized_vol_21d: float = 0.0
    realized_vol_63d: float = 0.0
    vol_vs_avg: float = 1.0  # ratio: current 21d vol / 252d average vol
    ma_regime: str = "unknown"  # "above_200", "below_200", "near_200"
    drawdown_from_peak: float = 0.0
    rsi_14: float = 50.0
    price_vs_sma200: float = 0.0
    regime_confidence: float = 0.5
    alpha_memory_context: str = ""
    nla_memory_context: str = ""

    def to_prompt_string(self) -> str:
        """Format context as structured text for LLM prompt injection."""
        context = (
            f"MARKET CONTEXT:\n"
            f"  Regime: {self.regime_label} (confidence: {self.regime_confidence:.0%})\n"
            f"  VIX: {self.vix_level:.1f} (at {self.vix_percentile:.0f}th percentile, trailing 1Y)\n"
            f"  Momentum:\n"
            f"    1-Month: {self.momentum_21d * 100:.1f}%\n"
            f"    3-Month: {self.momentum_63d * 100:.1f}%\n"
            f"    12-Month: {self.momentum_252d * 100:.1f}%\n"
            f"  Volatility:\n"
            f"    Realized Vol (21d ann.): {self.realized_vol_21d * 100:.1f}%\n"
            f"    Vol vs 1Y avg: {self.vol_vs_avg:.2f}x\n"
            f"  Trend:\n"
            f"    Price vs 200SMA: {self.ma_regime} ({self.price_vs_sma200 * 100:+.1f}%)\n"
            f"    RSI (14): {self.rsi_14:.1f}\n"
            f"  Drawdown from peak: {self.drawdown_from_peak * 100:.1f}%\n"
        )
        if self.alpha_memory_context:
            context += f"\n{self.alpha_memory_context}\n"
        if self.nla_memory_context:
            context += f"\n{self.nla_memory_context}\n"
        return context


def build_context(features_df: pd.DataFrame) -> RegimeContext:
    """
    Build a RegimeContext from the features DataFrame.

    Uses the latest row of features_df plus rolling calculations
    for percentile-based measures.
    """
    if features_df.empty:
        logger.warning("Empty features_df passed to build_context, returning defaults.")
        return RegimeContext()

    latest = features_df.iloc[-1]
    ctx = RegimeContext()

    # VIX
    vix = latest.get("vix_close", 20.0)
    ctx.vix_level = float(vix) if not pd.isna(vix) else 20.0

    # VIX percentile over trailing 252 days
    if "vix_close" in features_df.columns:
        vix_history = features_df["vix_close"].dropna().tail(252)
        if len(vix_history) > 10:
            ctx.vix_percentile = float(
                scipy_stats.percentileofscore(vix_history, ctx.vix_level)
            )
        else:
            ctx.vix_percentile = 50.0
    else:
        ctx.vix_percentile = 50.0

    # Momentum
    ctx.momentum_21d = float(latest.get("momentum_21d", 0.0) or 0.0)
    ctx.momentum_63d = float(latest.get("momentum_63d", 0.0) or 0.0)
    ctx.momentum_252d = float(latest.get("momentum_252d", 0.0) or 0.0)

    # Realized vol
    ctx.realized_vol_5d = float(latest.get("volatility_5d", 0.0) or 0.0)
    ctx.realized_vol_21d = float(latest.get("volatility_21d", 0.0) or 0.0)
    ctx.realized_vol_63d = float(latest.get("volatility_63d", 0.0) or 0.0)

    # Vol vs average
    if "volatility_21d" in features_df.columns:
        vol_history = features_df["volatility_21d"].dropna().tail(252)
        avg_vol = vol_history.mean() if len(vol_history) > 10 else ctx.realized_vol_21d
        ctx.vol_vs_avg = ctx.realized_vol_21d / avg_vol if avg_vol > 0 else 1.0
    else:
        ctx.vol_vs_avg = 1.0

    # MA regime
    close = latest.get("Close", None)
    sma200 = latest.get("sma_200", None)
    if close is not None and sma200 is not None and not pd.isna(sma200) and sma200 > 0:
        pct = (float(close) / float(sma200)) - 1.0
        ctx.price_vs_sma200 = pct
        if pct > 0.02:
            ctx.ma_regime = "above_200"
        elif pct < -0.02:
            ctx.ma_regime = "below_200"
        else:
            ctx.ma_regime = "near_200"
    else:
        ctx.ma_regime = "unknown"
        ctx.price_vs_sma200 = 0.0

    # Drawdown from peak
    dd = latest.get("drawdown_from_peak", 0.0)
    ctx.drawdown_from_peak = float(dd) if not pd.isna(dd) else 0.0

    # RSI
    rsi = latest.get("rsi_14", 50.0)
    ctx.rsi_14 = float(rsi) if not pd.isna(rsi) else 50.0

    # Regime label and confidence (from regime detector)
    ctx.regime_label = _classify_regime(ctx)
    ctx.regime_confidence = _compute_confidence(ctx)

    return ctx


def _classify_regime(ctx: RegimeContext) -> str:
    """Classify regime using VIX percentile instead of absolute levels."""
    vix_pct = ctx.vix_percentile

    if vix_pct > 85:
        vol_label = "Crisis"
    elif vix_pct > 65:
        vol_label = "HighVol"
    elif vix_pct > 35:
        vol_label = "MidVol"
    else:
        vol_label = "LowVol"

    mom = ctx.momentum_63d
    if mom > 0.05:
        trend_label = "Bull"
    elif mom < -0.05:
        trend_label = "Bear"
    else:
        trend_label = "Neutral"

    return f"{vol_label}-{trend_label}"


def _compute_confidence(ctx: RegimeContext) -> float:
    """
    Confidence score: how clearly we're in one regime vs on a boundary.
    Higher distance from 50th percentile VIX = more confident vol regime.
    Higher absolute momentum = more confident trend regime.
    """
    vol_confidence = 2 * abs(ctx.vix_percentile / 100.0 - 0.5)
    mom_confidence = min(abs(ctx.momentum_63d) / 0.10, 1.0)
    return (vol_confidence + mom_confidence) / 2.0
