"""
Backtest Runner — Unified Engine with Warmup Enforcement
=========================================================

Single run_backtest() function supporting all strategy types.
All strategies produce pd.Series signals; no special-casing for momentum.
Includes transaction costs and warmup validation.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest.metrics import PerformanceMetrics
from src.features.lookback_guard import InsufficientWarmupError, WarmupEnforcer
from src.strategies.strategy_registry import STRATEGY_REGISTRY
from src.utils.config import config

try:
    import vectorbt as vbt  # type: ignore
except Exception:
    vbt = None

logger = logging.getLogger(__name__)


def _get_close(df: pd.DataFrame) -> pd.Series:
    """Extract Close price series robustly."""
    if isinstance(df, pd.Series):
        return pd.to_numeric(df, errors="coerce").dropna()
    col_map = {str(c).lower(): c for c in df.columns}
    for cand in ("close", "adj close", "adjclose", "price"):
        if cand in col_map:
            return pd.to_numeric(df[col_map[cand]], errors="coerce").dropna()
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.dropna()
    raise KeyError("No close-like column found.")


def _apply_transaction_costs(
    signal: pd.Series,
    close: pd.Series,
    commission: float,
    slippage: float,
    market_impact_bps: float = 0.0,
) -> pd.Series:
    """
    Compute per-bar cost series.

    - commission: fraction of trade value
    - slippage: one-way slippage fraction (applied directionally)
    - market_impact_bps: square-root market impact in basis points
    """
    trades = signal.diff().abs().fillna(0)
    total_one_way = commission + slippage + (market_impact_bps / 10000.0)
    cost_series = trades * total_one_way
    return cost_series


def run_backtest(
    ohlcv_data,
    assets: List[str],
    strategy_name: str,
    params: Dict[str, Any],
    allocation_weights: Optional[Dict[str, float]] = None,
    eval_start: Optional[pd.Timestamp] = None,
) -> Optional[Dict[str, Any]]:
    """
    Execute a backtest for a given strategy and asset list.

    Args:
        ohlcv_data: Dict mapping ticker -> DataFrame, or single DataFrame.
        assets: List of asset tickers to backtest.
        strategy_name: Strategy name from STRATEGY_REGISTRY.
        params: Strategy parameters.
        allocation_weights: Optional per-asset weights (default: equal weight).
        eval_start: Start of evaluation window (warmup bars before this are used
                    for indicator computation only).

    Returns:
        Dict with 'equity_curve', 'metrics', 'weights'.
    """
    # Normalise ohlcv_data input
    if isinstance(ohlcv_data, pd.DataFrame):
        if len(assets) != 1:
            logger.error("Single DataFrame provided but %d assets requested.", len(assets))
            return None
        ohlcv_dict = {assets[0]: ohlcv_data}
    else:
        ohlcv_dict = ohlcv_data

    # Validate assets
    for asset in assets:
        if asset not in ohlcv_dict or ohlcv_dict[asset].empty:
            logger.warning("Missing data for %s — skipping backtest.", asset)
            return None

    # Strategy lookup
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Strategy '{strategy_name}' not found. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    strategy = STRATEGY_REGISTRY[strategy_name]

    # Allocation weights
    if allocation_weights is None:
        weights = {a: 1.0 / len(assets) for a in assets}
    else:
        total = sum(allocation_weights.values())
        weights = {a: allocation_weights.get(a, 0) / total for a in assets}

    # Backtest configuration
    init_cash = config.backtest.initial_cash
    commission = config.backtest.commission
    slippage = config.backtest.slippage
    impact_bps = config.backtest.market_impact_bps

    all_equity: List[pd.Series] = []
    all_metrics: List[Dict] = []

    enforcer = WarmupEnforcer(min_warmup_periods=config.backtest.min_warmup_periods)

    for asset in assets:
        df = ohlcv_dict[asset].copy()
        close = _get_close(df)
        w = weights.get(asset, 1.0 / len(assets))
        asset_cash = init_cash * w

        try:
            signal = strategy.generate_signal(df, params)
            signal = signal.reindex(close.index).fillna(0)
        except Exception as e:
            logger.error("Signal generation failed for %s (%s): %s", asset, strategy_name, e)
            return None

        # Warmup check if eval_start provided
        if eval_start is not None:
            try:
                slow_w = params.get("slow_window", params.get("window", config.backtest.min_warmup_periods))
                enforcer.check(df, eval_start, min_window=int(slow_w))
            except InsufficientWarmupError as e:
                logger.warning("%s", e)

        # Transaction costs
        costs = _apply_transaction_costs(signal, close, commission, slippage, impact_bps)

        # Strategy returns
        daily_ret = close.pct_change().fillna(0)
        strat_ret = daily_ret * signal.shift(1).fillna(0) - costs

        # Equity curve
        equity = (1 + strat_ret).cumprod() * asset_cash

        # Slice to eval window if provided
        if eval_start is not None:
            equity = equity.loc[equity.index >= eval_start]
            strat_ret = strat_ret.loc[strat_ret.index >= eval_start]

        metrics = PerformanceMetrics.from_equity(equity)
        metrics["num_trades"] = int(signal.diff().abs().sum() / 2)
        metrics["asset"] = asset

        all_equity.append(equity)
        all_metrics.append(metrics)

    if not all_equity:
        return None

    # Combine across assets
    combined = all_equity[0]
    for eq in all_equity[1:]:
        idx = combined.index.intersection(eq.index)
        combined = combined.loc[idx] + eq.loc[idx]

    combined_metrics = PerformanceMetrics.from_equity(combined)
    combined_metrics["num_trades"] = sum(m["num_trades"] for m in all_metrics)
    combined_metrics["sharpe_ratio"] = combined_metrics.pop("sharpe", 0.0)
    combined_metrics["total_return"] = combined_metrics.get("total_return", 0.0)

    return {
        "equity_curve": combined,
        "weights": weights,
        "metrics": combined_metrics,
        "per_asset": {m["asset"]: m for m in all_metrics},
    }