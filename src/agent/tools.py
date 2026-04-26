"""
Agent Tools — LangChain/LangGraph Tool Definitions
====================================================

Proper tool-calling interface so the LLM can introspect the strategy
registry and execute backtests.
"""

import json
import logging
from typing import Any, Dict, List

from src.utils.config import config

logger = logging.getLogger(__name__)


def list_available_strategies() -> List[Dict[str, Any]]:
    """List all available strategies and their parameter schemas."""
    from src.strategies.strategy_registry import STRATEGY_REGISTRY
    result = []
    for name, strategy in STRATEGY_REGISTRY.items():
        result.append({
            "name": name,
            "param_schema": strategy.param_schema,
            "description": strategy.__class__.__doc__ or "",
        })
    return result


def get_strategy_schema(name: str) -> Dict[str, Any]:
    """Get the parameter schema for a specific strategy."""
    from src.strategies.strategy_registry import STRATEGY_REGISTRY
    if name not in STRATEGY_REGISTRY:
        return {"error": f"Strategy '{name}' not found. Available: {list(STRATEGY_REGISTRY.keys())}"}
    strategy = STRATEGY_REGISTRY[name]
    return {"name": name, "param_schema": strategy.param_schema}


def get_market_summary() -> str:
    """Get current market regime context as a string."""
    from src.agent.context_builder import build_context
    from src.data.ingest import fetch_ohlcv_data
    from src.features.engine import compute_features

    ohlcv = fetch_ohlcv_data()
    ref = config.reference_asset
    if ref not in ohlcv:
        return "Error: reference asset data not available."

    features = compute_features(ohlcv, ref, config.vix_ticker)
    ctx = build_context(features)
    return ctx.to_prompt_string()


def run_backtest_tool(
    strategy_name: str,
    params: Dict[str, Any],
    asset: str = None,
) -> Dict[str, Any]:
    """Run a backtest and return metrics."""
    from src.backtest.runner import run_backtest

    asset = asset or config.reference_asset
    from src.data.ingest import fetch_ohlcv_data
    ohlcv = fetch_ohlcv_data()

    if asset not in ohlcv:
        return {"error": f"Asset '{asset}' data not available."}

    try:
        result = run_backtest(ohlcv, [asset], strategy_name, params)
        if result and "metrics" in result:
            return result["metrics"]
        return {"error": "Backtest returned no results."}
    except Exception as e:
        return {"error": str(e)}
