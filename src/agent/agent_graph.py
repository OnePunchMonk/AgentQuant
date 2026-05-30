"""
Agent Graph — ReAct Agent Loop with LangGraph StateGraph
==========================================================

Implements the real agentic loop:
  analyze → hypothesize → backtest → reflect → (loop or store)

The agent iterates up to max_iterations times, improving proposals
based on backtest results.
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from src.agent.context_builder import RegimeContext, build_context
from src.agent.proposal_generator import Proposal, ProposalGenerator
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.research.alpha_store import AlphaStore
from src.research.nla_memory import NLAMemoryStore
from src.utils.config import config

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """State flowing through the agent graph."""
    ohlcv_data: Dict[str, pd.DataFrame]
    features_df: pd.DataFrame
    context: Optional[RegimeContext]
    proposals: List[Proposal]
    results: List[Dict[str, Any]]
    best_result: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int
    strategy_type: str
    asset: str
    should_continue: bool
    memory_context: str
    run_log: List[str]


def analyze_node(state: AgentState) -> AgentState:
    """Build regime context from market data."""
    logger.info("=== ANALYZE: Building market context ===")
    from src.features.engine import compute_features
    from src.features.regime import detect_regime

    ohlcv = state["ohlcv_data"]
    asset = state.get("asset", config.reference_asset)

    features_df = compute_features(ohlcv, asset, config.vix_ticker)
    context = build_context(features_df)
    regime_label = detect_regime(features_df)
    context.regime_label = regime_label

    # Get memory context
    memory = StrategyMemory()
    memory_ctx = memory.to_prompt_context(regime_label, state.get("strategy_type", "momentum"))
    alpha_memory = AlphaStore()
    alpha_ctx = alpha_memory.to_prompt_context(regime_label, state.get("strategy_type", "momentum"))
    nla_memory = NLAMemoryStore()
    nla_ctx = nla_memory.to_prompt_context(regime_label, state.get("strategy_type", "momentum"))
    context.alpha_memory_context = alpha_ctx
    context.nla_memory_context = nla_ctx

    state["features_df"] = features_df
    state["context"] = context
    state["memory_context"] = f"{memory_ctx}\n\n{alpha_ctx}\n\n{nla_ctx}"
    state["run_log"] = state.get("run_log", [])
    state["run_log"].append(f"Regime: {regime_label} (confidence: {context.regime_confidence:.0%})")

    logger.info("Regime: %s, Confidence: %.0f%%", regime_label, context.regime_confidence * 100)
    return state


def hypothesize_node(state: AgentState) -> AgentState:
    """Generate strategy proposals via LLM or grid search."""
    iteration = state.get("iteration", 0) + 1
    state["iteration"] = iteration
    logger.info("=== HYPOTHESIZE (iteration %d): Generating proposals ===", iteration)

    generator = ProposalGenerator()
    strategy_type = state.get("strategy_type", "momentum")
    context = state["context"]

    proposals = generator.generate(
        context=context,
        n_proposals=5,
        strategy_type=strategy_type,
    )

    state["proposals"] = proposals
    state["run_log"].append(
        f"Iteration {iteration}: Generated {len(proposals)} proposals "
        f"(methods: {[p.generation_method for p in proposals]})"
    )

    for i, p in enumerate(proposals):
        logger.info("  Proposal %d: %s (confidence=%.2f, method=%s)",
                     i + 1, p.params, p.confidence, p.generation_method)
    return state


def backtest_node(state: AgentState) -> AgentState:
    """Run backtests on all proposals and rank by Sharpe."""
    logger.info("=== BACKTEST: Running tournament ===")
    from src.backtest.runner import run_backtest

    ohlcv = state["ohlcv_data"]
    asset = state.get("asset", config.reference_asset)
    strategy_type = state.get("strategy_type", "momentum")

    results = []
    for i, proposal in enumerate(state["proposals"]):
        try:
            bt_result = run_backtest(ohlcv, [asset], strategy_type, proposal.params)
            if bt_result and "metrics" in bt_result:
                metrics = bt_result["metrics"]
                results.append({
                    "proposal_idx": i,
                    "params": proposal.params,
                    "sharpe": metrics.get("sharpe_ratio", 0.0),
                    "total_return": metrics.get("total_return", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "num_trades": metrics.get("num_trades", 0),
                    "generation_method": proposal.generation_method,
                    "confidence": proposal.confidence,
                    "reasoning": proposal.reasoning,
                })
        except Exception as e:
            logger.warning("Backtest failed for proposal %d: %s", i, e)

    # Sort by Sharpe
    results.sort(key=lambda x: x.get("sharpe", 0.0), reverse=True)
    state["results"] = results

    if results:
        best = results[0]
        state["best_result"] = best
        state["run_log"].append(
            f"Best: Sharpe={best['sharpe']:.2f}, Return={best['total_return']:.1%}, "
            f"Params={best['params']}"
        )
        logger.info("Best result: Sharpe=%.2f, Return=%.1f%%, Params=%s",
                     best["sharpe"], best["total_return"] * 100, best["params"])
    else:
        state["best_result"] = None
        state["run_log"].append("No valid backtest results.")
        logger.warning("No valid backtest results produced.")

    return state


def reflect_node(state: AgentState) -> AgentState:
    """Evaluate results. Decide if acceptable or should retry."""
    logger.info("=== REFLECT: Evaluating results ===")

    best = state.get("best_result")
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", config.agent.max_iterations)
    min_sharpe = config.agent.min_acceptable_sharpe

    if best is None:
        state["should_continue"] = iteration < max_iter
        state["run_log"].append(f"Reflect: No results. {'Retrying...' if state['should_continue'] else 'Stopping.'}")
        return state

    sharpe = best.get("sharpe", 0.0)

    if sharpe >= min_sharpe:
        state["should_continue"] = False
        state["run_log"].append(
            f"Reflect: Sharpe {sharpe:.2f} >= threshold {min_sharpe:.2f}. ACCEPTING."
        )
        logger.info("Result accepted: Sharpe %.2f >= %.2f", sharpe, min_sharpe)
    elif iteration >= max_iter:
        state["should_continue"] = False
        state["run_log"].append(
            f"Reflect: Sharpe {sharpe:.2f} < {min_sharpe:.2f} but max iterations reached. Accepting best available."
        )
        logger.info("Max iterations reached. Accepting best: Sharpe %.2f", sharpe)
    else:
        state["should_continue"] = True
        state["run_log"].append(
            f"Reflect: Sharpe {sharpe:.2f} < {min_sharpe:.2f}. Retrying (iteration {iteration}/{max_iter})."
        )
        logger.info("Result below threshold. Will retry. (iteration %d/%d)", iteration, max_iter)

    return state


def store_node(state: AgentState) -> AgentState:
    """Persist best result to strategy memory."""
    logger.info("=== STORE: Persisting results ===")

    best = state.get("best_result")
    if best is None:
        state["run_log"].append("Store: Nothing to persist.")
        return state

    context = state.get("context")
    regime = context.regime_label if context else "Unknown"

    memory = StrategyMemory()
    result = PastResult(
        regime=regime,
        strategy_type=state.get("strategy_type", "momentum"),
        params=json.dumps(best["params"]),
        sharpe=best.get("sharpe", 0.0),
        total_return=best.get("total_return", 0.0),
        max_drawdown=best.get("max_drawdown", 0.0),
        confidence=best.get("confidence", 0.0),
        generation_method=best.get("generation_method", ""),
        reasoning=best.get("reasoning", ""),
    )
    run_id = memory.store(result)
    alpha = AlphaStore().store_backtest_result(
        regime=regime,
        strategy_type=state.get("strategy_type", "momentum"),
        params=best["params"],
        metrics={
            "sharpe_ratio": best.get("sharpe", 0.0),
            "total_return": best.get("total_return", 0.0),
            "max_drawdown": best.get("max_drawdown", 0.0),
            "num_trades": best.get("num_trades", 0),
        },
        assets=[state.get("asset", config.reference_asset)],
        generation_method=best.get("generation_method", ""),
        confidence=best.get("confidence", 0.0),
        reasoning=best.get("reasoning", ""),
        source="agent_graph",
    )
    nla = NLAMemoryStore().store_agent_summary(
        regime=regime,
        strategy_type=state.get("strategy_type", "momentum"),
        params=best["params"],
        metrics={
            "sharpe_ratio": best.get("sharpe", 0.0),
            "total_return": best.get("total_return", 0.0),
            "max_drawdown": best.get("max_drawdown", 0.0),
            "num_trades": best.get("num_trades", 0),
        },
        narrative=best.get("reasoning", "") or "Stored best proposal from explicit agent run.",
        alpha_id=alpha.alpha_id,
        tags=("agent_graph", best.get("generation_method", "")),
    )
    state["run_log"].append(
        f"Store: Persisted result {run_id}, alpha {alpha.alpha_id}, NLA note {nla.record_id}."
    )
    logger.info("Persisted result %s, alpha %s, NLA note %s.", run_id, alpha.alpha_id, nla.record_id)
    return state


def run_agent(
    ohlcv_data: Dict[str, pd.DataFrame],
    strategy_type: str = "momentum",
    asset: str = None,
    max_iterations: int = None,
) -> AgentState:
    """
    Run the full agent loop: analyze → hypothesize → backtest → reflect → (loop or store).

    This is a pure-Python implementation of the agent graph.
    If langgraph is available, it could be swapped for a StateGraph.
    """
    state: AgentState = {
        "ohlcv_data": ohlcv_data,
        "features_df": pd.DataFrame(),
        "context": None,
        "proposals": [],
        "results": [],
        "best_result": None,
        "iteration": 0,
        "max_iterations": max_iterations or config.agent.max_iterations,
        "strategy_type": strategy_type,
        "asset": asset or config.reference_asset,
        "should_continue": True,
        "memory_context": "",
        "run_log": [],
    }

    # Step 1: Analyze (once)
    state = analyze_node(state)

    # Step 2-4: Hypothesize → Backtest → Reflect (loop)
    while state["should_continue"] and state["iteration"] < state["max_iterations"]:
        state = hypothesize_node(state)
        state = backtest_node(state)
        state = reflect_node(state)

    # Step 5: Store
    state = store_node(state)

    return state
