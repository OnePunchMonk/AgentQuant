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
from src.agent.harness_config import EffectiveHarnessConfig
from src.agent.proposal_generator import Proposal, ProposalGenerator
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.agent.tools import get_default_registry
from src.agent.tools.orchestrator import ToolOrchestrator
from src.agent.trace import TraceRecorder, emit_trace
from src.research.alpha_store import AlphaStore, FailureRecord
from src.research.nla_memory import NLAMemoryStore
from src.utils.config import config

logger = logging.getLogger(__name__)

# Distinct, mutually exclusive run outcomes. "accepted" no longer conflates
# "cleared the quality bar" with "we ran out of budget and kept the best we
# had" -- those are different claims about how trustworthy the result is.
STATUS_PASSED_QUALITY_GATE = "passed_quality_gate"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_NO_VALID_CANDIDATE = "no_valid_candidate"
STATUS_EXECUTION_FAILED = "execution_failed"


class AgentState(TypedDict, total=False):
    """State flowing through the agent graph."""
    ohlcv_data: Dict[str, pd.DataFrame]
    full_ohlcv_data: Dict[str, pd.DataFrame]
    holdout_start: Optional[pd.Timestamp]
    features_df: pd.DataFrame
    context: Optional[RegimeContext]
    proposals: List[Proposal]
    results: List[Dict[str, Any]]
    all_results: List[Dict[str, Any]]
    best_result: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int
    strategy_type: str
    asset: str
    should_continue: bool
    memory_context: str
    run_log: List[str]
    trace: Optional[TraceRecorder]
    harness_config: Optional[EffectiveHarnessConfig]
    run_status: Optional[str]


def _split_search_and_holdout(
    ohlcv_data: Dict[str, pd.DataFrame],
    primary_asset: str,
    holdout_fraction: float,
) -> tuple:
    """
    Split OHLCV history into a search window (visible to the
    hypothesize/backtest/reflect retry loop) and a trailing holdout window
    the loop never sees.

    Returns (search_ohlcv_data, holdout_start). If the split can't be
    computed (bad fraction, missing/short data), returns the original data
    unchanged and holdout_start=None — the loop then behaves exactly as
    before (in-sample only), rather than failing the run.
    """
    if not (0.0 < holdout_fraction < 1.0):
        return ohlcv_data, None

    primary_df = ohlcv_data.get(primary_asset)
    if primary_df is None or primary_df.empty:
        return ohlcv_data, None

    idx = primary_df.index.sort_values()
    split_pos = int(len(idx) * (1 - holdout_fraction))
    if split_pos <= 0 or split_pos >= len(idx):
        return ohlcv_data, None

    # Too little history to carve out a meaningful search window (e.g. below
    # typical warmup requirements) -- fall back to in-sample-only behavior
    # rather than crippling the run.
    min_search_bars = 30
    if split_pos < min_search_bars:
        return ohlcv_data, None

    holdout_start = idx[split_pos]
    search_data = {
        ticker: df.loc[df.index < holdout_start]
        for ticker, df in ohlcv_data.items()
    }
    if search_data[primary_asset].empty:
        return ohlcv_data, None

    return search_data, holdout_start


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
    emit_trace(
        state.get("trace"),
        "analyze",
        f"Regime {regime_label} detected at {context.regime_confidence:.0%} confidence.",
        regime=regime_label,
        confidence=context.regime_confidence,
    )

    logger.info("Regime: %s, Confidence: %.0f%%", regime_label, context.regime_confidence * 100)
    return state


def _prompt_prefix_for(harness: Optional[EffectiveHarnessConfig]) -> str:
    """
    Render the harness's prompt_template/prompt_context into a text block
    prepended to the proposal-generation prompt, so changing
    harness.prompt_template provably changes the actually-submitted prompt
    (not just cosmetic metadata).
    """
    if harness is None or harness.prompt_template in ("", "default", "grid_search_default"):
        return ""
    lines = [f"[harness prompt profile: {harness.prompt_template}]"]
    for key, value in (harness.prompt_context or {}).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def hypothesize_node(state: AgentState) -> AgentState:
    """Generate strategy proposals via tool orchestrator with fallback to ProposalGenerator."""
    iteration = state.get("iteration", 0) + 1
    state["iteration"] = iteration
    logger.info("=== HYPOTHESIZE (iteration %d): Generating proposals ===", iteration)

    strategy_type = state.get("strategy_type", "momentum")
    context = state["context"]
    harness: Optional[EffectiveHarnessConfig] = state.get("harness_config")
    tools_enabled = harness.use_tools if harness is not None else True

    # Try tool-based orchestration first (uses Claude + web search if available),
    # but only when the resolved harness config actually enables tools. This is
    # the enforcement point for "disabling tools => zero external tool calls".
    proposals: List[Proposal] = []
    if tools_enabled:
        proposals = _hypothesize_with_tools(state, strategy_type, context, iteration, harness)
    else:
        logger.info("Tools disabled by harness config; skipping tool orchestration")

    # Fallback to traditional ProposalGenerator if tools fail or are disabled
    if not proposals:
        logger.info("Using ProposalGenerator (tools disabled or returned no proposals)")
        generator = ProposalGenerator()
        prompt_prefix = _prompt_prefix_for(harness)
        proposals = generator.generate(
            context=context,
            n_proposals=5,
            strategy_type=strategy_type,
            prior_results=state.get("all_results"),
            prompt_prefix=prompt_prefix,
        )

    state["proposals"] = proposals
    state["run_log"].append(
        f"Iteration {iteration}: Generated {len(proposals)} proposals "
        f"(methods: {[p.generation_method for p in proposals]})"
    )
    emit_trace(
        state.get("trace"),
        "hypothesize",
        f"Iteration {iteration}: generated {len(proposals)} candidate strategies.",
        iteration=iteration,
        methods=[p.generation_method for p in proposals],
        proposals=[p.params for p in proposals],
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
                    "strategy_type": strategy_type,
                    "params": proposal.params,
                    "sharpe": metrics.get("sharpe_ratio", 0.0),
                    "calmar": metrics.get("calmar", 0.0),
                    "sortino": metrics.get("sortino", 0.0),
                    "bootstrap_sharpe_p5": metrics.get("bootstrap_sharpe_p5", 0.0),
                    "total_return": metrics.get("total_return", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "num_trades": metrics.get("num_trades", 0),
                    "generation_method": proposal.generation_method,
                    "confidence": proposal.confidence,
                    "reasoning": proposal.reasoning,
                    "equity_curve": bt_result.get("equity_curve"),
                })
        except Exception as e:
            logger.warning("Backtest failed for proposal %d: %s", i, e)

    # Sort by Sharpe
    results.sort(key=lambda x: x.get("sharpe", 0.0), reverse=True)
    state["results"] = results
    state["all_results"] = state.get("all_results", []) + results

    if results:
        best = results[0]
        state["best_result"] = best
        state["run_log"].append(
            f"Best: Sharpe={best['sharpe']:.2f}, Return={best['total_return']:.1%}, "
            f"Params={best['params']}"
        )
        emit_trace(
            state.get("trace"),
            "backtest",
            (
                f"Best candidate Sharpe={best['sharpe']:.2f}, "
                f"Calmar={best.get('calmar', 0):.2f}, p5={best.get('bootstrap_sharpe_p5', 0):.2f}."
            ),
            best=best,
            results=results,
        )
        logger.info("Best result: Sharpe=%.2f, Return=%.1f%%, Params=%s",
                     best["sharpe"], best["total_return"] * 100, best["params"])
    else:
        state["best_result"] = None
        state["run_log"].append("No valid backtest results.")
        emit_trace(state.get("trace"), "backtest", "No valid candidate backtests completed.")
        logger.warning("No valid backtest results produced.")

    return state


def reflect_node(state: AgentState) -> AgentState:
    """Evaluate results. Decide if acceptable or should retry. Score falsifiable claims."""
    logger.info("=== REFLECT: Evaluating results ===")

    best = state.get("best_result")
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", config.agent.max_iterations)
    harness: Optional[EffectiveHarnessConfig] = state.get("harness_config")
    min_sharpe = harness.min_acceptable_sharpe if harness is not None else config.agent.min_acceptable_sharpe

    # Persist structured negative evidence so later iterations and runs can
    # avoid repeating the same regime/strategy mistake.
    alpha_store = AlphaStore()
    regime = state.get("context").regime_label if state.get("context") else "Unknown"
    for result in state.get("results", []):
        result_sharpe = float(result.get("sharpe", 0.0))
        if result_sharpe < min_sharpe:
            gap = result_sharpe - min_sharpe
            mode = "negative_sharpe" if result_sharpe < 0 else "below_sharpe_threshold"
            alpha_store.store_failure(FailureRecord(
                regime=regime, strategy_type=state.get("strategy_type", ""),
                params=result.get("params", {}), failure_mode=mode, metric_gap=gap,
                counterfactual_hypothesis=(
                    "Try shorter horizons or a different strategy family; this configuration "
                    "did not clear the out-of-sample Sharpe gate in this regime."
                ),
            ))

    if best is None:
        state["should_continue"] = iteration < max_iter
        if state["should_continue"]:
            state["run_log"].append("Reflect: No results. Retrying...")
        else:
            state["run_status"] = STATUS_NO_VALID_CANDIDATE
            state["run_log"].append(
                "Reflect: No valid candidate produced any backtest result. Stopping (no_valid_candidate)."
            )
        emit_trace(state.get("trace"), "reflect", state["run_log"][-1], accepted=False)
        return state

    sharpe = best.get("sharpe", 0.0)

    # Score falsifiable claims from proposals (for harness evaluation)
    _score_falsifiable_claims(state, best)

    if sharpe >= min_sharpe:
        state["should_continue"] = False
        state["run_status"] = STATUS_PASSED_QUALITY_GATE
        state["run_log"].append(
            f"Reflect: Sharpe {sharpe:.2f} >= threshold {min_sharpe:.2f}. ACCEPTING (passed_quality_gate)."
        )
        emit_trace(state.get("trace"), "reflect", state["run_log"][-1], accepted=True, status=state["run_status"])
        logger.info("Result accepted: Sharpe %.2f >= %.2f", sharpe, min_sharpe)
    elif iteration >= max_iter:
        state["should_continue"] = False
        state["run_status"] = STATUS_BUDGET_EXHAUSTED
        state["run_log"].append(
            f"Reflect: Sharpe {sharpe:.2f} < {min_sharpe:.2f} and max iterations reached. "
            f"Best-available result kept but NOT marked as passing quality gate (budget_exhausted)."
        )
        emit_trace(state.get("trace"), "reflect", state["run_log"][-1], accepted=False, status=state["run_status"])
        logger.info("Max iterations reached without clearing threshold. Best: Sharpe %.2f (budget_exhausted)", sharpe)
    else:
        state["should_continue"] = True
        state["run_log"].append(
            f"Reflect: Sharpe {sharpe:.2f} < {min_sharpe:.2f}. Retrying (iteration {iteration}/{max_iter})."
        )
        emit_trace(state.get("trace"), "reflect", state["run_log"][-1], accepted=False)
        logger.info("Result below threshold. Will retry. (iteration %d/%d)", iteration, max_iter)

    return state


def _score_falsifiable_claims(state: AgentState, best_result: Dict[str, Any]) -> None:
    """
    Log a provisional confidence/outcome diagnostic for proposal claims.

    Proposals currently store free-form claim text rather than a structured
    numerical Sharpe forecast, so this is not a prediction-accuracy score.
    """
    proposals = state.get("proposals", [])
    if not proposals:
        return

    realized_sharpe = best_result.get("sharpe", 0.0)
    for proposal in proposals:
        claim = proposal.reasoning
        confidence = proposal.confidence

        # Diagnostic only: structured forecasts are required before accuracy can
        # be computed against realized Sharpe.
        predicted_improvement = confidence > 0.5 and realized_sharpe > config.agent.min_acceptable_sharpe
        actual_improvement = realized_sharpe > config.agent.min_acceptable_sharpe

        claim_accurate = (predicted_improvement == actual_improvement)
        logger.debug(
            f"Claim scoring: {claim[:50]}... predicted={predicted_improvement}, actual={actual_improvement}, accurate={claim_accurate}"
        )


def holdout_eval_node(state: AgentState) -> AgentState:
    """
    Score the winning proposal exactly once against a trailing window the
    search loop never touched.

    The hypothesize/backtest/reflect loop is free to iterate against the
    search window as many times as it likes -- that's an in-sample search,
    not evidence. This node is the one point where the run is graded
    against data it hasn't seen, so it only runs once per agent run
    regardless of how many search iterations happened.
    """
    best = state.get("best_result")
    holdout_start = state.get("holdout_start")
    full_ohlcv = state.get("full_ohlcv_data")

    if best is None or holdout_start is None or not full_ohlcv:
        return state

    from src.backtest.runner import run_backtest

    asset = state.get("asset", config.reference_asset)
    strategy_type = state.get("strategy_type", "momentum")

    try:
        bt_result = run_backtest(
            full_ohlcv, [asset], strategy_type, best["params"], eval_start=holdout_start
        )
    except Exception as e:
        logger.warning("Holdout evaluation failed: %s", e)
        state["run_log"].append(f"Holdout: evaluation failed ({e}); reporting in-sample result only.")
        return state

    if not bt_result or "metrics" not in bt_result:
        state["run_log"].append("Holdout: no valid backtest result on held-out window.")
        return state

    holdout_sharpe = bt_result["metrics"].get("sharpe_ratio", 0.0)
    best["holdout_sharpe"] = holdout_sharpe
    best["holdout_total_return"] = bt_result["metrics"].get("total_return", 0.0)
    best["iterations_used"] = state.get("iteration", 1)

    gap = best.get("sharpe", 0.0) - holdout_sharpe
    state["run_log"].append(
        f"Holdout: Sharpe={holdout_sharpe:.2f} on unseen window "
        f"(in-sample was {best.get('sharpe', 0.0):.2f}, gap={gap:.2f})."
    )
    emit_trace(
        state.get("trace"),
        "holdout_eval",
        state["run_log"][-1],
        holdout_sharpe=holdout_sharpe,
        in_sample_sharpe=best.get("sharpe", 0.0),
    )
    logger.info(
        "Holdout Sharpe: %.2f (in-sample: %.2f, gap: %.2f)",
        holdout_sharpe, best.get("sharpe", 0.0), gap,
    )
    return state


def store_node(state: AgentState) -> AgentState:
    """Persist best result to strategy memory."""
    logger.info("=== STORE: Persisting results ===")

    best = state.get("best_result")
    if best is None:
        state["run_log"].append("Store: Nothing to persist.")
        emit_trace(state.get("trace"), "store", "No accepted result to persist.")
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
        holdout_sharpe=best.get("holdout_sharpe"),
        iterations_used=best.get("iterations_used", state.get("iteration", 0)),
    )
    run_id = memory.store(result)
    alpha = AlphaStore().store_backtest_result(
        regime=regime,
        strategy_type=state.get("strategy_type", "momentum"),
        params=best["params"],
        metrics={
            "sharpe_ratio": best.get("sharpe", 0.0),
            "calmar": best.get("calmar", 0.0),
            "sortino": best.get("sortino", 0.0),
            "bootstrap_sharpe_p5": best.get("bootstrap_sharpe_p5", 0.0),
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
            "calmar": best.get("calmar", 0.0),
            "sortino": best.get("sortino", 0.0),
            "bootstrap_sharpe_p5": best.get("bootstrap_sharpe_p5", 0.0),
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
    emit_trace(state.get("trace"), "store", state["run_log"][-1], run_id=run_id)
    logger.info("Persisted result %s, alpha %s, NLA note %s.", run_id, alpha.alpha_id, nla.record_id)
    return state


def _hypothesize_with_tools(
    state: AgentState,
    strategy_type: str,
    context: RegimeContext,
    iteration: int,
    harness: Optional[EffectiveHarnessConfig] = None,
) -> List[Proposal]:
    """
    Generate proposals using tool orchestrator with Claude tool-use.

    This is a tool-calling approach that:
    1. Gathers market context via tools (regime context, optional web search)
    2. Calls Claude with tool schemas to generate proposals
    3. Returns proposals with falsifiable claims

    Falls back gracefully if tools/Claude unavailable.

    Args:
        state: Current agent state
        strategy_type: Strategy type to generate proposals for
        context: Market regime context
        iteration: Current iteration number

    Returns:
        List of Proposal objects (empty if orchestration fails)
    """
    try:
        registry = get_default_registry()
        if harness is not None and not harness.use_web_search:
            registry = registry.without_tool("search_market_sentiment")
        orchestrator = ToolOrchestrator(registry)

        prompt_prefix = _prompt_prefix_for(harness)
        result = orchestrator.run_tool_loop(
            user_prompt=f"""
{prompt_prefix}Generate 5 high-confidence {strategy_type} strategy proposals for the current market regime.

For each proposal:
1. Identify which regime characteristic (volatility, trend, mean reversion) you're targeting
2. Explain the parameter choice briefly
3. Assign a confidence score (0.0-1.0)
4. Make a falsifiable claim about expected Sharpe improvement vs. recent baseline

Return proposals with: params (dict), confidence (float), reasoning (string), falsifiable_claim (string)
""",
            regime_context=context,
            strategy_type=strategy_type,
            max_turns=2,
        )

        logger.info(f"Tool orchestrator completed with {len(result.get('tool_calls', []))} tool calls")

        # Record tool calls in trace
        if state.get("trace"):
            for tool_call in result.get("tool_calls", []):
                emit_trace(
                    state["trace"],
                    "tool_call",
                    f"Called {tool_call['tool_name']}",
                    tool_name=tool_call["tool_name"],
                    result_success=tool_call.get("result", {}).get("success", False),
                )

        # Parse proposals from Claude response and convert to Proposal objects
        parsed_proposals = result.get("proposals", [])
        if parsed_proposals:
            logger.info(f"Tool orchestration returned {len(parsed_proposals)} parsed proposals")
            return parsed_proposals
        else:
            logger.info("Tool orchestration returned no proposals; will use fallback")
            return []

    except ImportError as e:
        logger.warning(f"Tool orchestration unavailable (missing dependency: {e}); will use fallback")
        return []
    except Exception as e:
        logger.warning(f"Tool orchestration failed: {e}; falling back to ProposalGenerator")
        return []


def run_agent(
    ohlcv_data: Dict[str, pd.DataFrame],
    strategy_type: str = "momentum",
    asset: str = None,
    max_iterations: int = None,
    trace: Optional[TraceRecorder] = None,
    harness_config: Optional[Any] = None,
) -> AgentState:
    """
    Run the full agent loop: analyze → hypothesize → backtest → reflect → (loop or store).

    This is a pure-Python implementation of the agent graph.
    If langgraph is available, it could be swapped for a StateGraph.

    Args:
        harness_config: An EffectiveHarnessConfig (already resolved via
            src.agent.harness_config.resolve_effective_config), or a raw
            HarnessConfig which will be resolved here. Governs tool
            admission, prompt content, and the quality-gate threshold for
            this run. If omitted, the loop behaves exactly as it did before
            harness config threading existed (global config.agent.*).
    """
    from src.agent.harness_config import HarnessConfig, resolve_effective_config

    resolved_asset = asset or config.reference_asset
    effective_max_iterations = max_iterations or config.agent.max_iterations

    effective_harness: Optional[EffectiveHarnessConfig] = None
    if isinstance(harness_config, HarnessConfig):
        effective_harness = resolve_effective_config(
            harness_config, max_iterations_override=effective_max_iterations
        )
    elif harness_config is not None:
        effective_harness = harness_config

    search_ohlcv, holdout_start = _split_search_and_holdout(
        ohlcv_data, resolved_asset, config.agent.holdout_fraction
    )

    state: AgentState = {
        "ohlcv_data": search_ohlcv,
        "full_ohlcv_data": ohlcv_data,
        "holdout_start": holdout_start,
        "features_df": pd.DataFrame(),
        "context": None,
        "proposals": [],
        "results": [],
        "best_result": None,
        "iteration": 0,
        "max_iterations": effective_max_iterations,
        "strategy_type": strategy_type,
        "asset": resolved_asset,
        "should_continue": True,
        "memory_context": "",
        "run_log": [],
        "trace": trace,
        "harness_config": effective_harness,
        "run_status": None,
    }
    if holdout_start is not None:
        state["run_log"].append(
            f"Holdout: reserving data from {holdout_start.date()} onward; "
            f"search loop only sees data before it."
        )
    if effective_harness is not None:
        state["run_log"].append(
            f"Harness config resolved (hash={effective_harness.config_hash}): "
            f"use_tools={effective_harness.use_tools}, "
            f"prompt_template={effective_harness.prompt_template!r}, "
            f"min_acceptable_sharpe={effective_harness.min_acceptable_sharpe}."
        )
        emit_trace(
            trace, "harness_config",
            state["run_log"][-1],
            requested=effective_harness.requested,
            effective=effective_harness.to_dict(),
        )

    try:
        # Step 1: Analyze (once, on the search window only)
        state = analyze_node(state)

        # Step 2-4: Hypothesize → Backtest → Reflect (loop, in-sample search)
        while state["should_continue"] and state["iteration"] < state["max_iterations"]:
            state = hypothesize_node(state)
            state = backtest_node(state)
            state = reflect_node(state)

        # Step 5: Grade the winner once against data the loop never saw
        state = holdout_eval_node(state)

        # Step 6: Store
        state = store_node(state)
    except Exception as exc:
        state["run_status"] = STATUS_EXECUTION_FAILED
        state["run_log"].append(f"Execution failed: {exc!r}")
        emit_trace(state.get("trace"), "execution_failed", str(exc))
        logger.exception("Agent run failed with an unhandled exception")
        raise
    finally:
        # The loop may exit (e.g. max_iterations=0) without ever reaching
        # reflect_node, which is otherwise the only place run_status is set.
        if state.get("run_status") is None:
            state["run_status"] = (
                STATUS_PASSED_QUALITY_GATE if state.get("best_result") is not None
                else STATUS_NO_VALID_CANDIDATE
            )
        try:
            from src.agent.run_manifest import build_manifest_from_state

            manifest = build_manifest_from_state(state)
            state["run_manifest"] = manifest.to_dict()
            manifest.save()
        except Exception:
            logger.warning("Failed to build/persist run manifest", exc_info=True)

    return state
