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
import os
import uuid
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from src.agent.context_builder import RegimeContext, build_context
from src.agent.harness_config import EffectiveHarnessConfig
from src.agent.proposal_generator import Proposal, ProposalGenerator
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.agent.tools import get_default_registry
from src.agent.tools.orchestrator import ToolOrchestrator
from src.agent.trace import TraceRecorder, emit_trace
from src.memory import MemoryQuery, MemoryService, Note, Trial, regime_vec_from_context
from src.memory.canonical import to_date
from src.research.alpha_store import AlphaStore
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
    run_id: str
    as_of: Optional[str]
    memory_pack: Optional[Any]
    memory_snapshot_id: Optional[str]
    attempt_history_policy: str
    attempt_history_char_budget: int
    space_exhausted: bool


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

    state["features_df"] = features_df
    state["context"] = context
    state["run_log"] = state.get("run_log", [])
    _recall_memory(state, context)
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


def _min_sharpe(state: AgentState) -> float:
    harness: Optional[EffectiveHarnessConfig] = state.get("harness_config")
    return harness.min_acceptable_sharpe if harness is not None else config.agent.min_acceptable_sharpe


def _data_window(ohlcv: Dict[str, pd.DataFrame], asset: str) -> tuple:
    df = (ohlcv or {}).get(asset)
    if df is None or df.empty:
        return None, None, 0
    return to_date(df.index.min()), to_date(df.index.max()), len(df)


def _recall_memory(state: AgentState, context: RegimeContext) -> None:
    """One unified memory read per run, cut off at the last bar the search loop sees."""
    service = MemoryService()
    if not service.can_read:
        context.memory_context = ""
        state["memory_pack"] = None
        state["memory_context"] = ""
        return
    query = MemoryQuery(
        as_of=state.get("as_of"),
        strategy_types=(state.get("strategy_type", "momentum"),),
        asset=state.get("asset"),
        regime_label=context.regime_label,
        regime_vec=regime_vec_from_context(context),
        exclude_run_id=state.get("run_id"),
        k_beliefs=config.memory.k_beliefs,
        token_budget=config.memory.token_budget,
        min_acceptable_sharpe=_min_sharpe(state),
    )
    pack = service.recall(query, run_id=state.get("run_id"), iteration=state.get("iteration", 0))
    context.memory_context = pack.to_prompt()
    state["memory_pack"] = pack
    state["memory_snapshot_id"] = pack.snapshot_id
    state["memory_context"] = context.memory_context
    state["run_log"].append(
        f"Memory: {pack.n_visible_trials} visible trials as of {query.as_of}; "
        f"{len(pack.beliefs)} beliefs, {len(pack.avoid)} to avoid, {len(pack.seeds)} seeds "
        f"(snapshot {pack.snapshot_id})."
    )


def _classify_outcome(result: Dict[str, Any], min_sharpe: float) -> tuple:
    sharpe = float(result.get("sharpe", 0.0) or 0.0)
    drawdown = abs(float(result.get("max_drawdown", 0.0) or 0.0))
    if sharpe < 0:
        return "rejected", "negative_sharpe"
    if sharpe < min_sharpe:
        return "watch", "below_threshold"
    if drawdown > config.agent.risk.max_drawdown:
        return "watch", "drawdown"
    return "accepted", None


def _record_trials(state: AgentState, results: List[Dict[str, Any]], errors: List[Dict[str, Any]]) -> None:
    """Persist every backtested proposal, winners and losers, as episodic memory."""
    service = MemoryService()
    if not service.can_write:
        return
    asset = state.get("asset", config.reference_asset)
    strategy_type = state.get("strategy_type", "momentum")
    data_start, data_end, n_days = _data_window(state.get("ohlcv_data"), asset)
    context = state.get("context")
    common = dict(
        run_id=state.get("run_id", ""),
        iteration=state.get("iteration", 0),
        asset=asset,
        strategy_type=strategy_type,
        data_start=data_start,
        data_end=data_end,
        regime_label=context.regime_label if context else "Unknown",
        regime_vec=regime_vec_from_context(context),
        source="agent_graph",
    )
    min_sharpe = _min_sharpe(state)
    for result in results:
        outcome, failure_mode = _classify_outcome(result, min_sharpe)
        trial = Trial(
            params=dict(result.get("params") or {}),
            generation_method=result.get("generation_method", ""),
            is_sharpe=result.get("sharpe"),
            is_return=result.get("total_return"),
            is_max_dd=result.get("max_drawdown"),
            is_trades=result.get("num_trades"),
            is_sortino=result.get("sortino"),
            is_calmar=result.get("calmar"),
            is_boot_p5=result.get("bootstrap_sharpe_p5"),
            is_n_days=n_days,
            outcome=outcome,
            failure_mode=failure_mode,
            reasoning=result.get("reasoning", "") or "",
            **common,
        )
        result["trial_id"] = service.record_trial(trial)
    for error in errors:
        service.record_trial(Trial(
            params=dict(error.get("params") or {}),
            generation_method=error.get("generation_method", ""),
            outcome="error",
            failure_mode="error",
            reasoning=error.get("error", ""),
            **common,
        ))


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


def _prior_attempts_for_prompt(state: AgentState) -> List[Dict[str, Any]]:
    attempts = state.get("all_results") or []
    if state.get("attempt_history_policy") != "folding":
        return attempts
    from src.agent.attempt_history import fold_attempt_history

    return fold_attempt_history(
        attempts,
        char_budget=state.get("attempt_history_char_budget", 6000),
    )


def _tried_param_keys(state: AgentState) -> set:
    """Parameter keys already backtested earlier in this run."""
    return {
        tuple(sorted((r.get("params") or {}).items()))
        for r in state.get("all_results", [])
        if r.get("params")
    }


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
    tried = _tried_param_keys(state)
    proposals: List[Proposal] = []
    if tools_enabled:
        proposals = _hypothesize_with_tools(state, strategy_type, context, iteration, harness)
        # A retry must test something new; drop configs already evaluated this run.
        proposals = [p for p in proposals if tuple(sorted(p.params.items())) not in tried]
    else:
        logger.info("Tools disabled by harness config; skipping tool orchestration")

    # Fallback to traditional ProposalGenerator if tools fail or are disabled
    if not proposals:
        logger.info("Using ProposalGenerator (tools disabled or returned no proposals)")
        generator = ProposalGenerator()
        prompt_prefix = _prompt_prefix_for(harness)
        prior_results = _prior_attempts_for_prompt(state)
        proposals = generator.generate(
            context=context,
            n_proposals=5,
            strategy_type=strategy_type,
            prior_results=prior_results,
            prompt_prefix=prompt_prefix,
            memory_pack=state.get("memory_pack"),
            exclude_params=tried,
        )

    if not proposals and tried:
        # Every config in the canonical grid has been evaluated; another
        # iteration cannot add information.
        state["space_exhausted"] = True
        state["iteration"] = iteration - 1  # no backtest ran this round
        state["proposals"] = []
        state["should_continue"] = False
        state["run_status"] = (
            STATUS_BUDGET_EXHAUSTED if state.get("best_result") is not None
            else STATUS_NO_VALID_CANDIDATE
        )
        state["run_log"].append(
            f"Iteration {iteration}: no new candidates to propose ({len(tried)} configs "
            f"already evaluated; the rest are excluded or already tried); stopping early."
        )
        emit_trace(state.get("trace"), "hypothesize", state["run_log"][-1],
                   iteration=iteration, methods=[], proposals=[])
        return state

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


def _sidecar_policy(seed_text: str):
    """Policy for the shadow advisory.

    Deterministic by default (no behaviour change). Setting AGENTQUANT_IDEA_SIDECAR_EPSILON
    to a value in (0, 1] logs with exploration, so the recorded propensities can be used
    to value other policies off-policy. The advisory is shadow-only: it is shown to the
    proposer but never drives the backtest, so exploring costs no evaluation budget.
    """
    import hashlib
    import os

    from src.agent import idea_sidecar

    raw = os.getenv("AGENTQUANT_IDEA_SIDECAR_EPSILON", "").strip()
    if not raw:
        return None
    try:
        epsilon = float(raw)
    except ValueError:
        return None
    if not 0.0 < epsilon <= 1.0:
        return None
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "big")
    return idea_sidecar.EpsilonGreedyPolicy(epsilon=epsilon, seed=seed)


def _common_eval_start(ohlcv: Dict[str, pd.DataFrame], asset: str, strategy_type: str):
    """First bar every grid config has a defined signal for, or None.

    A slow strategy is flat (earning exactly zero) until its longest window
    fills. Scoring over the whole window then lowers its Sharpe by roughly the
    square root of the active fraction, so the tournament favoured short
    windows for a reason unrelated to quality. Scoring all candidates from one
    shared start removes that. When the window is too short to afford the
    warmup (more than 40% of it), return None and keep whole-window scoring.
    """
    from src.agent.parameter_grid import ParameterGrid
    from src.backtest.core import common_warmup_bars

    df = (ohlcv or {}).get(asset)
    if df is None or df.empty:
        return None
    bars = common_warmup_bars(ParameterGrid().get_grid(strategy_type), n_bars=len(df))
    if bars <= 0 or bars >= len(df):
        return None
    return df.index[bars]


def backtest_node(state: AgentState) -> AgentState:
    """Run backtests on all proposals and rank by Sharpe."""
    logger.info("=== BACKTEST: Running tournament ===")
    from src.backtest.runner import run_backtest

    ohlcv = state["ohlcv_data"]
    asset = state.get("asset", config.reference_asset)
    strategy_type = state.get("strategy_type", "momentum")

    results = []
    errors = []
    eval_start = _common_eval_start(ohlcv, asset, strategy_type)
    if eval_start is None:
        state["run_log"].append(
            "Backtest: search window too short for common-window scoring; "
            "candidates are scored over the whole window."
        )
    for i, proposal in enumerate(state["proposals"]):
        try:
            bt_result = run_backtest(
                ohlcv, [asset], strategy_type, proposal.params, eval_start=eval_start
            )
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
            errors.append({"params": proposal.params, "generation_method": proposal.generation_method,
                           "error": repr(e)})

    try:
        _record_trials(state, results, errors)
    except Exception:
        logger.warning("Failed to record trials in memory", exc_info=True)

    # Sort by Sharpe
    results.sort(key=lambda x: x.get("sharpe", 0.0), reverse=True)
    state["results"] = results
    state["all_results"] = state.get("all_results", []) + results

    if results:
        best = results[0]
        previous_best = state.get("best_result")
        if previous_best is not None and previous_best.get("sharpe", float("-inf")) > best.get("sharpe", float("-inf")):
            # Keep the best candidate seen across all iterations. Previously this
            # was overwritten each round, so a worse later iteration replaced the
            # winner that holdout grading and storage then used.
            best = previous_best
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
        # Keep any earlier best; only clear it if nothing has ever succeeded.
        if not state.get("all_results"):
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

    # Negative evidence is persisted by backtest_node: every trial row carries
    # its outcome/failure_mode, so repeated failures of one config collapse
    # into one belief instead of one failure row per result per iteration.

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

    try:
        _, full_end, _ = _data_window(full_ohlcv, asset)
        MemoryService().attach_oos(
            best.get("trial_id", ""),
            oos_sharpe=holdout_sharpe,
            oos_return=best["holdout_total_return"],
            oos_max_dd=bt_result["metrics"].get("max_drawdown"),
            oos_start=to_date(holdout_start),
            oos_end=full_end,
            source="holdout",
            min_acceptable_sharpe=_min_sharpe(state),
        )
    except Exception:
        logger.warning("Failed to attach holdout evidence to memory", exc_info=True)

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
    narrative = best.get("reasoning", "")
    # Grid/random fallbacks carry boilerplate, not a thesis worth recalling.
    if narrative and best.get("generation_method") not in ("grid_search", "random"):
        try:
            _, data_end, _ = _data_window(state.get("ohlcv_data"), state.get("asset", config.reference_asset))
            if best.get("holdout_sharpe") is not None:
                _, data_end, _ = _data_window(state.get("full_ohlcv_data"), state.get("asset", config.reference_asset))
            trial = Trial(asset=state.get("asset", config.reference_asset),
                          strategy_type=state.get("strategy_type", "momentum"), params=best["params"])
            MemoryService().record_note(Note(
                kind="nla", body=narrative, data_end=data_end,
                strategy_type=trial.strategy_type, asset=trial.asset, regime_label=regime,
                config_key=trial.config_key,
                meta={"params": best["params"], "trial_id": best.get("trial_id"),
                      "holdout_sharpe": best.get("holdout_sharpe")},
                quality=float(best.get("holdout_sharpe", best.get("sharpe", 0.0)) or 0.0)
                - abs(float(best.get("max_drawdown", 0.0) or 0.0)),
                source="agent_graph",
            ))
        except Exception:
            logger.warning("Failed to record run narrative in memory", exc_info=True)
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
        # The deterministic assistant runner is a separate, read-only path.
        # Keep it opt-in so existing harnesses retain their exact behavior.
        assistant_context = _run_assistant_sidecar_context(state, strategy_type, context, iteration)
        registry = get_default_registry()
        if harness is not None and not harness.use_web_search:
            registry = registry.without_tool("search_market_sentiment")
        orchestrator = ToolOrchestrator(registry)

        prompt_prefix = _prompt_prefix_for(harness)
        from src.agent.proposal_generator import ProposalGenerator
        prior_attempts = ProposalGenerator._format_prior_results(_prior_attempts_for_prompt(state))
        result = orchestrator.run_tool_loop(
            user_prompt=f"""
{assistant_context}
{prompt_prefix}{prior_attempts}Generate 5 high-confidence {strategy_type} strategy proposals for the current market regime.

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
            if assistant_context:
                source_names = []
                for run in state.get("assistant_tool_results", []):
                    source_names.extend(run.get("selected_tools", []))
                    if run.get("tool_name"):
                        source_names.append(run["tool_name"])
                sources = sorted(set(source_names))
                provenance_note = "Advisory context sources: " + ", ".join(sources) + "."
                state["assistant_proposal_provenance"] = {
                    "sources": sources, "decision_time": str(state.get("as_of")),
                    "validated": False, "promotion_authority": False,
                }
                for proposal in parsed_proposals:
                    proposal.reasoning = (proposal.reasoning + " " + provenance_note).strip()
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


def _run_assistant_sidecar_context(
    state: AgentState, strategy_type: str, context: RegimeContext, iteration: int,
) -> str:
    """Optionally fetch bounded temporal-memory evidence for proposal context.

    This tool path has no holdout or promotion capability. It is explicitly
    gated by AGENTQUANT_ASSISTANT_TOOLS and defaults off for compatibility.
    """
    if os.getenv("AGENTQUANT_ASSISTANT_TOOLS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return ""
    as_of = state.get("as_of")
    if not as_of:
        return ""
    try:
        from src.agent.assistant_tool_runner import AssistantToolRunner, AssistantToolRunRequest
        from src.agent.memory_tool import MemorySearchTool
        from src.agent.sidecar_tools import (
            SidecarToolRegistry,
            ToolBudget,
            make_idea_sidecar_registry,
            register_hyde_retrieval,
            register_memory_search,
        )
        from src.memory import MemoryService

        registry = SidecarToolRegistry(ToolBudget(max_calls=1, max_wall_seconds=2.0,
                                                  max_input_bytes=4_000, max_output_bytes=16_000))
        register_memory_search(registry, MemorySearchTool(MemoryService()))
        # The deterministic generator is local and offline. Provider-backed
        # HyDE is intentionally never constructed by this production path.
        from src.agent.generators.hyde import HydeGenerator
        from src.agent.hyde_retrieval import HydeRetrievalTool
        from src.agent.idea_sidecar import Candidate as IdeaCandidate
        from src.agent.idea_sidecar import Observation
        from src.agent.parameter_grid import ParameterGrid
        from src.agent.rl_shadow_tool import advise

        grid = ParameterGrid().get_grid(strategy_type)
        # What this run has already evaluated, as grid indices, so candidate options
        # exclude it and the policy observation shows it (it used to be empty).
        tried_keys = _tried_param_keys(state)
        tried_indices = [i for i, g in enumerate(grid) if tuple(sorted(g.items())) in tried_keys]
        rounds_left = max(0, int(state.get("max_iterations", 1)) - int(iteration) + 1)
        # Memory lookup is an independent request with an explicit point-in-time cutoff.
        request_id = f"{state.get('run_id', 'run')}:{iteration}:memory"
        memory_request = AssistantToolRunRequest(
            request_id=request_id, intent="memory_lookup", decision_time=str(as_of),
            arguments={"query": f"{strategy_type} {context.regime_label}",
                       "strategy_type": strategy_type, "asset": state.get("asset"),
                       "regime_label": context.regime_label, "top_k": 4},
            capabilities=frozenset({"prior_results"}), max_tool_calls=1,
        )
        runner = AssistantToolRunner(registry)
        memory_run = runner.run(memory_request)
        # Expose only development-visible failures and prior parameter points.
        prior_params = [r.get("params", {}) for r in state.get("all_results", []) if r.get("params")]
        idea_registry = make_idea_sidecar_registry(budget=ToolBudget(
            max_calls=2, max_wall_seconds=2.0, max_input_bytes=12_000, max_output_bytes=16_000))
        idea_request = AssistantToolRunRequest(
            request_id=f"{state.get('run_id', 'run')}:{iteration}:ideas",
            intent="candidate_options", decision_time=str(as_of),
            arguments={"grid": grid, "tried_grid_indices": tried_indices, "failures": [],
                       "prior_params": prior_params},
            capabilities=frozenset({"prior_results"}), max_tool_calls=1,
        )
        idea_run = AssistantToolRunner(idea_registry).run(idea_request)
        # Run the fixed, non-learning shadow policy only over the validated
        # canonical candidates returned above. It cannot select outside this menu.
        if idea_run.results and idea_run.results[0].ok and idea_run.results[0].data:
            from src.agent import idea_sidecar
            raw_candidates = idea_run.results[0].data[:32]
            candidates = [IdeaCandidate(idea_sidecar.ProposalSource(item["source"]),
                                        int(item["grid_index"]), dict(item["params"]))
                          for item in raw_candidates]
            # Candidate serialization contract is validated in the adapter; RL
            # policy output is advisory metadata and never enters backtest inputs.
            observation = Observation(
                schema_version=idea_sidecar.SCHEMA_VERSION,
                episode_id=f"{state.get('run_id', 'run')}:{iteration}",
                decision_time=str(as_of), step=iteration, asset=state.get("asset", ""),
                regime=context.regime_label, strategy_family=strategy_type,
                grid_size=len(grid), tried_grid_indices=tuple(tried_indices), visible_failures=(),
                attempted=(), remaining_proposals=5 * rounds_left, remaining_tool_calls=0,
            )
            # What the proposer sees is always the deterministic advisory. An exploration policy,
            # when switched on, is only logged: if its sampled action reached the prompt,
            # changing epsilon would change which configs get evaluated.
            rl_advisory = advise(observation, candidates).to_dict()
            exploration = _sidecar_policy(seed_text=f"{state.get('run_id', 'run')}")
            rl_logged = rl_advisory if exploration is None else advise(observation, candidates, exploration).to_dict()
            rl_log_record = dict(
                rl_logged,
                observation=observation.to_dict(),
                candidates=[c.to_dict() for c in candidates],
            )
        else:
            rl_advisory = None
            rl_log_record = None
        hyde_run = None
        try:
            retriever = HydeRetrievalTool(HydeGenerator(), evidence=[])
            hyde_registry = SidecarToolRegistry(ToolBudget(max_calls=1, max_wall_seconds=2.0,
                                                            max_input_bytes=2_000, max_output_bytes=8_000))
            register_hyde_retrieval(hyde_registry, retriever)
            hyde_run = AssistantToolRunner(hyde_registry).run(AssistantToolRunRequest(
                request_id=f"{state.get('run_id', 'run')}:{iteration}:hyde",
                intent="hypothesis_retrieval", decision_time=str(as_of),
                arguments={"regime": context.regime_label,
                           "strategy_types": [strategy_type], "max_hypotheses": 2,
                           "max_evidence": 4}, capabilities=frozenset({"prior_results"}),
                max_tool_calls=1))
        except Exception:
            logger.debug("Offline HyDE sidecar unavailable", exc_info=True)
        payloads = [memory_run.to_dict(), idea_run.to_dict()]
        if hyde_run is not None:
            payloads.append(hyde_run.to_dict())
        if rl_advisory is not None:
            payloads.append({"tool_name": "rl_shadow_advisory", "ok": True,
                             "data": rl_log_record,
                             "provenance": {"source": "src.agent.rl_shadow_tool.advise",
                                            "validated_canonical_menu": True}})
        state.setdefault("assistant_tool_results", []).extend(payloads)
        for item in payloads:
            emit_trace(state.get("trace"), "assistant_tool_run",
                       f"Assistant sidecar {item.get('intent', item.get('tool_name'))}: {item.get('status', 'ok' if item.get('ok') else 'error')}.",
                       result=item)
        # Include only bounded output from successful advisory calls.
        snippets = []
        for run in (memory_run, idea_run, hyde_run):
            if run and run.status in {"ok", "partial"}:
                snippets.extend(r.to_dict() for r in run.results if r.ok)
        if rl_advisory is not None:
            snippets.append({"tool_name": "rl_shadow_advisory", "data": rl_advisory,
                             "provenance": {"source": "src.agent.rl_shadow_tool.advise",
                                            "validated_canonical_menu": True}})
        return ("Bounded read-only sidecar context (advisory only; unevaluated; cannot promote):\n"
                + json.dumps(snippets, default=str)[:12_000] if snippets else "")
    except Exception as exc:
        logger.info("Assistant sidecar lookup unavailable: %s", exc)
        return ""


def run_agent(
    ohlcv_data: Dict[str, pd.DataFrame],
    strategy_type: str = "momentum",
    asset: str = None,
    max_iterations: int = None,
    trace: Optional[TraceRecorder] = None,
    harness_config: Optional[Any] = None,
    attempt_history_policy: str = "recent",
    attempt_history_char_budget: int = 6000,
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

    if attempt_history_policy not in ("recent", "folding"):
        raise ValueError("attempt_history_policy must be 'recent' or 'folding'")
    if attempt_history_char_budget < 0:
        raise ValueError("attempt_history_char_budget must be non-negative")

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
    # Memory is cut off at the last bar the search loop can see, so neither
    # future data nor this run's own holdout window leaks in via memory.
    _, as_of, _ = _data_window(search_ohlcv, resolved_asset)

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
        "run_id": uuid.uuid4().hex[:12],
        "as_of": as_of,
        "memory_pack": None,
        "memory_snapshot_id": None,
        "attempt_history_policy": attempt_history_policy,
        "attempt_history_char_budget": attempt_history_char_budget,
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
            if state.get("space_exhausted"):
                break
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

            manifest = build_manifest_from_state(
                state, memory_snapshot_id=state.get("memory_snapshot_id")
            )
            state["run_manifest"] = manifest.to_dict()
            manifest.save()
        except Exception:
            logger.warning("Failed to build/persist run manifest", exc_info=True)

    return state
