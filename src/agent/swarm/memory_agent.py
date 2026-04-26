"""
Memory Agent — Pattern-Aware Cross-Session Learning
====================================================

Upgrades StrategyMemory from raw SQL dump to pattern extraction.
Identifies what has worked (and failed) in similar regimes,
and generates actionable natural-language patterns for specialist prompts.
"""

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.agent.strategy_memory import PastResult, StrategyMemory
from src.agent.swarm.state import SwarmState

logger = logging.getLogger(__name__)

PATTERN_MIN_SAMPLES = 2   # Minimum samples to report a pattern
GOOD_SHARPE = 0.5         # Threshold for "good" performance
BAD_SHARPE = 0.0          # Threshold for "bad" performance


class MemoryAgent:
    """
    Pattern-aware memory agent.
    Reads StrategyMemory and extracts actionable patterns,
    not just a raw SQL dump.
    """

    def __init__(self, memory: Optional[StrategyMemory] = None):
        self.memory = memory or StrategyMemory()

    def retrieve_patterns(self, regime_label: str, strategy_types: List[str]) -> List[str]:
        """
        Query memory and extract natural-language patterns.
        Returns list of pattern strings for injection into specialist prompts.
        """
        patterns: List[str] = []

        # 1. Per-strategy performance patterns
        for st in strategy_types:
            try:
                history = self.memory.query_regime(regime_label, st, limit=100)
                if not history:
                    patterns.append(
                        f"📝 {st.upper()}: No history in {regime_label} regime — "
                        "unexplored territory, consider diversifying."
                    )
                    continue

                patterns.extend(self._analyse_strategy_history(st, regime_label, history))

            except Exception as e:
                logger.debug("Memory pattern extraction failed for %s: %s", st, e)

        # 2. Cross-regime insights
        try:
            cross_patterns = self._cross_regime_patterns(strategy_types)
            patterns.extend(cross_patterns)
        except Exception:
            pass

        return patterns

    def _analyse_strategy_history(
        self, strategy_type: str, regime_label: str, history: List[PastResult]
    ) -> List[str]:
        """Extract patterns from historical runs for one strategy type in one regime."""
        patterns = []
        sharpes = [h.sharpe for h in history]
        mean_sharpe = sum(sharpes) / len(sharpes)
        good = [h for h in history if h.sharpe >= GOOD_SHARPE]
        bad = [h for h in history if h.sharpe <= BAD_SHARPE]

        # Overall performance statement
        patterns.append(
            f"📊 {strategy_type.upper()} in {regime_label}: "
            f"n={len(history)}, mean_sharpe={mean_sharpe:.2f}, "
            f"good_runs={len(good)}, bad_runs={len(bad)}"
        )

        # Parameter-level insights for momentum (most param-sensitive)
        if strategy_type == "momentum" and len(history) >= PATTERN_MIN_SAMPLES:
            slow_by_perf = defaultdict(list)
            for h in history:
                try:
                    params = json.loads(h.params)
                    slow = params.get("slow_window", 0)
                    bucket = slow // 30 * 30  # bucket into 30-day ranges
                    slow_by_perf[bucket].append(h.sharpe)
                except Exception:
                    pass

            for bucket, bucket_sharpes in sorted(slow_by_perf.items()):
                if len(bucket_sharpes) >= PATTERN_MIN_SAMPLES:
                    avg = sum(bucket_sharpes) / len(bucket_sharpes)
                    verdict = "✅ GOOD" if avg >= GOOD_SHARPE else ("❌ POOR" if avg <= BAD_SHARPE else "🔶 MIXED")
                    patterns.append(
                        f"  {verdict} slow_window ~{bucket}-{bucket+30}: "
                        f"avg_sharpe={avg:.2f} (n={len(bucket_sharpes)})"
                    )

        # Best and worst specific runs
        if good:
            best = max(good, key=lambda h: h.sharpe)
            try:
                best_params = json.loads(best.params)
                patterns.append(
                    f"  🏆 Best: {best_params} → Sharpe {best.sharpe:.2f}"
                )
            except Exception:
                pass

        if bad:
            worst = min(bad, key=lambda h: h.sharpe)
            try:
                worst_params = json.loads(worst.params)
                patterns.append(
                    f"  ⚠️  Worst: {worst_params} → Sharpe {worst.sharpe:.2f} — AVOID"
                )
            except Exception:
                pass

        return patterns

    def _cross_regime_patterns(self, strategy_types: List[str]) -> List[str]:
        """Identify cross-regime insights (e.g., what never works anywhere)."""
        patterns = []
        for st in strategy_types:
            try:
                all_history = self.memory.query_all(st, limit=200)
                if len(all_history) < 5:
                    continue
                avg = sum(h.sharpe for h in all_history) / len(all_history)
                if avg < 0:
                    patterns.append(
                        f"🔴 GLOBAL: {st.upper()} has negative avg Sharpe ({avg:.2f}) "
                        "across all regimes — consider deprioritising."
                    )
                elif avg > GOOD_SHARPE:
                    patterns.append(
                        f"🟢 GLOBAL: {st.upper()} has strong avg Sharpe ({avg:.2f}) — high priority."
                    )
            except Exception:
                pass
        return patterns

    def to_context_string(self, regime_label: str, strategy_types: List[str]) -> str:
        """Format patterns as a context string for LLM injection."""
        patterns = self.retrieve_patterns(regime_label, strategy_types)
        if not patterns:
            return f"No memory context available for {regime_label}."
        header = f"=== STRATEGY MEMORY ({regime_label}) ==="
        return header + "\n" + "\n".join(patterns)

    def store_swarm_results(
        self,
        final_ranking: List[Dict[str, Any]],
        regime_label: str,
    ) -> List[str]:
        """Persist top swarm results to memory for future runs."""
        run_ids = []
        for item in final_ranking[:5]:  # store top 5
            try:
                result = PastResult(
                    regime=regime_label,
                    strategy_type=item.get("strategy_type", ""),
                    params=json.dumps(item.get("params", {})),
                    sharpe=item.get("mean_sharpe", 0.0),
                    total_return=0.0,
                    max_drawdown=0.0,
                    confidence=item.get("robustness_score", 0.0),
                    generation_method="swarm",
                    reasoning=(
                        f"Multi-window robustness: mean={item.get('mean_sharpe', 0):.2f}, "
                        f"std={item.get('sharpe_std', 0):.2f}, "
                        f"min={item.get('min_sharpe', 0):.2f}"
                    ),
                )
                rid = self.memory.store(result)
                run_ids.append(rid)
            except Exception as e:
                logger.debug("Failed to store result: %s", e)
        return run_ids


def run_memory_agent(state: SwarmState) -> SwarmState:
    """
    Memory Agent node — called BEFORE specialists (to inject context)
    and AFTER coordinator (to persist results).

    Pre-run: reads patterns from history → injects into memory_context
    Post-run: persists final_ranking to memory

    This node is called twice in the orchestrator:
      1. After regime_analyst: retrieve patterns → memory_context
      2. After coordinator: store results
    """
    agent = MemoryAgent()
    regime_label = state.get("regime_context", None)
    if regime_label is None:
        return state
    regime_label = regime_label.regime_label
    strategy_types: List[str] = state.get("strategy_types", ["momentum"])

    # Phase A: Retrieve patterns (always run)
    if "memory_context" not in state or not state.get("memory_context"):
        patterns = agent.retrieve_patterns(regime_label, strategy_types)
        ctx_string = agent.to_context_string(regime_label, strategy_types)
        state["memory_patterns"] = patterns
        state["memory_context"] = ctx_string
        logger.info("[Memory Agent] Retrieved %d patterns for %s", len(patterns), regime_label)
        state.setdefault("run_log", []).append(
            f"[Memory Agent] {len(patterns)} patterns retrieved for {regime_label}"
        )

    # Phase B: Store results (if coordinator has run)
    if state.get("final_ranking"):
        run_ids = agent.store_swarm_results(state["final_ranking"], regime_label)
        if run_ids:
            logger.info("[Memory Agent] Stored %d swarm results.", len(run_ids))
            state["run_log"].append(f"[Memory Agent] Stored {len(run_ids)} results.")

    return state
