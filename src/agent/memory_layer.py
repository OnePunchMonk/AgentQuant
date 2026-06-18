"""
Agentic memory layer for cross-run strategy learning.

This module turns persisted strategy rows into operational patterns that can be
shown in the CLI, injected into prompts, and used by the swarm memory agent.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.strategy_memory import PastResult, StrategyMemory

PATTERN_MIN_SAMPLES = 2
GOOD_SHARPE = 0.5
BAD_SHARPE = 0.0


@dataclass
class StrategyPattern:
    """A compact statement about what memory has learned."""

    scope: str
    verdict: str
    evidence: str
    sample_size: int
    avg_sharpe: float
    best_sharpe: float
    params: Dict[str, Any] = field(default_factory=dict)

    def to_sentence(self) -> str:
        param_text = f" params={self.params}" if self.params else ""
        return (
            f"{self.verdict}: {self.scope} | n={self.sample_size}, "
            f"avg_sharpe={self.avg_sharpe:.2f}, best={self.best_sharpe:.2f} | "
            f"{self.evidence}{param_text}"
        )


class AgenticMemoryLayer:
    """Pattern extractor and browser over StrategyMemory."""

    def __init__(self, memory: Optional[StrategyMemory] = None):
        self.memory = memory or StrategyMemory()

    def recent_runs(
        self,
        regime: str = "",
        strategy_type: str = "",
        limit: int = 25,
    ) -> List[PastResult]:
        return self.memory.list_runs(regime=regime, strategy_type=strategy_type, limit=limit)

    def summary_rows(
        self,
        regime: str = "",
        strategy_type: str = "",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        return self.memory.summarize(regime=regime, strategy_type=strategy_type, limit=limit)

    def extract_patterns(
        self,
        regime: str = "",
        strategy_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[StrategyPattern]:
        if strategy_types:
            runs: List[PastResult] = []
            for strategy in strategy_types:
                runs.extend(
                    self.memory.list_runs(
                        regime=regime,
                        strategy_type=strategy,
                        limit=limit,
                        order_by="timestamp",
                    )
                )
        else:
            runs = self.memory.list_runs(regime=regime, limit=limit)

        patterns: List[StrategyPattern] = []
        by_scope: Dict[tuple, List[PastResult]] = defaultdict(list)
        for run in runs:
            by_scope[(run.regime, run.strategy_type)].append(run)

        for (scope_regime, strategy), history in sorted(by_scope.items()):
            if not history:
                continue
            sharpes = [h.sharpe for h in history]
            avg_sharpe = sum(sharpes) / len(sharpes)
            best = max(history, key=lambda h: h.sharpe)
            worst = min(history, key=lambda h: h.sharpe)
            verdict = self._verdict(avg_sharpe, best.sharpe, worst.sharpe)
            patterns.append(
                StrategyPattern(
                    scope=f"{strategy} in {scope_regime}",
                    verdict=verdict,
                    evidence=(
                        f"{sum(1 for h in history if h.sharpe >= GOOD_SHARPE)} good runs, "
                        f"{sum(1 for h in history if h.sharpe <= BAD_SHARPE)} rejected-or-poor runs"
                    ),
                    sample_size=len(history),
                    avg_sharpe=avg_sharpe,
                    best_sharpe=best.sharpe,
                    params=self._decode_params(best.params),
                )
            )
            patterns.extend(self._parameter_patterns(scope_regime, strategy, history))

        return patterns

    def to_prompt_context(
        self,
        regime: str,
        strategy_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> str:
        patterns = self.extract_patterns(regime=regime, strategy_types=strategy_types, limit=limit)
        if not patterns:
            return f"AGENTIC MEMORY: no learned patterns yet for {regime}."
        lines = [f"AGENTIC MEMORY PATTERNS FOR {regime}:"]
        lines.extend(f"- {pattern.to_sentence()}" for pattern in patterns[:12])
        return "\n".join(lines)

    def render_markdown(
        self,
        regime: str = "",
        strategy_type: str = "",
        limit: int = 25,
    ) -> str:
        runs = self.recent_runs(regime=regime, strategy_type=strategy_type, limit=limit)
        return self.memory.export_markdown(runs)

    def _parameter_patterns(
        self,
        regime: str,
        strategy: str,
        history: List[PastResult],
    ) -> List[StrategyPattern]:
        if len(history) < PATTERN_MIN_SAMPLES:
            return []

        buckets: Dict[str, List[PastResult]] = defaultdict(list)
        for run in history:
            params = self._decode_params(run.params)
            bucket = self._bucket_params(strategy, params)
            if bucket:
                buckets[bucket].append(run)

        patterns = []
        for bucket, bucket_runs in buckets.items():
            if len(bucket_runs) < PATTERN_MIN_SAMPLES:
                continue
            sharpes = [run.sharpe for run in bucket_runs]
            avg_sharpe = sum(sharpes) / len(sharpes)
            best = max(bucket_runs, key=lambda h: h.sharpe)
            patterns.append(
                StrategyPattern(
                    scope=f"{strategy} {bucket} in {regime}",
                    verdict=self._verdict(avg_sharpe, best.sharpe, min(sharpes)),
                    evidence="parameter cluster repeated across stored runs",
                    sample_size=len(bucket_runs),
                    avg_sharpe=avg_sharpe,
                    best_sharpe=best.sharpe,
                    params=self._decode_params(best.params),
                )
            )
        return patterns

    @staticmethod
    def _decode_params(raw: str) -> Dict[str, Any]:
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _bucket_params(strategy: str, params: Dict[str, Any]) -> str:
        if strategy == "momentum" and "slow_window" in params:
            slow = int(params["slow_window"])
            start = (slow // 30) * 30
            return f"slow_window={start}-{start + 29}"
        if strategy == "mean_reversion" and "window" in params:
            window = int(params["window"])
            start = (window // 10) * 10
            return f"window={start}-{start + 9}"
        if strategy == "volatility" and "vol_threshold" in params:
            threshold = round(float(params["vol_threshold"]), 2)
            return f"vol_threshold~{threshold:.2f}"
        return ""

    @staticmethod
    def _verdict(avg_sharpe: float, best_sharpe: float, worst_sharpe: float) -> str:
        if avg_sharpe >= GOOD_SHARPE:
            return "worked"
        if best_sharpe >= GOOD_SHARPE and worst_sharpe <= BAD_SHARPE:
            return "regime-sensitive"
        if avg_sharpe <= BAD_SHARPE:
            return "avoid"
        return "mixed"
