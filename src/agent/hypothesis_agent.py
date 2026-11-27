"""Hypothesis generation and evaluation orchestration primitives.

This module ranks research proposals only. Evaluation is a separate, explicit
step, so generated predictions cannot be mistaken for measured performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from src.agent.hyde_retrieval import (
    ActionableCandidate,
    Evidence,
    HydeRetrievalTool,
    HypothesisRequest,
)


@dataclass(frozen=True)
class HypothesisRun:
    regime: str
    method: str
    candidates: tuple[ActionableCandidate, ...]
    cost_units: float
    source_label: str = "fixture"
    metrics: Mapping[str, float] = field(default_factory=dict)


class HypothesisAgent:
    """Run a bounded generator/retriever under time-visible evidence rules."""

    DEFAULT_COST = {"literature": 1.0, "hyde": 1.0, "analytical": 0.0, "ensemble": 2.0}

    def __init__(self, generators: Mapping[str, object], evidence: Sequence[Evidence]):
        self.generators = dict(generators)
        self.evidence = tuple(evidence)

    def run(self, method: str, request: HypothesisRequest, *, cost_units: float | None = None):
        if method not in self.generators:
            raise ValueError(f"unknown generation method: {method}")
        tool = HydeRetrievalTool(self.generators[method], self.evidence)
        candidates = tuple(tool.run(request))
        return HypothesisRun(request.regime, method, candidates,
                             self.DEFAULT_COST.get(method, 0.0) if cost_units is None else cost_units)

    def evaluate(self, run: HypothesisRun, prices: Any, *, train_start: Any, train_end: Any,
                 holdout_end: Any, cost_bps: float = 5.0) -> HypothesisRun:
        """Score a run's candidates with the shared evaluator. This is the explicit second step.

        Generation only ranks ideas, so candidates leave ``run`` as ``unevaluated``. Here each one is
        scored on a training window (``train_start`` to ``train_end``) and on the holdout that follows
        (after ``train_end`` up to ``holdout_end``). The signal is computed on all history up to the end
        of the window being scored, so moving averages have their warmup, and nothing after
        ``train_end`` reaches the training score. ``prices`` is a frame with a DatetimeIndex and a
        ``Close`` column. Candidates whose strategy is unknown are left ``unevaluated``.
        """
        import pandas as pd

        from src.backtest.core import evaluate_window
        from src.strategies.strategy_registry import STRATEGY_REGISTRY

        cost = cost_bps / 10_000.0
        train_history = prices.loc[: pd.Timestamp(train_end)]
        full_history = prices.loc[: pd.Timestamp(holdout_end)]
        after = full_history.index[full_history.index > pd.Timestamp(train_end)]
        holdout_start = after[0] if len(after) else None
        scored, holdouts = [], []
        for candidate in run.candidates:
            if candidate.strategy_type not in STRATEGY_REGISTRY:
                scored.append(candidate)
                continue
            params = dict(candidate.params)
            train = evaluate_window(train_history, candidate.strategy_type, params, cost, eval_start=train_start)
            holdout = (evaluate_window(full_history, candidate.strategy_type, params, cost,
                                       eval_start=holdout_start) if holdout_start is not None else None)
            if holdout is not None:
                holdouts.append(holdout["sharpe"])
            scored.append(replace(candidate, evaluation_status="evaluated",
                                  evaluation={"train": train, "holdout": holdout, "cost_bps": cost_bps}))
        metrics = dict(run.metrics)
        if holdouts:
            metrics["holdout_mean_sharpe"] = sum(holdouts) / len(holdouts)
        return replace(run, candidates=tuple(scored), metrics=metrics)
