"""Hypothesis generation and evaluation orchestration primitives.

This module ranks research proposals only. Evaluation is a separate, explicit
step, so generated predictions cannot be mistaken for measured performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

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

