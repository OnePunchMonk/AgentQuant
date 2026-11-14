"""Bounded, offline HyDE-style hypothesis retrieval for research candidates.

This module generates synthetic descriptions, retrieves time-visible evidence,
and maps each description to a canonical parameter-grid point. Scores rank
research work only; they are not validation metrics or Sharpe estimates.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping, Protocol, Sequence

from src.agent.parameter_grid import ParameterGrid

MAX_HYPOTHESES = 16
MAX_EVIDENCE = 64


@dataclass(frozen=True)
class HypothesisRequest:
    regime: str
    strategy_types: tuple[str, ...]
    as_of: str
    max_hypotheses: int = 4
    max_evidence: int = 12
    knowledge_as_of: str | None = None

    def __post_init__(self) -> None:
        if not self.regime.strip() or not self.as_of.strip():
            raise ValueError("regime and market-time as_of are required")
        if not 0 <= self.max_hypotheses <= MAX_HYPOTHESES:
            raise ValueError(f"max_hypotheses must be between 0 and {MAX_HYPOTHESES}")
        if not 0 <= self.max_evidence <= MAX_EVIDENCE:
            raise ValueError(f"max_evidence must be between 0 and {MAX_EVIDENCE}")
        if self.knowledge_as_of is not None and not self.knowledge_as_of.strip():
            raise ValueError("knowledge_as_of must be non-empty when provided")


@dataclass(frozen=True)
class SyntheticHypothesis:
    description: str
    strategy_type: str
    proposed_params: Mapping[str, object] = field(default_factory=dict)


class Generator(Protocol):
    def generate(self, request: HypothesisRequest) -> Sequence[SyntheticHypothesis]: ...


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str
    source: str
    market_end: str | None
    known_at: str | None
    strategy_type: str
    params: Mapping[str, object] = field(default_factory=dict)
    quality: float = 0.0
    # Optional explicit measured metric; text alone is never interpreted as one.
    measured_sharpe: float | None = None
    evidence_tier: str = "unverified"


@dataclass(frozen=True)
class RetrievedEvidence:
    evidence_id: str
    source: str
    market_end: str | None
    known_at: str
    score: float
    quality: float
    evidence_tier: str
    measured_sharpe: float | None


@dataclass(frozen=True)
class ActionableCandidate:
    hypothesis: str
    strategy_type: str
    params: Mapping[str, object]
    retrieval_score: float
    quality_score: float
    evidence: tuple[RetrievedEvidence, ...]
    evaluation_status: str = "unevaluated"


class LexicalSimilarity:
    """Small deterministic token cosine scorer; intentionally dependency-free."""

    @staticmethod
    def _counts(text: str) -> dict[str, int]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        out: dict[str, int] = {}
        for word in words:
            out[word] = out.get(word, 0) + 1
        return out

    def __call__(self, left: str, right: str) -> float:
        a, b = self._counts(left), self._counts(right)
        if not a or not b:
            return 0.0
        dot = sum(value * b.get(token, 0) for token, value in a.items())
        norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
        return dot / norm if norm else 0.0


def _visible(record: Evidence, market_cutoff: date, knowledge_cutoff: date) -> bool:
    if record.market_end is None or record.known_at is None:
        return False
    try:
        market_date = date.fromisoformat(record.market_end[:10])
        knowledge_date = date.fromisoformat(record.known_at[:10])
        return market_date <= market_cutoff and knowledge_date <= knowledge_cutoff
    except ValueError:
        return False


def _canonical_params(h: SyntheticHypothesis, retrieved: Sequence[Evidence],
                      grids: ParameterGrid) -> dict[str, object]:
    grid = grids.get_grid(h.strategy_type)
    if not grid:
        return {}
    seeds = [dict(h.proposed_params)] + [dict(e.params) for e in retrieved
                                         if e.strategy_type == h.strategy_type]
    if not any(seed for seed in seeds):
        return {}
    def distance(candidate: Mapping[str, object]) -> float:
        best = float("inf")
        for seed in seeds:
            keys = [k for k, v in seed.items() if k in candidate and isinstance(v, (int, float))
                    and isinstance(candidate[k], (int, float))]
            d = sum(abs(float(seed[k]) - float(candidate[k])) /
                    (abs(float(seed[k])) + abs(float(candidate[k])) + 1e-9) for k in keys)
            if keys:
                best = min(best, d / len(keys))
        return best
    ranked = min(enumerate(grid), key=lambda pair: (distance(pair[1]), pair[0]))
    return dict(ranked[1]) if math.isfinite(distance(ranked[1])) else {}


class HydeRetrievalTool:
    """Generate, retrieve and canonically map hypotheses under explicit limits.

    `similarity` may be replaced with an embedding-backed callable. No provider
    is constructed or network access performed by this adapter.
    """

    def __init__(self, generator: Generator, evidence: Sequence[Evidence], *,
                 similarity: Callable[[str, str], float] | None = None,
                 grids: ParameterGrid | None = None):
        self.generator = generator
        self.evidence = tuple(evidence)
        self.similarity = similarity or LexicalSimilarity()
        self.grids = grids or ParameterGrid()

    def run(self, request: HypothesisRequest) -> list[ActionableCandidate]:
        cutoff = date.fromisoformat(request.as_of[:10])
        knowledge_cutoff = date.fromisoformat((request.knowledge_as_of or request.as_of)[:10])
        visible = [e for e in self.evidence if _visible(e, cutoff, knowledge_cutoff)]
        generated = self.generator.generate(request)[:request.max_hypotheses]
        results = []
        for hypothesis in generated:
            if hypothesis.strategy_type not in request.strategy_types:
                continue
            matching = [e for e in visible if e.strategy_type == hypothesis.strategy_type]
            ranked = sorted(matching,
                            key=lambda e: (-self.similarity(hypothesis.description, e.text), e.evidence_id))
            selected = ranked[:request.max_evidence]
            sims = [max(0.0, min(1.0, self.similarity(hypothesis.description, e.text)))
                    for e in selected]
            rel = sum(sims) / len(sims) if sims else 0.0
            qual = (sum(max(0.0, min(1.0, e.quality)) * s for e, s in zip(selected, sims)) /
                    sum(sims)) if sum(sims) else 0.0
            evidence = tuple(RetrievedEvidence(e.evidence_id, e.source, e.market_end, e.known_at,
                                               round(s, 6), max(0.0, min(1.0, e.quality)),
                                               e.evidence_tier, e.measured_sharpe)
                             for e, s in zip(selected, sims))
            params = _canonical_params(hypothesis, selected, self.grids)
            if not params:
                continue
            results.append(ActionableCandidate(hypothesis.description, hypothesis.strategy_type,
                                               params, round(rel, 6), round(qual, 6), evidence))
        return results


__all__ = ["ActionableCandidate", "Evidence", "Generator", "HydeRetrievalTool",
           "HypothesisRequest", "LexicalSimilarity", "RetrievedEvidence", "SyntheticHypothesis"]
