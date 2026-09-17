"""
Hypothesis Scorer — Embedding-Based Retrieval Scoring
========================================================

Scores synthetic hypotheses (from hypothesis_generator.py) by retrieving
the most similar prior strategy runs from StrategyMemory / NLAMemoryStore
and combining retrieval evidence with historical backtest quality.

Embedding note: this is a hashed bag-of-words embedding, not a learned
semantic model. It is deterministic, has no external dependency or API
cost, and is good enough to rank "which past runs talked about the same
regime/strategy combination" -- which is what evidence density needs.
Swapping in a real sentence embedding (OpenAI/Gemini embeddings API) is a
drop-in change: only `_embed` needs to be replaced.
"""

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from src.agent.hypothesis_generator import Hypothesis
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.research.nla_memory import NLAMemoryStore, NLARecord

logger = logging.getLogger(__name__)

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    """Deterministic across processes, unlike builtin hash() (salted per-run)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def _embed(text: str, dim: int = _EMBED_DIM) -> np.ndarray:
    """Hashed bag-of-words embedding, L2-normalized."""
    vec = np.zeros(dim, dtype=np.float64)
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        idx = _stable_hash(token) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class RetrievedMatch:
    """A single piece of evidence retrieved for a hypothesis."""

    source: str  # "strategy_memory" or "nla_memory"
    text: str
    similarity: float
    sharpe: float = 0.0


@dataclass
class ScoredHypothesis:
    """A hypothesis annotated with retrieval evidence and a composite score."""

    hypothesis: Hypothesis
    matches: List[RetrievedMatch] = field(default_factory=list)
    evidence_density: float = 0.0
    backtest_quality: float = 0.0
    composite_score: float = 0.0


class HypothesisScorer:
    """Ranks hypotheses by evidence density + historical backtest quality."""

    def __init__(
        self,
        strategy_memory: Optional[StrategyMemory] = None,
        nla_memory: Optional[NLAMemoryStore] = None,
        top_k: int = 5,
        evidence_weight: float = 0.6,
        backtest_weight: float = 0.4,
    ):
        self.strategy_memory = strategy_memory or StrategyMemory()
        self.nla_memory = nla_memory or NLAMemoryStore()
        self.top_k = top_k
        self.evidence_weight = evidence_weight
        self.backtest_weight = backtest_weight

    def score(self, hypothesis: Hypothesis) -> ScoredHypothesis:
        query_vec = _embed(hypothesis.hypothesis)

        past_results = self.strategy_memory.query_all(
            strategy_type=hypothesis.strategy_type, limit=200
        )
        nla_records = self.nla_memory.recall(
            strategy_type=hypothesis.strategy_type, n=200
        )

        matches = self._retrieve(query_vec, past_results, nla_records)
        top_matches = matches[: self.top_k]

        evidence_density = self._evidence_density(top_matches)
        backtest_quality = self._backtest_quality(top_matches)
        composite = (
            self.evidence_weight * evidence_density
            + self.backtest_weight * backtest_quality
        )

        return ScoredHypothesis(
            hypothesis=hypothesis,
            matches=top_matches,
            evidence_density=evidence_density,
            backtest_quality=backtest_quality,
            composite_score=composite,
        )

    def score_all(self, hypotheses: Sequence[Hypothesis]) -> List[ScoredHypothesis]:
        """Score and rank a batch of hypotheses, best first."""
        scored = [self.score(h) for h in hypotheses]
        scored.sort(key=lambda s: s.composite_score, reverse=True)
        return scored

    def _retrieve(
        self,
        query_vec: np.ndarray,
        past_results: Sequence[PastResult],
        nla_records: Sequence[NLARecord],
    ) -> List[RetrievedMatch]:
        matches: List[RetrievedMatch] = []

        for r in past_results:
            text = f"{r.strategy_type} {r.regime} {r.reasoning}".strip()
            sim = _cosine(query_vec, _embed(text))
            if sim <= 0:
                continue
            matches.append(RetrievedMatch(
                source="strategy_memory", text=text, similarity=sim, sharpe=r.sharpe,
            ))

        for rec in nla_records:
            text = f"{rec.strategy_type} {rec.regime} {rec.narrative}".strip()
            sim = _cosine(query_vec, _embed(text))
            if sim <= 0:
                continue
            matches.append(RetrievedMatch(
                source="nla_memory", text=text, similarity=sim, sharpe=rec.quality_score,
            ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    @staticmethod
    def _evidence_density(matches: Sequence[RetrievedMatch]) -> float:
        """Mean similarity of retrieved evidence, scaled toward 1.0 as
        both similarity and count of matches grow (more corroborating
        evidence at high similarity is stronger than one weak match)."""
        if not matches:
            return 0.0
        mean_sim = sum(m.similarity for m in matches) / len(matches)
        count_factor = 1.0 - math.exp(-len(matches) / 3.0)
        return mean_sim * count_factor

    @staticmethod
    def _backtest_quality(matches: Sequence[RetrievedMatch]) -> float:
        """Average historical Sharpe of retrieved evidence, similarity-weighted."""
        if not matches:
            return 0.0
        weight_sum = sum(m.similarity for m in matches)
        if weight_sum <= 0:
            return 0.0
        weighted_sharpe = sum(m.similarity * m.sharpe for m in matches) / weight_sum
        # Squash into a roughly [0, 1] range; Sharpe of 1.0+ maps near 1.0.
        return max(0.0, min(1.0, weighted_sharpe))
