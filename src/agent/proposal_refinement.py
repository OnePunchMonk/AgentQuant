"""Small, deterministic diffusion-inspired proposal refinement experiment.

This is a generic candidate search utility.  It does not model a diffusion
process: callers inject both evaluation and neighborhood generation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

Candidate = TypeVar("Candidate")


@dataclass(frozen=True)
class RefinementResult(Generic[Candidate]):
    best_candidate: Candidate
    best_score: float
    evaluations: int
    rounds: int
    stopped_early: bool
    final_uncertainty: float


def refine_proposals(
    initial: Candidate,
    evaluate: Callable[[Candidate], float],
    propose_batch: Callable[[Candidate, float, random.Random, int], list[Candidate]],
    *,
    budget: int,
    batch_size: int = 4,
    initial_scale: float = 1.0,
    scale_decay: float = 0.5,
    uncertainty_threshold: float = 0.0,
    patience: int = 2,
    seed: int = 0,
) -> RefinementResult[Candidate]:
    """Refine the best candidate with coarse-to-fine speculative batches.

    ``budget`` is a hard upper bound on evaluator calls, including the initial
    candidate.  Batch candidates are generated from the same incumbent before
    any score in that batch is observed.  Uncertainty is the population
    standard deviation of the latest batch scores.  Early stopping requires
    both low uncertainty and ``patience`` rounds without improvement.
    Scores are maximized.
    """
    if budget < 1 or batch_size < 1 or patience < 1:
        raise ValueError("budget, batch_size, and patience must be positive")
    if initial_scale <= 0 or not 0 < scale_decay < 1 or uncertainty_threshold < 0:
        raise ValueError("invalid scale or uncertainty configuration")

    best_candidate = initial
    best_score = float(evaluate(initial))
    if not math.isfinite(best_score):
        raise ValueError("evaluator must return a finite score")
    calls, rounds, stale_rounds = 1, 0, 0
    scale = initial_scale
    uncertainty = math.inf
    rng = random.Random(seed)
    stopped_early = False

    while calls < budget:
        count = min(batch_size, budget - calls)
        # Generate the entire speculative batch before evaluating any member.
        candidates = propose_batch(best_candidate, scale, rng, count)
        if len(candidates) != count:
            raise ValueError("propose_batch must return exactly the requested count")
        scores: list[float] = []
        prior_best = best_score
        for candidate in candidates:
            score = float(evaluate(candidate))
            if not math.isfinite(score):
                raise ValueError("evaluator must return a finite score")
            scores.append(score)
            calls += 1
            if score > best_score:
                best_candidate, best_score = candidate, score
        rounds += 1
        mean = sum(scores) / len(scores)
        uncertainty = math.sqrt(sum((score - mean) ** 2 for score in scores) / len(scores))
        stale_rounds = stale_rounds + 1 if best_score <= prior_best else 0
        scale *= scale_decay
        if uncertainty <= uncertainty_threshold and stale_rounds >= patience and calls < budget:
            stopped_early = True
            break

    if rounds == 0:
        uncertainty = 0.0
    return RefinementResult(best_candidate, best_score, calls, rounds, stopped_early, uncertainty)
