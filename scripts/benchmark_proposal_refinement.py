"""Offline synthetic comparison; no market data or trading claims."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.proposal_refinement import refine_proposals


def propose(center: float, scale: float, rng: random.Random, count: int) -> list[float]:
    return [center + rng.uniform(-scale, scale) for _ in range(count)]


def objective(x: float) -> float:
    # Deterministic multimodal toy objective with a known broad optimum.
    return -(x - 1.25) ** 2 + 0.12 * math.sin(5 * x)


def main() -> None:
    scheduler_rows = []
    baseline_rows = []
    nominal_budget, trials = 33, 100
    for seed in range(trials):
        refined = refine_proposals(
            0.0, objective, propose, budget=nominal_budget, batch_size=4,
            initial_scale=1.5, uncertainty_threshold=0.30, patience=2, seed=seed,
        )
        # Fixed-step comparator runs the same proposals for exactly the calls
        # used by the scheduler. It has no confidence-based termination.
        fixed = refine_proposals(
            0.0, objective, propose, budget=refined.evaluations, batch_size=4,
            initial_scale=1.5, uncertainty_threshold=0.0, patience=10**9, seed=seed,
        )
        assert fixed.evaluations == refined.evaluations
        scheduler_rows.append((refined.best_score, refined.evaluations, refined.stopped_early))
        baseline_rows.append((fixed.best_score, fixed.evaluations))

    def mean(values):
        return sum(values) / len(values)
    payload = {
        "fixture": "deterministic one-dimensional multimodal toy objective",
        "trials": trials,
        "nominal_evaluator_budget": nominal_budget,
        "equal_evaluator_calls": True,
        "scheduler": {
            "mean_best_score": mean([row[0] for row in scheduler_rows]),
            "mean_evaluator_calls": mean([row[1] for row in scheduler_rows]),
            "early_stop_trials": sum(row[2] for row in scheduler_rows),
        },
        "fixed_step_baseline_at_equal_calls": {
            "mean_best_score": mean([row[0] for row in baseline_rows]),
            "mean_evaluator_calls": mean([row[1] for row in baseline_rows]),
        },
        "interpretation": "Synthetic mechanics check only; no evidence of actual diffusion or trading benefit.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
