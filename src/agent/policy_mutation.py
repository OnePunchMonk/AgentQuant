"""
Policy Mutation — P2 bounded self-improvement outer loop.

Inner loop: the existing propose -> backtest -> reflect search
(src.agent.agent_graph.run_agent), invoked per-episode with a given
"policy" (a HarnessConfig -- prompt_template / prompt_context).

Outer loop (this module): a bounded mechanism that mutates that policy
between episodes. The single mutation family implemented here is
prompt_template / prompt_context mutation (the simplest knob
resolve_effective_config already wires all the way through).

Promotion is gated on a fixed, predeclared improvement threshold plus a
protected-episode regression check -- persisting a new config is NOT itself
self-improvement. A random-mutation baseline runs under the identical
budget for comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.episode_splits import Episode
from src.agent.harness_config import HarnessConfig

logger = logging.getLogger(__name__)

# Predeclared, fixed, practically-meaningful promotion threshold. A candidate
# must beat the incumbent's mean validation holdout Sharpe by more than this
# to be promoted. Documented here so it cannot be tuned post-hoc per result.
PROMOTION_EPSILON = 0.10

# Regression check: performance on this episode must not drop by more than
# this fraction relative to the incumbent, or the candidate is rejected even
# if it clears PROMOTION_EPSILON on average.
PROTECTED_EPISODE_ID = "ep00"
MAX_PROTECTED_REGRESSION = 0.25


PROMPT_MUTATION_POOL = [
    {"prompt_template": "grid_search_default", "prompt_context": {}},
    {"prompt_template": "tool_aware_default", "prompt_context": {"emphasis": ["momentum in bull markets"]}},
    {"prompt_template": "tool_aware_tuned_v2_learnings",
     "prompt_context": {"emphasis": ["short windows in crisis"], "avoid": ["overfit combos"]}},
    {"prompt_template": "research_informed", "prompt_context": {"hypothesis_generation": "enabled"}},
]


@dataclass
class PolicyMutationRecord:
    """A candidate policy mutation, fully auditable."""

    mutation_id: str
    parent_policy_id: str
    patch: Dict[str, Any]  # the diff applied to the parent config
    diagnosis: str  # why this mutation was proposed
    expected_benefit: str  # a number or short justification
    evaluation_budget: int  # number of episodes/backtests it's allowed
    rollback_ref: str  # parent policy id / config hash to revert to
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "parent_policy_id": self.parent_policy_id,
            "patch": self.patch,
            "diagnosis": self.diagnosis,
            "expected_benefit": self.expected_benefit,
            "evaluation_budget": self.evaluation_budget,
            "rollback_ref": self.rollback_ref,
            "created": self.created,
        }


def _policy_id(config: HarnessConfig) -> str:
    payload = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def propose_mutation(
    parent: HarnessConfig,
    rng: random.Random,
    diagnosis: str = "heuristic: rotate prompt template/context based on prior reflect output",
    evaluation_budget: int = 2,
) -> Tuple["PolicyMutationRecord", HarnessConfig]:
    """Propose a diagnosis-driven mutation: pick a different point in the
    prompt-template/context space than the parent."""
    pool = [p for p in PROMPT_MUTATION_POOL if p["prompt_template"] != parent.prompt_template]
    patch = rng.choice(pool) if pool else PROMPT_MUTATION_POOL[0]
    child = HarnessConfig(**{**parent.to_dict()})
    child.version = f"mut_{uuid.uuid4().hex[:8]}"
    child.epoch = parent.epoch + 1
    child.created = datetime.now(timezone.utc).isoformat()
    child.prompt_template = patch["prompt_template"]
    child.prompt_context = dict(patch["prompt_context"])
    record = PolicyMutationRecord(
        mutation_id=child.version,
        parent_policy_id=_policy_id(parent),
        patch=patch,
        diagnosis=diagnosis,
        expected_benefit="untested; evaluated empirically on dev episodes",
        evaluation_budget=evaluation_budget,
        rollback_ref=_policy_id(parent),
    )
    return record, child


def propose_random_mutation(
    parent: HarnessConfig, rng: random.Random, evaluation_budget: int = 2,
) -> Tuple["PolicyMutationRecord", HarnessConfig]:
    """Random-mutation baseline: identical interface and budget, but no
    diagnosis-driven selection -- purely random choice from the same pool."""
    patch = rng.choice(PROMPT_MUTATION_POOL)
    child = HarnessConfig(**{**parent.to_dict()})
    child.version = f"randmut_{uuid.uuid4().hex[:8]}"
    child.epoch = parent.epoch + 1
    child.created = datetime.now(timezone.utc).isoformat()
    child.prompt_template = patch["prompt_template"]
    child.prompt_context = dict(patch["prompt_context"])
    record = PolicyMutationRecord(
        mutation_id=child.version,
        parent_policy_id=_policy_id(parent),
        patch=patch,
        diagnosis="random_baseline: uniformly sampled mutation, no diagnosis",
        expected_benefit="none claimed (baseline)",
        evaluation_budget=evaluation_budget,
        rollback_ref=_policy_id(parent),
    )
    return record, child


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    candidate_mean: Optional[float]
    incumbent_mean: Optional[float]
    protected_delta: Optional[float]


def evaluate_promotion(
    incumbent_val_scores: List[float],
    candidate_val_scores: List[float],
    incumbent_protected_score: Optional[float],
    candidate_protected_score: Optional[float],
    epsilon: float = PROMOTION_EPSILON,
    max_protected_regression: float = MAX_PROTECTED_REGRESSION,
) -> PromotionDecision:
    """Only promote a candidate policy over the incumbent if it clears a
    fixed, predeclared improvement threshold on validation episodes AND does
    not regress badly on the protected episode. Ties/inconclusive results
    keep the incumbent."""
    if not incumbent_val_scores or not candidate_val_scores:
        return PromotionDecision(False, "missing validation scores; keeping incumbent", None, None, None)

    inc_mean = sum(incumbent_val_scores) / len(incumbent_val_scores)
    cand_mean = sum(candidate_val_scores) / len(candidate_val_scores)
    delta = cand_mean - inc_mean

    protected_delta = None
    if incumbent_protected_score is not None and candidate_protected_score is not None:
        protected_delta = candidate_protected_score - incumbent_protected_score
        if protected_delta < -abs(max_protected_regression):
            return PromotionDecision(
                False,
                f"protected-episode regression {protected_delta:.3f} exceeds "
                f"-{max_protected_regression}; keeping incumbent",
                cand_mean, inc_mean, protected_delta,
            )

    if delta > epsilon:
        return PromotionDecision(
            True, f"candidate beats incumbent by {delta:.3f} > epsilon={epsilon}; promoting",
            cand_mean, inc_mean, protected_delta,
        )

    return PromotionDecision(
        False,
        f"candidate delta {delta:.3f} does not clear epsilon={epsilon} (inconclusive/tie/loss); "
        "keeping incumbent",
        cand_mean, inc_mean, protected_delta,
    )


class FinalHoldoutGuard:
    """Tracks whether the final holdout episodes have already been used for
    a frozen evaluation. A repeat run errors loudly rather than silently
    re-peeking sealed data."""

    def __init__(self, path: Path = Path("experiments/final_holdout_used.json")):
        self.path = path

    def check_and_mark_used(self, run_label: str) -> None:
        used: Dict[str, Any] = {}
        if self.path.exists():
            used = json.loads(self.path.read_text())
        if used.get("used"):
            raise RuntimeError(
                f"Final holdout episodes were already consumed by run "
                f"{used.get('run_label')!r} at {used.get('used_at')!r}. "
                f"Refusing to re-peek sealed holdout data for run {run_label!r}. "
                f"Delete {self.path} only if you deliberately intend to reset the benchmark."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "used": True, "run_label": run_label,
            "used_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))


EvalFn = Callable[[HarnessConfig, Episode, int], float]


def run_bounded_self_improvement(
    incumbent: HarnessConfig,
    dev_episodes: List[Episode],
    val_episodes: List[Episode],
    final_episodes: List[Episode],
    protected_episode: Episode,
    eval_fn: EvalFn,
    seeds: List[int],
    n_mutations: int = 3,
    holdout_guard: Optional[FinalHoldoutGuard] = None,
    use_random_baseline: bool = False,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Ties inner (eval_fn) and outer (mutation/promotion) loops together.

    eval_fn(policy, episode, seed) -> holdout sharpe for that episode/seed,
    reusing whatever inner-loop machinery the caller wires up (e.g.
    src.agent.search_arms.run_frozen_agent_arm under the hood).
    """
    rng = random.Random(rng_seed)
    propose_fn = propose_random_mutation if use_random_baseline else propose_mutation

    mutation_records = []
    candidates: List[HarnessConfig] = []
    for _ in range(n_mutations):
        record, child = propose_fn(incumbent, rng)
        mutation_records.append(record)
        candidates.append(child)

    # Evaluate each candidate on dev episodes (search/selection data).
    dev_scores: Dict[str, List[float]] = {}
    for cand in candidates:
        scores = [eval_fn(cand, ep, seed) for ep in dev_episodes for seed in seeds]
        dev_scores[cand.version] = [s for s in scores if s is not None]

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("-inf")

    best_candidate = max(candidates, key=lambda c: _mean(dev_scores[c.version]))

    # Selection happens on a SEPARATE validation episode slice.
    incumbent_val = [eval_fn(incumbent, ep, seed) for ep in val_episodes for seed in seeds]
    candidate_val = [eval_fn(best_candidate, ep, seed) for ep in val_episodes for seed in seeds]
    incumbent_val = [s for s in incumbent_val if s is not None]
    candidate_val = [s for s in candidate_val if s is not None]

    incumbent_protected = eval_fn(incumbent, protected_episode, seeds[0])
    candidate_protected = eval_fn(best_candidate, protected_episode, seeds[0])

    decision = evaluate_promotion(incumbent_val, candidate_val, incumbent_protected, candidate_protected)
    selected_policy = best_candidate if decision.promote else incumbent

    # ONE final frozen evaluation of the SELECTED policy on held-out final
    # episodes. Guarded against reuse.
    guard = holdout_guard or FinalHoldoutGuard()
    run_label = f"{selected_policy.version}-{uuid.uuid4().hex[:6]}"
    guard.check_and_mark_used(run_label)
    final_scores = [eval_fn(selected_policy, ep, seed) for ep in final_episodes for seed in seeds]
    final_scores = [s for s in final_scores if s is not None]

    return {
        "mutation_records": [r.to_dict() for r in mutation_records],
        "dev_scores": dev_scores,
        "best_candidate_version": best_candidate.version,
        "incumbent_val_scores": incumbent_val,
        "candidate_val_scores": candidate_val,
        "protected_episode_id": protected_episode.episode_id,
        "incumbent_protected_score": incumbent_protected,
        "candidate_protected_score": candidate_protected,
        "promotion_decision": {
            "promote": decision.promote,
            "reason": decision.reason,
            "candidate_mean": decision.candidate_mean,
            "incumbent_mean": decision.incumbent_mean,
            "protected_delta": decision.protected_delta,
        },
        "selected_policy_version": selected_policy.version,
        "final_holdout_scores": final_scores,
        "final_holdout_mean": _mean(final_scores) if final_scores else None,
        "used_random_baseline": use_random_baseline,
    }
