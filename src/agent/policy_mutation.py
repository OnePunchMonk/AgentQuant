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
class StructuredFailure:
    """One structured, machine-checkable failure signal about the incumbent
    policy's recent episode(s) -- e.g. derived from reflect/backtest output,
    not free-text. `tag` must be one of FAILURE_TAG_TO_PATCH's keys."""

    tag: str
    evidence: str  # e.g. "protected episode drawdown 0.31 > max_acceptable_drawdown 0.20"


# Fixed, documented mapping from a structured failure tag to the single
# candidate patch it selects. This is the entire evidence-conditioned
# candidate space -- adding a tag or a patch requires touching this table,
# so the reason-to-patch mapping stays traceable and cannot silently drift.
# Every value must be a template name present in PROMPT_MUTATION_POOL.
FAILURE_TAG_TO_PATCH: Dict[str, str] = {
    # Sharpe/return well below threshold with no crisis/drawdown signal:
    # try surfacing tool-derived signals instead of the un-augmented grid.
    "low_return_no_tools": "tool_aware_default",
    # Large drawdown / crisis-period underperformance: the pool's only patch
    # that explicitly narrows windows and avoids overfit combos in crises.
    "high_drawdown": "tool_aware_tuned_v2_learnings",
    "overfit_selection": "tool_aware_tuned_v2_learnings",
    # Reflect output indicates the policy is not grounding proposals in any
    # external evidence/hypothesis: switch to the research-informed template.
    "ungrounded_hypothesis": "research_informed",
    # No specific failure identified (e.g. regressed for an unclear reason):
    # fall back to the unaugmented baseline rather than guessing a "fix".
    "unspecified": "grid_search_default",
}

_TEMPLATE_TO_PATCH = {p["prompt_template"]: p for p in PROMPT_MUTATION_POOL}
assert set(FAILURE_TAG_TO_PATCH.values()) <= set(_TEMPLATE_TO_PATCH), (
    "FAILURE_TAG_TO_PATCH references a template not in PROMPT_MUTATION_POOL"
)


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


def _patch_for_failures(
    failures: List["StructuredFailure"], parent_template: str
) -> Tuple[Dict[str, Any], str]:
    """Deterministically select the patch dictated by the dominant (first)
    structured failure, falling through to the next failure in order if the
    dominant one maps to the parent's own template (a no-op mutation), and
    finally to the fixed "unspecified" fallback. Never falls back to random
    choice -- an unrecognized tag is a caller bug, not evidence to ignore."""
    for failure in failures:
        if failure.tag not in FAILURE_TAG_TO_PATCH:
            raise ValueError(
                f"unrecognized structured failure tag {failure.tag!r}; must be one of "
                f"{sorted(FAILURE_TAG_TO_PATCH)} (add it to FAILURE_TAG_TO_PATCH, don't guess)"
            )
        template = FAILURE_TAG_TO_PATCH[failure.tag]
        if template != parent_template:
            reason = f"evidence-conditioned: failure={failure.tag!r} ({failure.evidence}) -> {template!r}"
            return _TEMPLATE_TO_PATCH[template], reason

    # Every mapped patch was a no-op relative to the parent (or no failures
    # were supplied): fall back to the fixed unspecified-failure patch,
    # rotating to the next distinct pool entry if even that is a no-op.
    fallback_template = FAILURE_TAG_TO_PATCH["unspecified"]
    if fallback_template == parent_template:
        distinct = [p for p in PROMPT_MUTATION_POOL if p["prompt_template"] != parent_template]
        patch = distinct[0] if distinct else PROMPT_MUTATION_POOL[0]
    else:
        patch = _TEMPLATE_TO_PATCH[fallback_template]
    tags = [f.tag for f in failures]
    reason = (
        f"no non-no-op mapped patch among structured failures {tags}; "
        f"falling back to unspecified-failure patch {patch['prompt_template']!r}"
        if failures else
        "no structured failure evidence supplied; using unspecified-failure patch "
        f"{patch['prompt_template']!r}"
    )
    return patch, reason


def propose_mutation(
    parent: HarnessConfig,
    rng: random.Random,
    failures: Optional[List["StructuredFailure"]] = None,
    evaluation_budget: int = 2,
) -> Tuple["PolicyMutationRecord", HarnessConfig]:
    """Propose an evidence-conditioned mutation: `failures` (structured,
    machine-checkable failure signals from the parent's prior episode(s))
    deterministically selects the patch via FAILURE_TAG_TO_PATCH -- diagnosis
    text is never itself the selector, so a mutation can always be traced
    back to the specific failure tag/evidence that caused it. `rng` is
    accepted for interface parity with propose_random_mutation but is not
    used to choose the patch."""
    del rng  # unused: patch selection here is evidence-driven, not random
    patch, reason = _patch_for_failures(failures or [], parent.prompt_template)
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
        diagnosis=reason,
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
    coverage_mismatch: bool = False,
) -> PromotionDecision:
    """Only promote a candidate policy over the incumbent if it clears a
    fixed, predeclared improvement threshold on validation episodes AND does
    not regress badly on the protected episode. Ties/inconclusive results
    keep the incumbent.

    Fails closed (never promotes) if:
      - `coverage_mismatch` is True -- the caller determined the incumbent
        and candidate were not evaluated on the same set of episodes/seeds,
        so comparing their (independently-filtered) score lists would not
        be a valid apples-to-apples comparison.
      - either protected-episode score is missing -- the regression check
        cannot be verified, so promotion must not proceed as if it passed.
    """
    if coverage_mismatch:
        return PromotionDecision(
            False,
            "incumbent and candidate were evaluated on mismatched validation "
            "episode/seed coverage; comparison is invalid, keeping incumbent",
            None, None, None,
        )

    if not incumbent_val_scores or not candidate_val_scores:
        return PromotionDecision(False, "missing validation scores; keeping incumbent", None, None, None)

    if incumbent_protected_score is None or candidate_protected_score is None:
        return PromotionDecision(
            False,
            "protected-episode evaluation unavailable for incumbent and/or candidate; "
            "failing closed (cannot verify no regression), keeping incumbent",
            None, None, None,
        )

    inc_mean = sum(incumbent_val_scores) / len(incumbent_val_scores)
    cand_mean = sum(candidate_val_scores) / len(candidate_val_scores)
    delta = cand_mean - inc_mean

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
    structured_failures: Optional[List["StructuredFailure"]] = None,
) -> Dict[str, Any]:
    """Ties inner (eval_fn) and outer (mutation/promotion) loops together.

    eval_fn(policy, episode, seed) -> holdout sharpe for that episode/seed,
    reusing whatever inner-loop machinery the caller wires up (e.g.
    src.agent.search_arms.run_frozen_agent_arm under the hood).

    `structured_failures`, when supplied, drives evidence-conditioned patch
    selection for every candidate (see `_patch_for_failures`); it is ignored
    by the random baseline, which must sample the identical candidate space
    without using the evidence -- that's the whole point of the control.
    """
    rng = random.Random(rng_seed)

    mutation_records = []
    candidates: List[HarnessConfig] = []
    for _ in range(n_mutations):
        if use_random_baseline:
            record, child = propose_random_mutation(incumbent, rng)
        else:
            record, child = propose_mutation(incumbent, rng, failures=structured_failures)
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

    # Selection happens on a SEPARATE validation episode slice. Keyed by
    # (episode_id, seed) so incumbent/candidate coverage can be compared
    # before any independent None-filtering -- filtering each side
    # separately (dropping missing results independently) can silently
    # compare unequal sample sets, which is not a valid comparison.
    val_keys = [(ep.episode_id, seed) for ep in val_episodes for seed in seeds]
    incumbent_val_raw = {}
    candidate_val_raw = {}
    for ep in val_episodes:
        for seed in seeds:
            key = (ep.episode_id, seed)
            incumbent_val_raw[key] = eval_fn(incumbent, ep, seed)
            candidate_val_raw[key] = eval_fn(best_candidate, ep, seed)

    coverage_mismatch = any(
        (incumbent_val_raw[k] is None) != (candidate_val_raw[k] is None) for k in val_keys
    )
    matched_keys = [k for k in val_keys
                    if incumbent_val_raw[k] is not None and candidate_val_raw[k] is not None]
    incumbent_val = [incumbent_val_raw[k] for k in matched_keys]
    candidate_val = [candidate_val_raw[k] for k in matched_keys]

    incumbent_protected = eval_fn(incumbent, protected_episode, seeds[0])
    candidate_protected = eval_fn(best_candidate, protected_episode, seeds[0])

    decision = evaluate_promotion(incumbent_val, candidate_val, incumbent_protected, candidate_protected,
                                   coverage_mismatch=coverage_mismatch)
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
        # Additive (P3 research-workspace consumers): full config dicts and
        # hashes for every policy touched in this episode, so a caller can
        # build a config diff / narrative without re-deriving policy_id
        # itself. Does not change any promotion/selection logic above.
        "policy_configs": {
            _policy_id(incumbent): incumbent.to_dict(),
            **{_policy_id(c): c.to_dict() for c in candidates},
        },
        "incumbent_policy_id": _policy_id(incumbent),
        "policy_id_by_version": {
            incumbent.version: _policy_id(incumbent),
            **{c.version: _policy_id(c) for c in candidates},
        },
    }
