"""
Episode Report — P3 research-workspace narrative/inspection layer.

Turns the raw dict produced by
`src.agent.policy_mutation.run_bounded_self_improvement` (plus, optionally,
a `src.agent.run_manifest.RunManifest`) into:

  1. A structured, chronological narrative of one P2 outer-loop episode:
     hypothesis -> evidence-at-the-time -> experiment -> decision ->
     policy change -> fresh result.
  2. A query layer over attempted (including rejected) candidates.
  3. A policy config diff between any two known HarnessConfig dicts.

This module does not run anything itself -- it is a read-only reporting
layer over already-persisted/returned P0/P1/P2 artifacts. It never
fabricates a value: any field it cannot find in the inputs is reported as
explicitly unavailable (`None` with a companion note), never silently
defaulted or invented.

NOTE ON SCOPE: this module (and research_memo.py) covers *retrospective*
research episodes only. Prospective/live paper-trading research is
explicitly deferred -- see docs/RESEARCH_MEMO.md and issue #28, which says
to add that only once the experiment contract here is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

UNAVAILABLE = "unavailable: not present in supplied episode/manifest data"


@dataclass
class CandidateRecord:
    """One attempted hypothesis (mutation) within an episode's outer loop,
    win or lose."""

    mutation_id: str
    parent_policy_id: str
    diagnosis: str
    expected_benefit: str
    patch: Dict[str, Any]
    dev_scores: List[float]
    dev_mean: Optional[float]
    is_best_on_dev: bool
    was_selected_for_promotion_check: bool
    promoted: Optional[bool]
    rejection_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mutation_id": self.mutation_id,
            "parent_policy_id": self.parent_policy_id,
            "diagnosis": self.diagnosis,
            "expected_benefit": self.expected_benefit,
            "patch": self.patch,
            "dev_scores": self.dev_scores,
            "dev_mean": self.dev_mean,
            "is_best_on_dev": self.is_best_on_dev,
            "was_selected_for_promotion_check": self.was_selected_for_promotion_check,
            "promoted": self.promoted,
            "rejection_reason": self.rejection_reason,
        }


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def list_candidates(episode_result: Dict[str, Any]) -> List[CandidateRecord]:
    """List every attempted mutation candidate for one episode, including
    ones that never made it past dev-phase selection, with an explicit
    rejection reason for each candidate that did not end up promoted.

    Reuses the `mutation_records` / `dev_scores` / `promotion_decision` /
    `best_candidate_version` / `selected_policy_version` fields already
    persisted by `run_bounded_self_improvement`.
    """
    mutation_records = episode_result.get("mutation_records", [])
    dev_scores = episode_result.get("dev_scores", {})
    best_version = episode_result.get("best_candidate_version")
    selected_version = episode_result.get("selected_policy_version")
    promotion = episode_result.get("promotion_decision", {})
    promoted = bool(promotion.get("promote"))

    records: List[CandidateRecord] = []
    for rec in mutation_records:
        mutation_id = rec.get("mutation_id")
        scores = dev_scores.get(mutation_id, [])
        is_best = mutation_id == best_version
        this_promoted: Optional[bool]
        reason: Optional[str]

        if not is_best:
            this_promoted = False
            reason = (
                "not selected as best-on-dev candidate "
                f"(dev_mean={_mean(scores)!r} vs best={best_version!r})"
            )
        else:
            # This was the candidate carried forward to the validation-slice
            # promotion check.
            if promoted and selected_version == mutation_id:
                this_promoted = True
                reason = None
            else:
                this_promoted = False
                reason = (
                    f"best-on-dev but failed promotion check: {promotion.get('reason')}"
                )

        records.append(
            CandidateRecord(
                mutation_id=mutation_id,
                parent_policy_id=rec.get("parent_policy_id"),
                diagnosis=rec.get("diagnosis", UNAVAILABLE),
                expected_benefit=rec.get("expected_benefit", UNAVAILABLE),
                patch=rec.get("patch", {}),
                dev_scores=scores,
                dev_mean=_mean(scores),
                is_best_on_dev=is_best,
                was_selected_for_promotion_check=is_best,
                promoted=this_promoted,
                rejection_reason=reason,
            )
        )
    return records


def candidates_report(episode_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Same as `list_candidates` but as plain dicts, for easy printing/export."""
    return [c.to_dict() for c in list_candidates(episode_result)]


def compare_policies(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> Dict[str, Any]:
    """Structured diff between two HarnessConfig `.to_dict()` payloads.

    Returns which top-level fields changed vs. stayed the same, plus a
    metrics comparison if metrics dicts are embedded under the
    conventional keys ("holdout_metrics" / "metrics") -- otherwise the
    metrics section explicitly says metrics were not supplied.
    """
    keys = sorted(set(config_a.keys()) | set(config_b.keys()))
    changed: Dict[str, Dict[str, Any]] = {}
    unchanged: Dict[str, Any] = {}
    for k in keys:
        va = config_a.get(k, UNAVAILABLE)
        vb = config_b.get(k, UNAVAILABLE)
        if va != vb:
            changed[k] = {"a": va, "b": vb}
        else:
            unchanged[k] = va

    metrics_a = config_a.get("holdout_metrics") or config_a.get("metrics")
    metrics_b = config_b.get("holdout_metrics") or config_b.get("metrics")
    if metrics_a is not None and metrics_b is not None:
        metric_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
        metrics_diff = {
            k: {"a": metrics_a.get(k), "b": metrics_b.get(k)} for k in metric_keys
        }
    else:
        metrics_diff = UNAVAILABLE

    return {
        "changed_fields": changed,
        "unchanged_fields": unchanged,
        "metrics_diff": metrics_diff,
    }


@dataclass
class EpisodeNarrative:
    """The full hypothesis -> evidence -> experiment -> decision ->
    policy-change -> fresh-result story for one P2 outer-loop episode."""

    hypothesis: Dict[str, Any]
    evidence_at_the_time: Dict[str, Any]
    experiment: Dict[str, Any]
    decision: Dict[str, Any]
    policy_change: Dict[str, Any]
    fresh_result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "evidence_at_the_time": self.evidence_at_the_time,
            "experiment": self.experiment,
            "decision": self.decision,
            "policy_change": self.policy_change,
            "fresh_result": self.fresh_result,
        }


def build_episode_narrative(
    episode_result: Dict[str, Any],
    *,
    run_manifest: Optional[Dict[str, Any]] = None,
    memory_entries_visible: Optional[List[Dict[str, Any]]] = None,
    used_final_holdout: bool = True,
) -> EpisodeNarrative:
    """Assemble the chronological narrative for one episode from the dict
    returned by `run_bounded_self_improvement`, an optional run manifest
    dict (`RunManifest.to_dict()`), and an optional explicit list of memory
    rows that were visible at decision time (already filtered through
    `src.agent.search_arms.filter_visible_memory` by the caller, so no
    future-dated leakage).

    Every field that cannot be determined from the supplied inputs is set
    to the `UNAVAILABLE` sentinel string rather than fabricated.
    """
    mutation_records = episode_result.get("mutation_records", [])
    best_version = episode_result.get("best_candidate_version")
    best_record = next((r for r in mutation_records if r.get("mutation_id") == best_version), None)

    hypothesis = {
        "mutation_id": best_version or UNAVAILABLE,
        "diagnosis": (best_record or {}).get("diagnosis", UNAVAILABLE),
        "expected_benefit": (best_record or {}).get("expected_benefit", UNAVAILABLE),
        "patch": (best_record or {}).get("patch", UNAVAILABLE),
        "parent_policy_id": (best_record or {}).get("parent_policy_id", UNAVAILABLE),
    }

    evidence_at_the_time = {
        "memory_entries_visible": memory_entries_visible
        if memory_entries_visible is not None
        else UNAVAILABLE,
        "n_memory_entries_visible": (
            len(memory_entries_visible) if memory_entries_visible is not None else UNAVAILABLE
        ),
        "note": (
            "Visibility is enforced by src.agent.search_arms.filter_visible_memory: "
            "only entries from strictly earlier episodes than the current one are "
            "included, so this list is evidence of no future-dated-memory leakage."
        ),
    }

    experiment = {
        "dev_scores_by_candidate": episode_result.get("dev_scores", UNAVAILABLE),
        "n_candidates_evaluated": len(mutation_records),
        "validation_episode_scores": {
            "incumbent": episode_result.get("incumbent_val_scores", UNAVAILABLE),
            "candidate": episode_result.get("candidate_val_scores", UNAVAILABLE),
        },
        "protected_episode_id": episode_result.get("protected_episode_id", UNAVAILABLE),
        "protected_scores": {
            "incumbent": episode_result.get("incumbent_protected_score", UNAVAILABLE),
            "candidate": episode_result.get("candidate_protected_score", UNAVAILABLE),
        },
        "run_manifest": run_manifest if run_manifest is not None else UNAVAILABLE,
    }

    decision = dict(episode_result.get("promotion_decision", {})) or UNAVAILABLE
    if isinstance(decision, dict):
        decision = {
            "promote": decision.get("promote", UNAVAILABLE),
            "reason": decision.get("reason", UNAVAILABLE),
            "candidate_mean": decision.get("candidate_mean", UNAVAILABLE),
            "incumbent_mean": decision.get("incumbent_mean", UNAVAILABLE),
            "protected_delta": decision.get("protected_delta", UNAVAILABLE),
        }

    incumbent_policy_id = episode_result.get("incumbent_policy_id", UNAVAILABLE)
    selected_version = episode_result.get("selected_policy_version")
    policy_id_by_version = episode_result.get("policy_id_by_version", {})
    selected_policy_id = policy_id_by_version.get(selected_version, UNAVAILABLE)
    promoted = bool(episode_result.get("promotion_decision", {}).get("promote"))

    policy_change = {
        "promoted": promoted,
        "old_policy_hash": incumbent_policy_id,
        "new_policy_hash": selected_policy_id if promoted else incumbent_policy_id,
        "note": (
            "Promoted: policy changed from old_policy_hash to new_policy_hash."
            if promoted
            else "Not promoted: incumbent policy retained (old_policy_hash == new_policy_hash)."
        ),
    }

    final_scores = episode_result.get("final_holdout_scores")
    final_mean = episode_result.get("final_holdout_mean")
    if used_final_holdout:
        fresh_result = {
            "graded_on_final_holdout": True,
            "final_holdout_scores": final_scores if final_scores is not None else UNAVAILABLE,
            "final_holdout_mean": final_mean if final_mean is not None else UNAVAILABLE,
        }
    else:
        fresh_result = {
            "graded_on_final_holdout": False,
            "note": (
                "This episode did not use the sealed final-holdout episodes for grading; "
                "no fresh out-of-sample result is claimed here. Treat any dev/validation "
                "scores above as selection-phase evidence only, not a graded result."
            ),
        }

    return EpisodeNarrative(
        hypothesis=hypothesis,
        evidence_at_the_time=evidence_at_the_time,
        experiment=experiment,
        decision=decision,
        policy_change=policy_change,
        fresh_result=fresh_result,
    )
