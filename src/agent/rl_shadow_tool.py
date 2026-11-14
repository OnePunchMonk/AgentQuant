"""Small, bounded assistant-facing adapter for the non-learning shadow policy.

This module only advises over an already constructed observation and candidate
menu. It does not build observations from market data, evaluate candidates,
compute rewards, touch holdout data, or promote a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from src.agent import idea_sidecar as sidecar

DEFAULT_MAX_CANDIDATES = 64
MAX_CANDIDATES = 256


@dataclass(frozen=True)
class Advisory:
    """Serializable recommendation and concise provenance for a single call."""

    schema_version: str
    policy_name: str
    policy_version: str
    policy_hash: str
    considered_candidates: int
    source_allocation: Dict[str, int]
    action: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
            "considered_candidates": self.considered_candidates,
            "source_allocation": dict(self.source_allocation),
            "action": dict(self.action),
        }


def advise(
    observation: sidecar.Observation,
    candidates: Sequence[sidecar.Candidate],
    policy: sidecar.Policy | None = None,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Advisory:
    """Return one validated policy action from a bounded candidate prefix.

    Callers supply a dev-safe observation and canonical candidates. The adapter
    imposes a hard candidate cap and relies on ``validate_action`` to reject
    any policy action that is outside the supplied canonical menu.
    """
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool):
        raise ValueError("max_candidates must be an integer")
    if not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError(f"max_candidates must be between 1 and {MAX_CANDIDATES}")
    if not candidates:
        raise ValueError("candidates must not be empty")

    selected_policy = policy or sidecar.HeuristicPolicy()
    bounded = list(candidates[:max_candidates])
    action = selected_policy.act(observation, bounded)
    by_index = {candidate.grid_index: candidate for candidate in bounded}
    if not by_index:
        raise ValueError("candidates must not be empty")
    # Candidate IDs are canonical grid indices and can be sparse after capping.
    # Populate a temporary index-addressable grid, then use the shared validator.
    grid: list[Dict[str, Any]] = [{} for _ in range(max(by_index) + 1)]
    for candidate in bounded:
        grid[candidate.grid_index] = dict(candidate.params)
    sidecar.validate_action(action, grid, bounded)

    allocation: Dict[str, int] = {}
    for candidate in bounded:
        allocation[candidate.source.value] = allocation.get(candidate.source.value, 0) + 1
    return Advisory(
        schema_version=sidecar.SCHEMA_VERSION,
        policy_name=selected_policy.name,
        policy_version=selected_policy.version,
        policy_hash=sidecar.policy_hash(selected_policy),
        considered_candidates=len(bounded),
        source_allocation=allocation,
        action=action.to_dict(),
    )
