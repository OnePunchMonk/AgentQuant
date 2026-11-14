"""Tests for the bounded assistant-facing advisory adapter."""

import pytest

from src.agent import idea_sidecar as sidecar
from src.agent import rl_shadow_tool


def _inputs():
    candidates = [
        sidecar.Candidate(sidecar.ProposalSource.RETRIEVED_PRIOR, 0, {"x": 1}),
        sidecar.Candidate(sidecar.ProposalSource.UNEXPLORED_REGION, 1, {"x": 2}),
    ]
    obs = sidecar.Observation(
        schema_version=sidecar.SCHEMA_VERSION,
        episode_id="fixture",
        decision_time="2024-01-01",
        step=0,
        asset="SIM",
        regime="unknown",
        strategy_family="fixture",
        grid_size=2,
        tried_grid_indices=(),
        visible_failures=(),
        attempted=(),
        remaining_proposals=1,
        remaining_tool_calls=0,
    )
    return obs, candidates


def test_advisory_includes_hash_source_counts_and_propensity():
    obs, candidates = _inputs()
    result = rl_shadow_tool.advise(obs, candidates)
    payload = result.to_dict()
    assert payload["policy_hash"] == sidecar.policy_hash(sidecar.HeuristicPolicy())
    assert payload["considered_candidates"] == 2
    assert payload["source_allocation"] == {"retrieved_prior": 1, "unexplored_region": 1}
    assert payload["action"]["source"] == "retrieved_prior"
    assert payload["action"]["propensity"] == 1.0


def test_candidate_cap_applies_before_policy_call():
    obs, candidates = _inputs()
    result = rl_shadow_tool.advise(obs, candidates, max_candidates=1)
    assert result.considered_candidates == 1
    assert result.action["grid_index"] == 0


@pytest.mark.parametrize("cap", [0, -1, 257, 1.5, True])
def test_invalid_cap_rejected(cap):
    obs, candidates = _inputs()
    with pytest.raises(ValueError):
        rl_shadow_tool.advise(obs, candidates, max_candidates=cap)


def test_policy_action_outside_bounded_menu_is_rejected():
    obs, candidates = _inputs()

    class RoguePolicy:
        name, version = "rogue", "1"

        def config(self):
            return {}

        def act(self, observation, menu):
            return sidecar.Action(sidecar.ProposalSource.UNEXPLORED_REGION, 1, {"x": 2}, 1.0)

    with pytest.raises(sidecar.ActionOutOfSpace):
        rl_shadow_tool.advise(obs, candidates, RoguePolicy(), max_candidates=1)


def test_empty_candidate_menu_rejected():
    obs, _ = _inputs()
    with pytest.raises(ValueError, match="must not be empty"):
        rl_shadow_tool.advise(obs, [])
