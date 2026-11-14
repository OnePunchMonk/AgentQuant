from dataclasses import FrozenInstanceError

import pytest

from src.agent.capabilities import (
    AllocationRequest,
    CapabilityRegistry,
    Observation,
    ProposalRequest,
    RunBudget,
    RunContext,
    RunVisibility,
)


def test_context_carries_budget_visibility_and_immutable_metadata():
    metadata = {"epoch": 2}
    context = RunContext(
        "run-1", "momentum", RunBudget(max_proposals=3, max_backtests=7),
        RunVisibility(allow_holdout_data=False), metadata,
    )
    metadata["epoch"] = 9

    assert context.budget.max_backtests == 7
    assert context.visibility.allow_holdout_data is False
    assert context.metadata["epoch"] == 2
    with pytest.raises(TypeError):
        context.metadata["epoch"] = 4
    with pytest.raises(FrozenInstanceError):
        context.run_id = "changed"


def test_requests_enforce_budget_limits():
    context = RunContext("r", "momentum", RunBudget(max_proposals=2, max_backtests=4))
    ProposalRequest(context, 2)
    AllocationRequest(context, [], 4)
    with pytest.raises(ValueError):
        ProposalRequest(context, 3)
    with pytest.raises(ValueError):
        AllocationRequest(context, [], 5)


def test_registry_is_separated_by_capability_and_rejects_duplicate_names():
    class Provider:
        def propose(self, request):
            return []

    registry = CapabilityRegistry()
    provider = Provider()
    assert registry.register_proposal_provider("grid", provider) is provider
    assert registry.proposal_provider("grid") is provider
    with pytest.raises(ValueError, match="already registered"):
        registry.register_proposal_provider("grid", Provider())
    with pytest.raises(KeyError, match="unknown capability"):
        registry.allocator("grid")


def test_observation_payload_is_immutable_snapshot():
    data = {"count": 1}
    observation = Observation("r", "proposals_created", data)
    data["count"] = 5

    assert observation.payload["count"] == 1
    with pytest.raises(TypeError):
        observation.payload["count"] = 2
