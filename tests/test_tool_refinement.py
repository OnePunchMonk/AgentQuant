import pytest

from src.agent.sidecar_tools import ToolRequest
from src.agent.tool_refinement import VerificationResult, refine_tool_request


def test_refinement_repairs_arguments_and_exits_when_verified():
    initial = ToolRequest("r1", "lookup", {"query": "  alpha  "}, frozenset({"prior"}))
    seen = []

    def verify(request):
        seen.append(dict(request.arguments))
        return VerificationResult(request.arguments["query"] == "alpha", "clean query")

    result = refine_tool_request(
        initial, ["lookup", "fallback"], verify,
        repair=lambda req, verdict: {"arguments": {"query": req.arguments["query"].strip()}},
        on_step=lambda step: None,
    )
    assert result.verified and result.early_exit
    assert result.steps == 2
    assert result.request.arguments == {"query": "alpha"}
    assert [step.repaired for step in result.trace] == [True, False]
    assert seen == [{"query": "  alpha  "}, {"query": "alpha"}]
    assert initial.arguments["query"] == "  alpha  "


def test_refinement_is_bounded_and_reports_exhaustion():
    result = refine_tool_request(
        ToolRequest("r2", "lookup", {}, frozenset()), ["lookup"],
        lambda req: False, max_steps=2,
        repair=lambda req, verdict: {"arguments": {"n": len(req.arguments) + 1}},
    )
    assert not result.verified and not result.early_exit
    assert result.steps == 2
    assert result.request.arguments == {"n": 1}


@pytest.mark.parametrize("repair, message", [
    (lambda req, verdict: {"request_id": "other"}, "request_id"),
    (lambda req, verdict: {"tool_name": "unlisted"}, "frozen candidate"),
    (lambda req, verdict: {"capabilities": {"prior", "admin"}}, "add capabilities"),
])
def test_repair_cannot_expand_frozen_scope(repair, message):
    request = ToolRequest("r3", "lookup", {}, frozenset({"prior"}))
    with pytest.raises(ValueError, match=message):
        refine_tool_request(request, ["lookup", "fallback"], lambda req: False,
                            repair=repair, max_steps=2)


def test_initial_request_must_be_in_candidate_set():
    with pytest.raises(ValueError, match="initial tool_name"):
        refine_tool_request(ToolRequest("r4", "other", {}, frozenset()),
                            ["lookup"], lambda req: True)


def test_max_steps_validation():
    with pytest.raises(ValueError, match="positive integer"):
        refine_tool_request(ToolRequest("r5", "lookup", {}, frozenset()),
                            ["lookup"], lambda req: True, max_steps=0)
