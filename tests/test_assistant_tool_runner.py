from src.agent.assistant_tool_runner import AssistantToolRunner, AssistantToolRunRequest
from src.agent.sidecar_tools import ToolBudget, make_idea_sidecar_registry, register_hyde_retrieval


def test_runner_discovers_selects_and_stamps_decision_time():
    registry = make_idea_sidecar_registry()
    runner = AssistantToolRunner(registry)
    result = runner.run(AssistantToolRunRequest(
        "run-1", "failure_lookup", "2024-01-10",
        {"failures": [
            {"failure_id": "old", "timestamp": "2024-01-09"},
            {"failure_id": "future", "timestamp": "2024-01-11"},
        ]},
    ))
    assert result.status == "ok"
    assert result.selected_tools == ("idea_visible_failures",)
    assert [f["failure_id"] for f in result.results[0].data] == ["old"]
    assert result.results[0].request_id == "run-1:1"
    assert {tool["name"] for tool in result.discovery} == {
        "idea_visible_failures", "idea_candidate_options"
    }
    assert result.to_dict()["authority"] == {
        "can_access_holdout": False, "can_promote": False, "can_evaluate": False
    }


def test_runner_records_isolated_tool_error_and_respects_per_run_limit():
    registry = make_idea_sidecar_registry(budget=ToolBudget(max_calls=1))
    runner = AssistantToolRunner(registry, {"two": (
        "idea_visible_failures", "idea_candidate_options")})
    result = runner.run(AssistantToolRunRequest(
        "run-2", "two", "2024-01-10",
        {"failures": []},
        max_tool_calls=2,
    ))
    assert result.status == "partial"
    assert len(result.attempts) == 2
    assert result.attempts[0].ok
    assert not result.attempts[1].ok
    assert result.attempts[1].error == "call_budget_exceeded"


def test_unknown_intent_does_not_invoke_tools():
    result = AssistantToolRunner(make_idea_sidecar_registry()).run(
        AssistantToolRunRequest("run-3", "unknown", "2024-01-10"))
    assert result.status == "no_matching_tool"
    assert not result.attempts


def test_reused_runner_has_independent_run_budgets():
    runner = AssistantToolRunner(make_idea_sidecar_registry(budget=ToolBudget(max_calls=1)))
    for index in range(3):
        result = runner.run(AssistantToolRunRequest(
            f"reuse-{index}", "failure_lookup", "2024-01-10", {"failures": []}))
        assert result.status == "ok"


def test_concurrent_runs_do_not_consume_each_others_budgets():
    from concurrent.futures import ThreadPoolExecutor

    runner = AssistantToolRunner(make_idea_sidecar_registry(budget=ToolBudget(max_calls=1)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: runner.run(AssistantToolRunRequest(
            f"parallel-{index}", "failure_lookup", "2024-01-10", {"failures": []})), range(8)))
    assert all(result.status == "ok" for result in results)


def test_runner_rejects_conflicting_decision_time():
    runner = AssistantToolRunner(make_idea_sidecar_registry())
    result = runner.run(AssistantToolRunRequest(
        "run-4", "failure_lookup", "2024-01-10",
        {"decision_time": "2024-01-11", "failures": []},
    ))
    assert result.status == "error"
    assert "match" in result.attempts[0].error


def test_hypothesis_tool_cannot_raise_knowledge_cutoff_past_decision_time():
    class Retriever:
        def run(self, request):
            raise AssertionError("future knowledge must be rejected before invocation")

    registry = make_idea_sidecar_registry()
    register_hyde_retrieval(registry, Retriever())
    result = AssistantToolRunner(registry).run(AssistantToolRunRequest(
        "run-5", "hypothesis_retrieval", "2024-01-10",
        {"regime": "trend", "strategy_types": ["momentum"],
         "knowledge_as_of": "2024-01-11"},
    ))
    assert result.status == "error"
    assert result.attempts[0].error == "request_rejected: knowledge_as_of cannot exceed decision_time"


def test_non_integer_tool_limit_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="non-negative integer"):
        AssistantToolRunRequest("run-6", "failure_lookup", "2024-01-10", max_tool_calls=1.5)


def test_selector_ranks_relevant_registered_tool_for_unmapped_intent():
    registry = make_idea_sidecar_registry()
    runner = AssistantToolRunner(registry)
    result = runner.run(AssistantToolRunRequest(
        "run-7", "candidate options", "2024-01-10",
        {"grid": [], "tried_grid_indices": [], "failures": [], "prior_params": []},
    ))
    assert result.selected_tools == ("idea_candidate_options",)
    assert result.selection_trace
    assert result.selection_trace[0]["score"] > 0
    assert "selection_trace" in result.to_dict()


def test_selector_abstains_and_injected_scorer_failure_falls_back():
    registry = make_idea_sidecar_registry()
    unknown = AssistantToolRunner(registry).run(AssistantToolRunRequest(
        "run-8", "weather forecast", "2024-01-10"))
    assert unknown.status == "no_matching_tool"
    assert not unknown.selected_tools

    def broken_scorer(request, tool):
        raise RuntimeError("scorer unavailable")

    fallback = AssistantToolRunner(registry, scorer=broken_scorer,
                                   scorer_version="local-test-v1").run(
        AssistantToolRunRequest("run-9", "failure_lookup", "2024-01-10",
                                {"failures": []}))
    assert fallback.selected_tools == ("idea_visible_failures",)
    assert "scorer_failure_fallback" in fallback.selection_trace[0]
