from src.agent.sidecar_tools import (
    AssistantTool,
    SidecarToolRegistry,
    ToolBudget,
    ToolRequest,
    make_idea_sidecar_registry,
    register_hyde_retrieval,
    register_memory_search,
)


def test_idea_sidecar_tools_filter_future_failures_and_emit_provenance():
    registry = make_idea_sidecar_registry()
    result = registry.call(ToolRequest(
        request_id="r1", tool_name="idea_visible_failures",
        arguments={"decision_time": "2024-01-10", "failures": [
            {"failure_id": "past", "timestamp": "2024-01-09", "params": {"x": 1}},
            {"failure_id": "future", "timestamp": "2024-01-11", "params": {"x": 2}},
            {"failure_id": "unknown", "timestamp": "not-a-date", "params": {"x": 3}},
        ]}, capabilities=frozenset({"prior_results"}),
    ))
    assert result.ok
    assert [failure["failure_id"] for failure in result.data] == ["past"]
    assert result.provenance["source"] == "src.agent.idea_sidecar.filter_visible_failures"


def test_candidate_tool_is_advisory_and_returns_only_untried_grid_points():
    registry = make_idea_sidecar_registry()
    result = registry.call(ToolRequest(
        request_id="r2", tool_name="idea_candidate_options",
        arguments={"decision_time": "2024-01-10", "grid": [{"window": 2}, {"window": 4}],
                   "tried_grid_indices": [0], "failures": [], "prior_params": [{"window": 4}]},
        capabilities=frozenset({"prior_results"}),
    ))
    assert result.ok
    assert result.data == [{"source": "retrieved_prior", "grid_index": 1, "params": {"window": 4}}]
    assert "promote_candidate" not in result.provenance.get("capabilities_used", [])


def test_registry_isolates_contract_capability_and_handler_failures():
    registry = make_idea_sidecar_registry(budget=ToolBudget(max_calls=1))
    denied = registry.call(ToolRequest(
        "r1", "idea_visible_failures", {"decision_time": "2024-01-10", "failures": []}
    ))
    assert not denied.ok and denied.error == "missing_capability"
    bad = registry.call(ToolRequest(
        "r2", "idea_visible_failures", {"decision_time": 10, "failures": []},
        frozenset({"prior_results"}),
    ))
    assert not bad.ok and "field_type" in bad.error
    exhausted = registry.call(ToolRequest(
        "r3", "idea_visible_failures", {"decision_time": "2024-01-10", "failures": []},
        frozenset({"prior_results"}),
    ))
    assert exhausted.ok
    limited = registry.call(ToolRequest(
        "r4", "idea_visible_failures", {"decision_time": "2024-01-10", "failures": []},
        frozenset({"prior_results"}),
    ))
    assert not limited.ok and limited.error == "call_budget_exceeded"


def test_registry_rejects_holdout_and_promotion_tools():
    registry = SidecarToolRegistry()
    tool = AssistantTool("unsafe", "unsafe", {"type": "object"}, {"type": "object"},
                         frozenset({"holdout_data"}), lambda _: {}, "test")
    try:
        registry.register(tool)
    except ValueError as exc:
        assert "holdout" in str(exc) or "promotion" in str(exc)
    else:
        raise AssertionError("holdout access must be rejected")


def test_hyde_retrieval_registers_as_bounded_assistant_tool():
    from src.agent.hyde_retrieval import Evidence, HydeRetrievalTool, SyntheticHypothesis

    class Generator:
        def generate(self, request):
            return [SyntheticHypothesis("momentum breakout", "momentum", {"fast_window": 10})]

    registry = SidecarToolRegistry()
    register_hyde_retrieval(registry, HydeRetrievalTool(
        Generator(), [Evidence("e1", "momentum breakout", "fixture", "2020-01-01",
                               "2020-01-02", "momentum", {"fast_window": 10}, 0.8)],
    ))
    result = registry.call(ToolRequest(
        "hyde", "retrieve_hypothesis_evidence",
        {"regime": "bull", "strategy_types": ["momentum"], "as_of": "2021-01-01"},
        frozenset({"prior_results"}),
    ))
    assert result.ok
    assert result.data[0]["evaluation_status"] == "unevaluated"
    assert result.data[0]["evidence"][0]["evidence_id"] == "e1"
    assert result.provenance["source"] == "src.agent.hyde_retrieval.HydeRetrievalTool"


def test_memory_search_registers_with_cutoffs_and_read_only_result():
    from src.agent.memory_tool import MemorySearchTool

    class ReadOnlyMemory:
        calls = []

        def graph_search(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return []

    memory = ReadOnlyMemory()
    registry = SidecarToolRegistry()
    register_memory_search(registry, MemorySearchTool(memory))
    result = registry.call(ToolRequest(
        "mem", "search_memory_graph",
        {"query": "momentum failure", "as_of": "2022-01-01", "knowledge_as_of": "2022-02-01",
         "strategy_type": "momentum", "top_k": 3},
        frozenset({"prior_results"}),
    ))
    assert result.ok and result.data["read_only"] is True
    assert result.data["promoted_to_belief"] is False
    assert memory.calls[0][1]["knowledge_as_of"] == "2022-02-01"
    assert result.provenance["source"] == "src.agent.memory_tool.MemorySearchTool"
