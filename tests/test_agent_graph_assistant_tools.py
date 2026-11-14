from types import SimpleNamespace

from src.agent.agent_graph import _run_assistant_sidecar_context


def test_assistant_sidecar_lookup_is_opt_in(monkeypatch):
    monkeypatch.delenv("AGENTQUANT_ASSISTANT_TOOLS", raising=False)
    state = {"as_of": "2024-01-31", "run_id": "r1", "asset": "SPY"}
    assert _run_assistant_sidecar_context(
        state, "momentum", SimpleNamespace(regime_label="Bull"), 1
    ) == ""
    assert "assistant_tool_results" not in state


def test_assistant_sidecar_lookup_runs_bounded_memory_tool(monkeypatch):
    monkeypatch.setenv("AGENTQUANT_ASSISTANT_TOOLS", "1")
    calls = []

    class FakeMemoryService:
        def graph_search(self, query, **kwargs):
            calls.append((query, kwargs))
            return []

    monkeypatch.setattr("src.agent.memory_tool.MemoryService", FakeMemoryService)
    state = {"as_of": "2024-01-31", "run_id": "r1", "asset": "SPY"}
    context = SimpleNamespace(regime_label="Bull")
    prompt_context = _run_assistant_sidecar_context(state, "momentum", context, 2)
    assert "Bounded read-only sidecar context" in prompt_context
    # The registry runs handlers in an isolated child, so observe the call
    # through its structured result rather than shared mutable test state.
    assert len(calls) == 0
    payload = state["assistant_tool_results"][0]
    assert payload["status"] == "ok"
    assert payload["selected_tools"] == ["search_memory_graph"]
    assert payload["authority"] == {
        "can_access_holdout": False, "can_promote": False, "can_evaluate": False
    }
    assert payload["results"][0]["provenance"]["source"] == "src.agent.memory_tool.MemorySearchTool"
    assert payload["results"][0]["data"]["cutoffs"]["market_as_of"] == "2024-01-31"
    assert {item["intent"] for item in state["assistant_tool_results"] if "intent" in item} >= {
        "memory_lookup", "candidate_options", "hypothesis_retrieval"
    }
