from src.agent.memory_tool import MemorySearchTool
from src.memory.models import Note, Trial
from src.memory.service import MemoryService


def _trial(trial_id, *, created="2021-01-01T00:00:00+00:00", end="2020-12-31",
           asset="SPY", regime="bull"):
    return Trial(
        trial_id=trial_id, run_id=f"run-{trial_id}", asset=asset,
        strategy_type="momentum", params={"fast": 5}, created_at=created,
        data_end=end, regime_label=regime, outcome="accepted", is_sharpe=1.1,
    )


def test_memory_tool_returns_scoped_provenance_and_explanation(tmp_path):
    service = MemoryService(db_path=str(tmp_path / "memory.db"), mode="read_write")
    service.record_trial(_trial("visible"))
    service.record_trial(_trial("future", end="2022-01-01"))
    tool = MemorySearchTool(service)

    result = tool(query="accepted", as_of="2021-01-01", strategy_type="momentum",
                  asset="SPY", regime_label="bull", top_k=3)

    assert result["read_only"] is True
    assert result["promoted_to_belief"] is False
    assert result["cutoffs"]["market_as_of"] == "2021-01-01"
    assert result["results"]
    provenance = result["results"][0]["evidence"][0]
    assert provenance["record_id"] == "visible"
    assert provenance["known_at"].startswith("2021-01-01")
    assert "query-match base" in result["results"][0]["score_explanation"]


def test_memory_tool_applies_knowledge_cutoff_and_top_k_bound(tmp_path):
    service = MemoryService(db_path=str(tmp_path / "memory.db"), mode="read_write")
    service.record_trial(_trial("recorded-late", created="2022-01-01T00:00:00+00:00"))
    tool = MemorySearchTool(service)

    hidden = tool(query="accepted", as_of="2021-01-01", knowledge_as_of="2021-12-31")
    assert hidden["results"] == []
    assert hidden["cutoffs"]["knowledge_as_of"] == "2021-12-31"
    try:
        tool(query="accepted", top_k=1000)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("oversized top_k should be rejected")


def test_memory_tool_requires_and_validates_cutoffs(tmp_path):
    tool = MemorySearchTool(MemoryService(db_path=str(tmp_path / "memory.db"), mode="read"))
    for args, message in [
        ({"query": "accepted"}, "as_of is required"),
        ({"query": "accepted", "as_of": "yesterday"}, "valid ISO"),
        ({"query": "accepted", "as_of": "2021-01-01", "knowledge_as_of": "2021-13-01"},
         "valid ISO"),
    ]:
        try:
            tool(**args)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected cutoff validation error containing {message!r}")


def test_memory_tool_is_read_only_in_read_mode(tmp_path):
    service = MemoryService(db_path=str(tmp_path / "memory.db"), mode="read_write")
    service.record_note(Note(note_id="note-1", kind="memo", body="", created_at="2021-01-01",
                             data_end="2020-12-31", strategy_type="momentum", asset="SPY",
                             meta={"claim": {"predicate": "robust", "value": "yes",
                                             "subject": "SPY"}}))
    read_service = MemoryService(db_path=str(tmp_path / "memory.db"), mode="read")
    tool = MemorySearchTool(read_service)
    result = tool(query="robust", as_of="2021-01-01")
    assert result["results"][0]["evidence"][0]["record_type"] == "note"
    assert read_service.can_write is False
