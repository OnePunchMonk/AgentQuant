from src.memory.graph import EvidenceGraph
from src.memory.models import MemoryQuery, Note, Trial
from src.memory.service import MemoryService


def _trial(trial_id, *, known="2021-01-01T00:00:00+00:00", end="2020-12-31",
           outcome="accepted", is_sharpe=1.0, oos_sharpe=None, oos_end=None):
    return Trial(trial_id=trial_id, run_id=f"run-{trial_id}", asset="SPY",
                 strategy_type="momentum", params={"fast": 5}, created_at=known,
                 data_end=end, regime_label="bull", outcome=outcome,
                 is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, oos_end=oos_end,
                 oos_source="holdout" if oos_sharpe is not None else None)


def test_oos_graph_edge_does_not_leak_before_oos_close():
    graph = EvidenceGraph.rebuild([_trial("a", oos_sharpe=1.2, oos_end="2022-06-30")])
    early = graph.retrieve(as_of="2022-06-29", query="oos_sharpe")
    late = graph.retrieve(as_of="2022-06-30", query="oos_sharpe")
    assert early == []
    assert len(late) == 1
    assert late[0].evidence[0].visible_at == "2022-06-30"


def test_contradictory_outcomes_are_preserved_and_linked():
    graph = EvidenceGraph.rebuild([
        _trial("good", outcome="accepted"),
        _trial("bad", outcome="rejected"),
    ])
    claims = graph.retrieve(as_of="2022-01-01", query="outcome")
    assert {h.node.attributes["value"] for h in claims} == {"accepted", "rejected"}
    assert all(h.contradictions for h in claims)
    assert {e.source_record_id for h in claims for e in h.evidence} == {"good", "bad"}


def test_note_claim_keeps_note_provenance():
    note = Note(note_id="n1", kind="memo", body="", created_at="2020-01-02T00:00:00+00:00",
                data_end="2020-01-01", strategy_type="momentum", asset="SPY",
                meta={"claim": {"predicate": "robust", "value": "yes", "subject": "SPY"}})
    hits = EvidenceGraph.rebuild([], [note]).retrieve(as_of="2021-01-01", query="robust")
    assert len(hits) == 1
    assert hits[0].evidence[0].source_record_id == "n1"
    assert hits[0].evidence[0].source_record_type == "note"


def test_rebuild_and_ranking_are_deterministic():
    trials = [_trial("z"), _trial("a")]
    first = EvidenceGraph.rebuild(trials)
    second = EvidenceGraph.rebuild(list(reversed(trials)))
    assert first.nodes == second.nodes
    assert first.edges == second.edges
    assert first.retrieve(as_of="2022-01-01") == second.retrieve(as_of="2022-01-01")


def test_future_created_record_is_hidden_even_if_data_window_is_old():
    graph = EvidenceGraph.rebuild([_trial("future", known="2024-01-01T00:00:00+00:00")])
    assert graph.retrieve(market_as_of="2022-01-01", knowledge_as_of="2022-01-01",
                          query="accepted") == []


def test_retrospective_replay_ignores_materialization_wall_clock():
    graph = EvidenceGraph.rebuild([_trial("prior", known="2026-11-14T00:00:00+00:00",
                                          end="2020-12-31")])
    hits = graph.retrieve(market_as_of="2021-01-01", query="accepted")
    assert len(hits) == 1
    assert hits[0].evidence[0].known_at.startswith("2026-")
    assert graph.retrieve(market_as_of="2021-01-01", knowledge_as_of="2021-01-01",
                          query="accepted") == []


def test_future_market_window_is_excluded_from_historical_replay():
    graph = EvidenceGraph.rebuild([_trial("later", end="2022-01-01")])
    assert graph.retrieve(market_as_of="2021-12-31", query="accepted") == []


def test_regime_filter_uses_source_record_association():
    bull = _trial("bull", outcome="accepted")
    bear = _trial("bear", outcome="rejected")
    bear.regime_label = "bear"
    graph = EvidenceGraph.rebuild([bull, bear])
    hits = graph.retrieve(market_as_of="2022-01-01", regime_label="bull", query="outcome")
    assert {hit.node.attributes["value"] for hit in hits} == {"accepted"}


def test_unknown_required_market_clock_fails_closed():
    graph = EvidenceGraph.rebuild([_trial("undated", end=None)])
    assert graph.retrieve(market_as_of="2022-01-01", query="accepted") == []


def test_service_graph_search_respects_market_and_knowledge_cutoffs(tmp_path):
    service = MemoryService(db_path=str(tmp_path / "memory.sqlite"), mode="read_write")
    service.record_trial(_trial("past", known="2026-01-01T00:00:00+00:00", end="2020-12-31",
                                oos_sharpe=1.2, oos_end="2022-06-30"))
    query = MemoryQuery(as_of="2021-01-01", strategy_types=("momentum",), asset="SPY")
    assert service.graph_search(query, text="is_sharpe")
    assert service.graph_search(query, text="oos_sharpe") == []
    assert service.graph_search(query, text="is_sharpe", knowledge_as_of="2025-12-31") == []
