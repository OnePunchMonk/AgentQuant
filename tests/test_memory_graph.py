"""Tests: memory graph."""
from src.memory.compaction import MemoryCompactor, RetrievalProbe
from src.memory.graph import EvidenceGraph
from src.memory.models import MemoryQuery, Note
from src.memory.service import MemoryService
from tests.helpers import make_trial


def _dated_trial(trial_id, outcome, *, data_end="2024-01-31", created_at="2024-02-01T00:00:00+00:00"):
    return make_trial(trial_id, outcome=outcome, params={"lookback": 20}, data_end=data_end, created_at=created_at,
                      is_sharpe=0.8 if outcome == "accepted" else -0.2,
                      reasoning="breakout decision from January bars")


def test_lossless_bundle_preserves_ids_clocks_provenance_graph_and_contradictions():
    accepted = _dated_trial("trial-accepted", "accepted")
    rejected = _dated_trial("trial-rejected", "rejected", data_end="2024-02-15")
    note = Note(kind="hypothesis", body="breakout failed in holdout", note_id="note-1",
                created_at="2024-02-20T00:00:00+00:00", data_end="2024-02-15",
                status="rejected", strategy_type="momentum", asset="SPY",
                meta={"claim": {"subject": "cfg-1", "predicate": "hypothesis_status",
                                "value": "rejected"}}, source="fixture:holdout")
    compactor = MemoryCompactor()
    bundle = compactor.stage([rejected, accepted], [note])
    payload = bundle.unpack()

    assert set(bundle.source_ids) == {"trial-accepted", "trial-rejected", "note-1"}
    assert bundle.market_dates["trial-rejected"] == "2024-02-15"
    assert bundle.knowledge_dates["note-1"] == "2024-02-20T00:00:00+00:00"
    assert any(edge["source_record_id"] == "trial-accepted" for edge in payload["graph"]["edges"])
    assert any(edge["source_record_id"] == "note-1" for edge in payload["graph"]["edges"])
    graph = EvidenceGraph.rebuild([accepted, rejected], [note])
    assert any(hit.contradictions for hit in graph.retrieve(query="outcome accepted"))
    assert {r["outcome"] for r in payload["records"] if r["record_type"] == "trial"} == {
        "accepted", "rejected"}


def test_pre_archive_gate_blocks_loss_of_decision_signal_and_rollback_restores_records(tmp_path):
    trial = _dated_trial("trial-1", "accepted")
    probe = RetrievalProbe("decision", expected_source_ids=("trial-1",),
                           expected_decisions=("outcome:accepted",))
    compactor = MemoryCompactor()

    def lossy(payload):
        payload["records"][0]["outcome"] = "watch"
        return payload

    result = compactor.compact([trial], probes=[probe], candidate_transform=lossy)
    assert not result.archived
    assert not result.evaluation.passed
    assert "decision retention below threshold" in result.evaluation.failures
    assert result.rollback_token is None

    archive_path = tmp_path / "memory.archive.json"
    good = compactor.compact([trial], probes=[probe], archive_path=archive_path)
    assert good.archived and good.rollback_token
    restored_trials, restored_notes = compactor.rollback(good.rollback_token, archive_path)
    assert restored_trials == [trial.__dict__]
    assert restored_notes == []


def test_stale_memory_is_not_citable_before_market_cutoff():
    old = _dated_trial("old", "accepted", data_end="2024-01-31")
    stale = _dated_trial("stale", "rejected", data_end="2024-03-01")
    probe = RetrievalProbe("historical", market_as_of="2024-02-15",
                           expected_source_ids=("old", "stale"))
    compactor = MemoryCompactor(min_citation=1.0)

    def loses_stale(payload):
        payload["records"] = [r for r in payload["records"] if r.get("trial_id") != "stale"]
        # Keep the source manifest so the citation gate, rather than the manifest,
        # demonstrates the temporal retrieval regression.
        return payload

    result = compactor.compact([old, stale], probes=[probe], candidate_transform=loses_stale)
    assert not result.archived
    assert result.evaluation.citation_retention == 1.0  # stale was invisible on both sides
    assert "source records were lost or introduced" in result.evaluation.failures


def test_probe_citations_and_relevance_fail_when_visible_source_is_dropped():
    trial = _dated_trial("visible", "accepted")
    probe = RetrievalProbe("visible citation", market_as_of="2024-02-15",
                           knowledge_as_of="2024-02-15", expected_source_ids=("visible",))
    compactor = MemoryCompactor()

    def drop(payload):
        payload["records"] = []
        return payload

    result = compactor.compact([trial], probes=[probe], candidate_transform=drop)
    assert not result.archived
    assert result.evaluation.relevance_retention == 0.0
    assert result.evaluation.citation_retention == 0.0


def test_archive_checksum_rejects_corruption_and_compaction_requires_durability(tmp_path):
    trial = _dated_trial("durable", "accepted")
    probe = RetrievalProbe("all", expected_source_ids=("durable",),
                           expected_decisions=("outcome:accepted",))
    compactor = MemoryCompactor()
    unstored = compactor.compact([trial], probes=[probe])
    assert not unstored.archived
    assert "durable archive path required" in unstored.evaluation.failures

    path = tmp_path / "archive.json"
    result = compactor.compact([trial], probes=[probe], archive_path=path)
    assert result.archived
    loaded = result.bundle.load(path)
    assert loaded.unpack()["records"][0]["trial_id"] == "durable"
    path.write_text(path.read_text().replace("durable", "corrupt"), encoding="utf-8")
    try:
        result.bundle.load(path)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted archive unexpectedly verified")


def test_probe_uses_actual_graph_retrieval_under_both_cutoffs():
    visible = _dated_trial("visible", "accepted")
    stale_market = _dated_trial("future-market", "rejected", data_end="2024-03-01")
    late_knowledge = _dated_trial("future-known", "accepted", created_at="2024-04-01T00:00:00+00:00")
    probe = RetrievalProbe("historical result", query="outcome accepted",
        market_as_of="2024-02-15", knowledge_as_of="2024-02-15",
        expected_source_ids=("visible", "future-market", "future-known"))
    result = MemoryCompactor().compact([visible, stale_market, late_knowledge], probes=[probe])
    assert result.evaluation.original_relevant == 1
    assert result.evaluation.compacted_relevant == 1
    assert result.evaluation.citation_retention == 1.0


def test_oos_graph_edge_does_not_leak_before_oos_close():
    graph = EvidenceGraph.rebuild([make_trial("a", oos_sharpe=1.2, oos_end="2022-06-30")])
    early = graph.retrieve(as_of="2022-06-29", query="oos_sharpe")
    late = graph.retrieve(as_of="2022-06-30", query="oos_sharpe")
    assert early == []
    assert len(late) == 1
    assert late[0].evidence[0].visible_at == "2022-06-30"


def test_contradictory_outcomes_are_preserved_and_linked():
    graph = EvidenceGraph.rebuild([
        make_trial("good", outcome="accepted"),
        make_trial("bad", outcome="rejected"),
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
    trials = [make_trial("z"), make_trial("a")]
    first = EvidenceGraph.rebuild(trials)
    second = EvidenceGraph.rebuild(list(reversed(trials)))
    assert first.nodes == second.nodes
    assert first.edges == second.edges
    assert first.retrieve(as_of="2022-01-01") == second.retrieve(as_of="2022-01-01")


def test_future_created_record_is_hidden_even_if_data_window_is_old():
    graph = EvidenceGraph.rebuild([make_trial("future", created_at="2024-01-01T00:00:00+00:00")])
    assert graph.retrieve(market_as_of="2022-01-01", knowledge_as_of="2022-01-01",
                          query="accepted") == []


def test_retrospective_replay_ignores_materialization_wall_clock():
    graph = EvidenceGraph.rebuild([make_trial("prior", created_at="2026-11-14T00:00:00+00:00",
                                          data_end="2020-12-31")])
    hits = graph.retrieve(market_as_of="2021-01-01", query="accepted")
    assert len(hits) == 1
    assert hits[0].evidence[0].known_at.startswith("2026-")
    assert graph.retrieve(market_as_of="2021-01-01", knowledge_as_of="2021-01-01",
                          query="accepted") == []


def test_future_market_window_is_excluded_from_historical_replay():
    graph = EvidenceGraph.rebuild([make_trial("later", data_end="2022-01-01")])
    assert graph.retrieve(market_as_of="2021-12-31", query="accepted") == []


def test_regime_filter_uses_source_record_association():
    bull = make_trial("bull", outcome="accepted")
    bear = make_trial("bear", outcome="rejected")
    bear.regime_label = "bear"
    graph = EvidenceGraph.rebuild([bull, bear])
    hits = graph.retrieve(market_as_of="2022-01-01", regime_label="bull", query="outcome")
    assert {hit.node.attributes["value"] for hit in hits} == {"accepted"}


def test_unknown_required_market_clock_fails_closed():
    graph = EvidenceGraph.rebuild([make_trial("undated", data_end=None)])
    assert graph.retrieve(market_as_of="2022-01-01", query="accepted") == []


def test_service_graph_search_respects_market_and_knowledge_cutoffs(tmp_path):
    service = MemoryService(db_path=str(tmp_path / "memory.sqlite"), mode="read_write")
    service.record_trial(make_trial("past", created_at="2026-01-01T00:00:00+00:00", data_end="2020-12-31",
                                oos_sharpe=1.2, oos_end="2022-06-30"))
    query = MemoryQuery(as_of="2021-01-01", strategy_types=("momentum",), asset="SPY")
    assert service.graph_search(query, text="is_sharpe")
    assert service.graph_search(query, text="oos_sharpe") == []
    assert service.graph_search(query, text="is_sharpe", knowledge_as_of="2025-12-31") == []


def test_top_k_keeps_contradictions_outside_query_and_regime():
    good = make_trial("good", outcome="accepted")
    bad = make_trial("bad", outcome="rejected")
    bad.regime_label = "bear"
    future = make_trial("future", outcome="rejected", created_at="2025-01-01")
    graph = EvidenceGraph.rebuild([good, bad, future])
    hits = graph.retrieve(query="accepted", regime_label="bull", limit=1,
                          market_as_of="2022-01-01", knowledge_as_of="2022-01-01")
    assert len(hits) == 1
    assert {edge.source_record_id for edge in hits[0].evidence} == {"good"}
    assert {edge.source_record_id for edge in hits[0].contradictions} == {"bad"}


def test_contradictions_are_resolved_only_for_top_k(monkeypatch):
    graph = EvidenceGraph.rebuild([make_trial(str(i), outcome="accepted" if i % 2 else "rejected")
                                   for i in range(40)])
    original = EvidenceGraph._contradicts
    calls = []

    def counted(left, right):
        calls.append(left.node_id)
        return original(left, right)

    monkeypatch.setattr(EvidenceGraph, "_contradicts", staticmethod(counted))
    hits = graph.retrieve(query="outcome", limit=1)
    assert len(hits) == 1
    assert set(calls) == {hits[0].node.node_id}
