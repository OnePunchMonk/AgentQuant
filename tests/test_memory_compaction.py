"""Deterministic fixtures for reversible graph memory compaction."""

from src.memory.compaction import MemoryCompactor, RetrievalProbe
from src.memory.graph import EvidenceGraph
from src.memory.models import Note, Trial


def _trial(trial_id: str, outcome: str, *, data_end: str = "2024-01-31",
           created_at: str = "2024-02-01T00:00:00+00:00") -> Trial:
    return Trial(asset="SPY", strategy_type="momentum", params={"lookback": 20},
                 trial_id=trial_id, run_id=f"run-{trial_id}", created_at=created_at,
                 data_end=data_end, outcome=outcome, is_sharpe=0.8 if outcome == "accepted" else -0.2,
                 reasoning="breakout decision from January bars")


def test_lossless_bundle_preserves_ids_clocks_provenance_graph_and_contradictions():
    accepted = _trial("trial-accepted", "accepted")
    rejected = _trial("trial-rejected", "rejected", data_end="2024-02-15")
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
    trial = _trial("trial-1", "accepted")
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
    old = _trial("old", "accepted", data_end="2024-01-31")
    stale = _trial("stale", "rejected", data_end="2024-03-01")
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
    trial = _trial("visible", "accepted")
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
    trial = _trial("durable", "accepted")
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
    visible = _trial("visible", "accepted")
    stale_market = _trial("future-market", "rejected", data_end="2024-03-01")
    late_knowledge = _trial("future-known", "accepted", created_at="2024-04-01T00:00:00+00:00")
    probe = RetrievalProbe("historical result", query="outcome accepted",
        market_as_of="2024-02-15", knowledge_as_of="2024-02-15",
        expected_source_ids=("visible", "future-market", "future-known"))
    result = MemoryCompactor().compact([visible, stale_market, late_knowledge], probes=[probe])
    assert result.evaluation.original_relevant == 1
    assert result.evaluation.compacted_relevant == 1
    assert result.evaluation.citation_retention == 1.0

