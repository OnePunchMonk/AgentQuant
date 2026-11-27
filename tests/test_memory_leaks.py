"""Point-in-time safety of out-of-sample verdicts, plus lazy and cached database access (N13).

attach_oos used to overwrite a trial's failure_mode with 'oos_decay' and flip a hypothesis
note to confirmed/rejected, both derived from data after the trial's in-sample window. The
evidence graph asserted them with only the in-sample date, so an earlier query saw a
verdict that came from the future.
"""

import json
import sqlite3

import pytest

from src.memory import schema
from src.memory.models import MemoryQuery, Note, Trial
from src.memory.service import MemoryService


def _setup(tmp_path):
    svc = MemoryService(str(tmp_path / "m.db"), mode="read_write")
    hyp = svc.record_note(Note(kind="hypothesis", body="slow trend persists", strategy_type="momentum",
                               asset="SPY", config_key="k1", data_end="2020-06-30", status="proposed"))
    trial = Trial(asset="SPY", strategy_type="momentum", params={"fast_window": 10, "slow_window": 50},
                  run_id="r1", data_end="2020-06-30", is_sharpe=1.4, outcome="accepted",
                  hypothesis_id=hyp, regime_label="LowVol-Bull")
    svc.record_trial(trial)
    svc.attach_oos(trial.trial_id, oos_sharpe=-0.6, oos_start="2020-07-01", oos_end="2021-03-31",
                   min_acceptable_sharpe=0.3)
    return svc, trial


def _claims(svc, as_of):
    q = MemoryQuery(as_of=as_of, strategy_types=("momentum",), asset="SPY")
    return {json.dumps(hit.node.attributes, default=str, sort_keys=True)
            for hit in svc.graph_search(q, text="", limit=100)}, svc.visible_trials(q)


def test_future_verdicts_are_hidden_from_earlier_queries(tmp_path):
    svc, _ = _setup(tmp_path)
    facts, trials = _claims(svc, "2020-12-31")  # before the out-of-sample window closed
    blob = " ".join(facts)
    assert "oos_decay" not in blob and "confirmed" not in blob and "rejected" not in blob
    assert trials[0].oos_sharpe is None and trials[0].oos_failure_mode is None
    assert trials[0].failure_mode is None  # the in-sample label is untouched


def test_the_same_verdicts_appear_once_the_window_has_closed(tmp_path):
    svc, _ = _setup(tmp_path)
    facts, trials = _claims(svc, "2021-04-30")
    blob = " ".join(facts)
    assert "oos_decay" in blob and "rejected" in blob
    assert trials[0].oos_failure_mode == "oos_decay"


def test_attach_oos_does_not_overwrite_the_in_sample_failure_mode(tmp_path):
    svc = MemoryService(str(tmp_path / "m.db"), mode="read_write")
    trial = Trial(asset="SPY", strategy_type="momentum", params={"fast_window": 5, "slow_window": 20},
                  run_id="r", data_end="2020-06-30", is_sharpe=0.2, outcome="watch",
                  failure_mode="below_threshold")
    svc.record_trial(trial)
    svc.attach_oos(trial.trial_id, oos_sharpe=-1.0, oos_start="2020-07-01", oos_end="2020-12-31",
                   min_acceptable_sharpe=0.3)
    stored = svc.get_trial(trial.trial_id)
    assert stored.failure_mode == "below_threshold"


def test_legacy_rows_are_read_and_migrated_safely(tmp_path):
    path = tmp_path / "legacy.db"
    svc = MemoryService(str(path), mode="read_write")
    trial = Trial(asset="SPY", strategy_type="momentum", params={"fast_window": 5, "slow_window": 20},
                  run_id="r", data_end="2020-06-30", is_sharpe=1.5, outcome="rejected")
    svc.record_trial(trial)
    with sqlite3.connect(path) as conn:  # what a schema-1 database held after attach_oos
        conn.execute("UPDATE mem_trials SET failure_mode = 'oos_decay', oos_failure_mode = NULL")
        conn.execute("PRAGMA user_version = 1")
    schema._SCHEMA_READY_FOR_TEST = True  # noqa: B010 - documents intent; the cache is per process
    from src.memory import service as service_module

    service_module._SCHEMA_READY.discard(str(path.resolve()))
    reader = MemoryService(str(path), mode="read")
    seen = reader.get_trial(trial.trial_id)  # reading migrates nothing but still masks the label
    assert seen.failure_mode == "negative_sharpe" and seen.oos_failure_mode == "oos_decay"
    writer = MemoryService(str(path), mode="read_write")
    writer.record_note(Note(kind="memo", body="x"))  # the first write migrates the file
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT failure_mode, oos_failure_mode FROM mem_trials").fetchone()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
    assert row == ("negative_sharpe", "oos_decay")


def test_reading_a_missing_database_creates_nothing(tmp_path):
    path = tmp_path / "never.db"
    svc = MemoryService(str(path), mode="read_write")
    assert not path.exists()  # construction does not touch the disk
    assert svc.visible_trials(MemoryQuery(as_of=None, strategy_types=("momentum",))) == []
    assert svc.stats()["trials"] == 0
    assert not path.exists()  # reads do not create it either
    svc.record_note(Note(kind="memo", body="first write"))
    assert path.exists()


def test_one_service_per_run_not_one_per_call(monkeypatch):
    from src.agent import agent_graph

    built = []

    class Counting(MemoryService):
        def __init__(self, *a, **k):
            built.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(agent_graph, "MemoryService", Counting)
    agent_graph._RUN_SERVICES.clear()
    state = {"run_id": "run-1"}
    first = agent_graph._memory_service(state)
    assert agent_graph._memory_service(state) is first and agent_graph._memory_service(state) is first
    assert len(built) == 1
    assert agent_graph._memory_service({"run_id": "run-2"}) is not first
    agent_graph._RUN_SERVICES.clear()


def test_the_schema_is_created_once_per_process(tmp_path, monkeypatch):
    calls = []
    real = schema.ensure_schema
    monkeypatch.setattr("src.memory.service.ensure_schema", lambda conn: (calls.append(1), real(conn))[1])
    svc = MemoryService(str(tmp_path / "once.db"), mode="read_write")
    for i in range(5):
        svc.record_note(Note(kind="memo", body=f"n{i}"))
    assert len(calls) == 1
    with pytest.raises(AttributeError):
        _ = svc.nonexistent  # sanity: the service is a plain object


def test_the_schema_cache_follows_the_file_not_the_path_string(tmp_path, monkeypatch):
    """Two tests (or runs) using the same relative path in different directories are different databases."""
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.chdir(directory)
        svc = MemoryService("results.db", mode="read_write")
        svc.record_note(Note(kind="memo", body=name))  # would fail with 'no such table' on the second pass
        assert svc.stats()["notes"] == 1
