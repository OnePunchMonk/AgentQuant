"""Tests: memory."""
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.agent.memory_layer import AgenticMemoryLayer
from src.agent.memory_tool import MemorySearchTool
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.memory import MemoryQuery, MemoryService, Note, Trial, schema
from src.memory.beliefs import BeliefParams, compute_beliefs, deflation
from src.memory.canonical import config_key
from src.utils.config import config
from tests.helpers import make_trial


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


BULL = {"vix_percentile": 20.0, "momentum_63d": 0.10, "vol_vs_avg": 0.8,
        "drawdown_from_peak": -0.01, "price_vs_sma200": 0.08}


CRASH = {"vix_percentile": 95.0, "momentum_63d": -0.15, "vol_vs_avg": 2.2,
         "drawdown_from_peak": -0.25, "price_vs_sma200": -0.12}


PARAMS_A = {"fast_window": 20, "slow_window": 100}


PARAMS_B = {"fast_window": 5, "slow_window": 20}


def _svc(tmp_path, **kw):
    return MemoryService(db_path=str(tmp_path / "mem.db"), mode=kw.pop("mode", "read_write"), **kw)


def _trial(params=PARAMS_A, *, data_end="2021-06-30", is_sharpe=1.0, oos_sharpe=None,
           oos_end=None, regime_vec=None, regime_label="LowVol-Bull", run_id="r1",
           outcome="accepted", asset="SPY", strategy_type="momentum", n_days=750):
    return Trial(asset=asset, strategy_type=strategy_type, params=dict(params), run_id=run_id,
                 data_start="2019-01-01", data_end=data_end, regime_label=regime_label,
                 regime_vec=dict(regime_vec if regime_vec is not None else BULL),
                 is_sharpe=is_sharpe, is_n_days=n_days, oos_sharpe=oos_sharpe,
                 oos_start="2021-07-01" if oos_end else None, oos_end=oos_end,
                 outcome=outcome, source="test")


def _query(as_of, **kw):
    kw.setdefault("strategy_types", ("momentum",))
    kw.setdefault("regime_vec", BULL)
    kw.setdefault("regime_label", "LowVol-Bull")
    return MemoryQuery(as_of=as_of, **kw)


def test_record_trials_batch_and_rollback(tmp_path, monkeypatch):
    svc = _svc(tmp_path)
    first, second = _trial(), _trial(run_id="r2")
    assert svc.record_trials([first, second]) == [first.trial_id, second.trial_id]
    assert svc.stats()["trials"] == 2
    assert svc.record_trials([]) == []
    readonly = _svc(tmp_path, mode="read")
    assert readonly.record_trials([first, second]) == ["", ""]

    third, bad = _trial(run_id="r3"), _trial(run_id="r4")
    def fail():
        raise ValueError("bad trial")
    monkeypatch.setattr(bad, "to_row", fail)
    with pytest.raises(ValueError, match="bad trial"):
        svc.record_trials([third, bad])
    assert svc.stats()["trials"] == 2


def _fixture(seed=3, n=400):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(0.0002 + 0.01 * rng.standard_normal(n)))
    return {"DEMO": pd.DataFrame({"Open": close * 0.998, "High": close * 1.005, "Low": close * 0.995,
                                  "Close": close, "Volume": 1_000_000}, index=dates)}


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setattr(config, "results_db_path", str(tmp_path / "results.db"))
    monkeypatch.setattr(config.memory, "mode", "read_write")
    return tmp_path / "results.db"


def test_agentic_memory_extracts_strategy_patterns(tmp_path):
    memory = StrategyMemory(db_path=str(tmp_path / "memory.db"))
    memory.store(
        PastResult(
            regime="LowVol-Bull",
            strategy_type="momentum",
            params=json.dumps({"fast_window": 20, "slow_window": 100}),
            sharpe=0.8,
            total_return=0.12,
            max_drawdown=0.08,
        )
    )
    memory.store(
        PastResult(
            regime="LowVol-Bull",
            strategy_type="momentum",
            params=json.dumps({"fast_window": 50, "slow_window": 150}),
            sharpe=0.6,
            total_return=0.10,
            max_drawdown=0.10,
        )
    )

    patterns = AgenticMemoryLayer(memory).extract_patterns(
        regime="LowVol-Bull",
        strategy_types=["momentum"],
    )

    assert patterns
    assert any(pattern.verdict == "worked" for pattern in patterns)
    assert "LowVol-Bull" in AgenticMemoryLayer(memory).to_prompt_context("LowVol-Bull", ["momentum"])


def test_memory_markdown_export_contains_metrics(tmp_path):
    memory = StrategyMemory(db_path=str(tmp_path / "memory.db"))
    memory.store(
        PastResult(
            regime="HighVol-Bear",
            strategy_type="mean_reversion",
            params=json.dumps({"window": 20, "num_std": 2.0}),
            sharpe=0.4,
            total_return=0.05,
            max_drawdown=0.04,
            reasoning="Worked during volatile chop.",
        )
    )

    markdown = AgenticMemoryLayer(memory).render_markdown(limit=5)

    assert "AgentQuant Strategy Memory" in markdown
    assert "HighVol-Bear" in markdown
    assert "0.400" in markdown


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


def test_future_trials_are_invisible(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(data_end="2022-01-01", oos_sharpe=0.9, oos_end="2022-06-01"))
    assert svc.visible_trials(_query("2021-12-31")) == []


def test_oos_evidence_hidden_until_its_window_closes(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(data_end="2022-01-01", oos_sharpe=0.9, oos_end="2022-06-01"))
    [mid] = svc.visible_trials(_query("2022-03-01"))
    assert mid.oos_sharpe is None and mid.oos_end is None
    [after] = svc.visible_trials(_query("2022-06-01"))
    assert after.oos_sharpe == pytest.approx(0.9)


def test_live_query_sees_everything(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(data_end="2030-01-01"))
    assert len(svc.visible_trials(_query(None))) == 1


def test_undated_rows_only_visible_when_opted_in(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(data_end=None))
    assert svc.visible_trials(_query("2021-01-01")) == []
    opted_in = _svc(tmp_path, include_undated=True)
    assert len(opted_in.visible_trials(_query("2021-01-01"))) == 1


def test_own_run_excluded(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(run_id="current"))
    assert svc.visible_trials(_query(None, exclude_run_id="current")) == []


def test_matches_filter_visible_memory_oracle(tmp_path):
    """Chronological episodes: the store-level cutoff agrees with the list-level oracle."""
    from src.agent.search_arms import filter_visible_memory

    svc = _svc(tmp_path)
    episodes = [("ep1", "2019-12-31"), ("ep2", "2020-12-31"), ("ep3", "2021-12-31")]
    entries = []
    for ep, end in episodes:
        t = _trial(data_end=end, run_id=ep)
        svc.record_trial(t)
        entries.append({"episode_id": ep, "trial_id": t.trial_id})
    order = [ep for ep, _ in episodes]
    for i, (ep, end) in enumerate(episodes):
        # The agent for episode i runs with as_of just before its own data end.
        as_of = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        store_ids = {t.trial_id for t in svc.visible_trials(_query(as_of))}
        oracle_ids = {e["trial_id"] for e in filter_visible_memory(entries, ep, order)}
        assert store_ids == oracle_ids


def test_repeated_config_collapses_into_one_belief(tmp_path):
    svc = _svc(tmp_path)
    for i in range(3):
        svc.record_trial(_trial(run_id=f"r{i}", is_sharpe=-0.4, outcome="rejected"))
    beliefs = svc.beliefs(_query(None))
    assert len(beliefs) == 1
    assert beliefs[0].n_trials == 3
    assert beliefs[0].verdict == "avoid"


def test_in_sample_only_never_works_or_seeds(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(is_sharpe=3.0))
    pack = svc.recall(_query(None))
    [belief] = pack.beliefs
    assert belief.verdict in ("promising", "unproven")
    assert pack.seeds == []


def test_oos_backed_config_works_and_seeds(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(oos_sharpe=1.2, oos_end="2021-12-31"))
    pack = svc.recall(_query(None))
    assert pack.seeds and pack.seeds[0].verdict == "works"
    assert pack.seeds[0].params == PARAMS_A


def test_deflation_grows_with_number_of_configs():
    assert deflation(2, 1.0, 750) < deflation(20, 1.0, 750) < deflation(200, 1.0, 750)


def test_deflation_counts_distinct_configs_not_repeats():
    q = _query(None)
    one_config = [_trial(run_id=f"r{i}", is_sharpe=1.5) for i in range(10)]
    many_configs = [_trial({"fast_window": i, "slow_window": 100 + i}, is_sharpe=1.5) for i in range(10)]
    [repeat] = compute_beliefs(one_config, q)
    varied = compute_beliefs(many_configs, q)[0]
    assert repeat.deflation < varied.deflation


def test_decay_detected_and_avoided(tmp_path):
    svc = _svc(tmp_path)
    for i in range(2):
        svc.record_trial(_trial(run_id=f"r{i}", is_sharpe=2.0, oos_sharpe=-0.3, oos_end="2021-12-31"))
    pack = svc.recall(_query(None))
    assert [b.verdict for b in pack.avoid] == ["decays"]
    assert pack.avoid_param_keys("momentum") == {(("fast_window", 20), ("slow_window", 100))}


def test_regime_similarity_beats_label_match(tmp_path):
    svc = _svc(tmp_path)
    near = _trial(PARAMS_A, regime_vec=BULL, regime_label="MidVol-Bull", oos_sharpe=1.0, oos_end="2021-12-31")
    far = _trial(PARAMS_B, regime_vec=CRASH, regime_label="LowVol-Bull", oos_sharpe=1.0, oos_end="2021-12-31")
    svc.record_trials([near, far])
    beliefs = svc.beliefs(_query(None, regime_label="LowVol-Bull", regime_vec=BULL))
    by_key = {b.config_key: b for b in beliefs}
    assert by_key[near.config_key].max_similarity > 0.9
    assert far.config_key not in by_key or by_key[far.config_key].max_similarity < 0.2
    assert beliefs[0].config_key == near.config_key


def test_config_key_is_canonical():
    assert config_key("momentum", "SPY", {"a": 1, "b": 2.0}) == config_key("momentum", "SPY", {"b": 2, "a": 1})
    assert config_key("momentum", "SPY", {"a": np.int64(1)}) == config_key("momentum", "SPY", {"a": 1})


def test_other_assets_count_at_half_weight():
    q = _query(None, asset="SPY")
    spy = _trial(asset="SPY", oos_sharpe=1.0, oos_end="2021-12-31")
    qqq = _trial(asset="QQQ", oos_sharpe=1.0, oos_end="2021-12-31")
    beliefs = {b.asset: b for b in compute_beliefs([spy, qqq], q, BeliefParams(), preferred_asset="SPY")}
    assert beliefs["SPY"].shrunk > beliefs["QQQ"].shrunk


def test_prompt_respects_token_budget(tmp_path):
    svc = _svc(tmp_path)
    rng = np.random.default_rng(0)
    trials = [
        _trial({"fast_window": int(f), "slow_window": int(f) * 4}, run_id=f"r{i}",
               is_sharpe=float(rng.normal(0.5, 1)), oos_sharpe=float(rng.normal(0.3, 1)),
               oos_end="2021-12-31")
        for i, f in enumerate(rng.integers(2, 60, size=2000))
    ]
    svc.record_trials(trials)
    svc.record_note(Note(kind="memo", body="x" * 5000, data_end="2021-01-01", strategy_type="momentum"))
    pack = svc.recall(_query(None, token_budget=300, k_beliefs=50))
    assert len(pack.to_prompt()) / 4 <= 300 + 60  # header lines are outside section budgets


def test_snapshot_is_deterministic_and_logged(tmp_path):
    svc = _svc(tmp_path)
    svc.record_trial(_trial(oos_sharpe=1.0, oos_end="2021-12-31"))
    a = svc.recall(_query("2022-01-01"), run_id="run-a")
    b = svc.recall(_query("2022-01-01"), run_id="run-b")
    assert a.snapshot_id == b.snapshot_id
    assert svc.served_snapshots("run-a") == [a.snapshot_id]


def test_notes_obey_as_of(tmp_path):
    svc = _svc(tmp_path)
    svc.record_note(Note(kind="memo", body="future insight", data_end="2023-01-01", strategy_type="momentum"))
    svc.record_note(Note(kind="memo", body="past insight", data_end="2020-01-01", strategy_type="momentum"))
    bodies = [n.body for n in svc.visible_notes(_query("2021-01-01"))]
    assert bodies == ["past insight"]


def test_mode_off_reads_and_writes_nothing(tmp_path):
    svc = _svc(tmp_path, mode="off")
    assert svc.record_trial(_trial()) == ""
    assert svc.recall(_query(None)).is_empty
    assert not (tmp_path / "mem.db").exists()


def test_mode_read_never_writes(tmp_path):
    _svc(tmp_path).record_trial(_trial(oos_sharpe=1.0, oos_end="2021-12-31"))
    reader = _svc(tmp_path, mode="read")
    assert reader.record_trial(_trial(run_id="x")) == ""
    assert not reader.recall(_query(None)).is_empty
    assert reader.stats()["trials"] == 1
    assert reader.stats()["reads"] == 0


def test_attach_oos_flags_decay_and_resolves_hypothesis(tmp_path):
    svc = _svc(tmp_path)
    hyp = svc.record_note(Note(kind="hypothesis", body="slow trend persists", status="proposed"))
    t = _trial(is_sharpe=1.8)
    t.hypothesis_id = hyp
    svc.record_trial(t)
    assert svc.attach_oos(t.trial_id, oos_sharpe=-0.2, oos_start="2021-07-01", oos_end="2021-12-31",
                          min_acceptable_sharpe=0.3)
    stored = svc.get_trial(t.trial_id)
    assert stored.oos_failure_mode == "oos_decay"  # kept apart from the in-sample failure_mode
    assert stored.failure_mode != "oos_decay"
    with svc._conn() as conn:
        status = conn.execute("SELECT status FROM mem_notes WHERE note_id = ?", (hyp,)).fetchone()[0]
    assert status == "rejected"
    assert not svc.attach_oos("missing", oos_sharpe=1.0, oos_start=None, oos_end=None)


def test_agent_run_records_trials_holdout_and_snapshot(isolated_memory):
    from src.agent.agent_graph import run_agent

    data = _fixture()
    state = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=2)
    svc = MemoryService(db_path=str(isolated_memory))

    trials = svc.list_trials(asset="DEMO", limit=100)
    assert len(trials) == len(state["all_results"])
    assert {t.run_id for t in trials} == {state["run_id"]}
    search_end = state["ohlcv_data"]["DEMO"].index.max().strftime("%Y-%m-%d")
    assert {t.data_end for t in trials} == {search_end}
    assert state["as_of"] == search_end

    best = state["best_result"]
    assert best.get("holdout_sharpe") is not None
    stored = svc.get_trial(best["trial_id"])
    assert stored.oos_sharpe == pytest.approx(best["holdout_sharpe"])
    assert stored.oos_end == data["DEMO"].index.max().strftime("%Y-%m-%d")

    assert state["run_manifest"]["run_id"] == state["run_id"]
    assert state["run_manifest"]["memory_snapshot_id"] == state["memory_snapshot_id"]


def test_second_run_reads_first_run_but_not_its_holdout(isolated_memory):
    from src.agent.agent_graph import run_agent

    data = _fixture()
    first = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1)
    second = run_agent(data, strategy_type="momentum", asset="DEMO", max_iterations=1)
    pack = second["memory_pack"]
    assert pack.n_visible_trials == len(first["all_results"])
    # Same data => the first run's holdout window ends after this run's as_of,
    # so its OOS evidence must stay hidden from this run's search loop.
    assert all(b.n_oos == 0 for b in pack.beliefs + pack.avoid)
    assert "MEMORY (as of" in second["context"].memory_context


def test_agent_with_memory_off_writes_no_trials(isolated_memory, monkeypatch):
    from src.agent.agent_graph import run_agent

    monkeypatch.setattr(config.memory, "mode", "off")
    state = run_agent(_fixture(), strategy_type="momentum", asset="DEMO", max_iterations=1)
    assert state["memory_pack"] is None
    assert MemoryService(db_path=str(isolated_memory), mode="read").stats()["trials"] == 0


def test_memory_tool_returns_scoped_provenance_and_explanation(tmp_path):
    service = MemoryService(db_path=str(tmp_path / "memory.db"), mode="read_write")
    service.record_trial(make_trial("visible"))
    service.record_trial(make_trial("future", data_end="2022-01-01"))
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
    service.record_trial(make_trial("recorded-late", created_at="2022-01-01T00:00:00+00:00"))
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


def test_partial_oos_fields_are_hidden_until_the_window_closes(tmp_path):
    """A trial with OOS metadata but no OOS Sharpe (a partial or directly built record) must not
    show its return, window end or failure mode to a query made before the window closes."""
    svc = _svc(tmp_path)
    svc.record_trial(Trial(asset="SPY", strategy_type="momentum", params={"fast_window": 5}, data_end="2020-01-01",
                           oos_end="2021-01-01", oos_return=0.2, oos_failure_mode="oos_decay", outcome="accepted"))
    early = svc.visible_trials(MemoryQuery(as_of="2020-06-01"))[0]
    assert (early.oos_end, early.oos_return, early.oos_failure_mode) == (None, None, None)
    late = svc.visible_trials(MemoryQuery(as_of="2021-06-01"))[0]
    assert (late.oos_end, late.oos_return, late.oos_failure_mode) == ("2021-01-01", 0.2, "oos_decay")
