"""Tests for the unified memory layer (src/memory). See docs/MEMORY_LAYER_DESIGN.md §10."""

import numpy as np
import pandas as pd
import pytest

from src.memory import MemoryQuery, MemoryService, Note, Trial
from src.memory.beliefs import BeliefParams, compute_beliefs, deflation
from src.memory.canonical import config_key
from src.utils.config import config

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


# -- point-in-time visibility ------------------------------------------------

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


# -- beliefs -----------------------------------------------------------------

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


# -- pack / prompt ------------------------------------------------------------

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


# -- modes --------------------------------------------------------------------

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
    assert stored.failure_mode == "oos_decay"
    with svc._conn() as conn:
        status = conn.execute("SELECT status FROM mem_notes WHERE note_id = ?", (hyp,)).fetchone()[0]
    assert status == "rejected"
    assert not svc.attach_oos("missing", oos_sharpe=1.0, oos_start=None, oos_end=None)


# -- agent loop integration ---------------------------------------------------

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
