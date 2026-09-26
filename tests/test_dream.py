"""Tests for dreaming mode (src/memory/dream.py), the offline consolidation sidecar."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.memory import MemoryQuery, MemoryService, Trial
from src.memory.dream import Dreamer, heartbeat_age_seconds

PARAMS = {"fast_window": 10, "slow_window": 30}


def _prices(n=600, seed=0, drift=0.0008):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(drift + 0.01 * rng.standard_normal(n)))
    return pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                         "Close": close, "Volume": 1_000_000}, index=dates)


@pytest.fixture
def svc(tmp_path):
    return MemoryService(db_path=str(tmp_path / "mem.db"), mode="read_write")


def _trial(df, *, end_pos, is_sharpe=1.5, run_id="r1"):
    return Trial(asset="DEMO", strategy_type="momentum", params=dict(PARAMS), run_id=run_id,
                 data_start=df.index[0].strftime("%Y-%m-%d"),
                 data_end=df.index[end_pos].strftime("%Y-%m-%d"),
                 regime_label="MidVol-Bull", is_sharpe=is_sharpe, is_n_days=end_pos, source="test")


def test_replay_attaches_forward_window_as_oos(svc):
    df = _prices()
    trial = _trial(df, end_pos=399)
    svc.record_trial(trial)
    report = Dreamer(svc, loader=lambda asset: df, oos_bars=126, min_oos_bars=42).run_once()

    assert report.replayed == 1 and not report.errors
    stored = svc.get_trial(trial.trial_id)
    assert stored.oos_sharpe is not None
    assert stored.oos_source == "dream_replay"
    assert stored.oos_start == df.index[400].strftime("%Y-%m-%d")
    assert stored.oos_end == df.index[400 + 125].strftime("%Y-%m-%d")


def test_replay_evidence_respects_as_of(svc):
    df = _prices()
    trial = _trial(df, end_pos=399)
    svc.record_trial(trial)
    Dreamer(svc, loader=lambda asset: df).run_once()
    stored = svc.get_trial(trial.trial_id)

    before = MemoryQuery(as_of=stored.data_end, strategy_types=("momentum",))
    [seen] = svc.visible_trials(before)
    assert seen.oos_sharpe is None
    after = MemoryQuery(as_of=stored.oos_end, strategy_types=("momentum",))
    [seen] = svc.visible_trials(after)
    assert seen.oos_sharpe == pytest.approx(stored.oos_sharpe)


def test_replay_waits_for_enough_forward_data(svc):
    df = _prices()
    trial = _trial(df, end_pos=len(df) - 10)
    svc.record_trial(trial)
    report = Dreamer(svc, loader=lambda asset: df, min_oos_bars=42).run_once()
    assert report.replay_skipped_no_data == 1
    assert svc.get_trial(trial.trial_id).oos_sharpe is None
    # Checked trials are not retried within the recheck window.
    assert Dreamer(svc, loader=lambda asset: df).run_once().replay_candidates == 0


def test_missing_data_does_not_fail_cycle(svc):
    df = _prices()
    svc.record_trial(_trial(df, end_pos=399))
    report = Dreamer(svc, loader=lambda asset: None).run_once()
    assert report.replay_skipped_no_data == 1 and not report.errors


def test_consolidation_notes_on_verdict_change_only(svc):
    df = _prices()
    for i in range(2):
        t = _trial(df, end_pos=300 + i, is_sharpe=2.5, run_id=f"r{i}")
        t.oos_sharpe, t.oos_start, t.oos_end = -0.5, "2020-06-01", "2020-12-31"
        svc.record_trial(t)
    dreamer = Dreamer(svc, loader=lambda asset: None, max_replays=0)
    first = dreamer.run_once()
    assert first.consolidations_written == 1
    assert dreamer.run_once().consolidations_written == 0

    notes = svc.visible_notes(MemoryQuery(as_of=None, strategy_types=("momentum",)))
    [note] = [n for n in notes if n.kind == "consolidation"]
    assert note.meta["verdict"] == "decays"
    assert "decays" in note.body
    # Dated at the latest market date its evidence used (the OOS window end).
    assert note.data_end == "2020-12-31"


def test_backfill_imports_legacy_tables_once(svc):
    from src.agent.strategy_memory import PastResult, StrategyMemory
    from src.research.nla_memory import NLAMemoryStore

    StrategyMemory(db_path=svc.db_path).store(PastResult(
        regime="LowVol-Bull", strategy_type="momentum",
        params=json.dumps(PARAMS), sharpe=0.9, holdout_sharpe=0.4))
    NLAMemoryStore(db_path=svc.db_path).store_agent_summary(
        regime="LowVol-Bull", strategy_type="momentum", params=PARAMS,
        narrative="trend held", metrics={"sharpe": 0.9})

    dreamer = Dreamer(svc, loader=lambda asset: None, max_replays=0)
    assert dreamer.run_once().backfilled == {
        "strategy_runs": 1, "alpha_candidates": 0, "nla_records": 1, "hypotheses": 0}
    assert dreamer.run_once().backfilled["strategy_runs"] == 0

    [legacy] = svc.list_trials()
    assert legacy.data_end is None and legacy.oos_sharpe == pytest.approx(0.4)
    # Undated rows are invisible to dated agent queries by default.
    assert svc.visible_trials(MemoryQuery(as_of="2030-01-01", strategy_types=("momentum",))) == []


def test_prunes_old_reads_and_writes_heartbeat(svc):
    svc.recall(MemoryQuery(as_of=None, strategy_types=("momentum",)))
    with sqlite3.connect(svc.db_path) as conn:
        conn.execute("UPDATE mem_reads SET created_at = '2000-01-01T00:00:00+00:00'")
    report = Dreamer(svc, loader=lambda asset: None, retention_days=30).run_once()
    assert report.reads_pruned == 1
    assert heartbeat_age_seconds(svc.db_path) is not None


def test_read_only_mode_refuses_to_dream(tmp_path):
    svc = MemoryService(db_path=str(tmp_path / "mem.db"), mode="read")
    report = Dreamer(svc, loader=lambda asset: None).run_once()
    assert report.errors


def test_watch_runs_bounded_cycles(svc):
    Dreamer(svc, loader=lambda asset: None).watch(interval_seconds=0, max_cycles=2)
    assert heartbeat_age_seconds(svc.db_path) is not None


def test_cli_dream_once(tmp_path, monkeypatch, capsys):
    from src.cli import build_parser
    from src.utils.config import config

    monkeypatch.setattr(config, "results_db_path", str(tmp_path / "results.db"))
    monkeypatch.setattr(config, "data_path", str(tmp_path / "no_data"))
    args = build_parser().parse_args(["dream", "--max-replays", "5"])
    assert args.func(args) == 0
    assert '"replayed": 0' in capsys.readouterr().out
