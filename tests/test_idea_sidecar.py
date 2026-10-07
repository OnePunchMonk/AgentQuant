"""Fixture tests for the shadow idea-generation sidecar (issue #43, milestone 1)."""

import json

import pytest

from src.agent import idea_sidecar as sc
from src.agent import search_arms
from src.agent.episode_splits import build_episodes, synthetic_ohlcv
from src.agent.parameter_grid import ParameterGrid
from src.agent.policy_mutation import FinalHoldoutGuard
from src.research.alpha_store import FailureRecord

GRID = ParameterGrid().get_grid("momentum")


@pytest.fixture
def setup():
    ohlcv = synthetic_ohlcv(seed=7, n_days=600, asset="SIM")
    ep = build_episodes(ohlcv, "SIM", n_episodes=1, dev_days=300, holdout_days=60)[0]
    return ohlcv, ep


def _failure(ts, params=None):
    return FailureRecord(timestamp=ts, regime="x", strategy_type="momentum",
                         params=params or GRID[0], failure_mode="below_threshold")


def test_feature_flag_default_off(monkeypatch):
    monkeypatch.delenv(sc.FEATURE_FLAG_ENV, raising=False)
    assert not sc.sidecar_enabled()
    monkeypatch.setenv(sc.FEATURE_FLAG_ENV, "1")
    assert sc.sidecar_enabled()


def test_observation_hides_future_memory_and_holdout(setup):
    ohlcv, ep = setup
    failures = [
        _failure(f"{ep.dev_start}T00:00:00+00:00"),                   # visible
        _failure(f"{ep.holdout_start}T00:00:00+00:00", GRID[1]),      # future: dropped
        _failure("not-a-date", GRID[2]),                              # unparseable: dropped
    ]
    rec = sc.run_shadow_episode(ohlcv, ep, sc.HeuristicPolicy(), sc.Budget(3), 5.0,
                                failures=failures)
    for step in rec.steps:
        assert len(step.observation.visible_failures) == 1
        blob = json.dumps(step.observation.to_dict())
        assert ep.holdout_start not in blob and ep.holdout_end not in blob
        assert "holdout" not in blob


def test_policy_only_sees_dev_rows(setup, monkeypatch):
    ohlcv, ep = setup
    import pandas as pd
    seen = []
    orig = search_arms._evaluate_params
    monkeypatch.setattr(search_arms, "_evaluate_params",
                        lambda df, p, c: (seen.append(df), orig(df, p, c))[1])
    rec = sc.run_shadow_episode(ohlcv, ep, sc.HeuristicPolicy(), sc.Budget(4), 5.0)
    dev_calls = seen[:rec.dev_proposals_evaluated]
    assert dev_calls and all((d.index <= pd.Timestamp(ep.dev_end)).all() for d in dev_calls)


def test_actions_stay_in_canonical_space(setup):
    ohlcv, ep = setup
    for pol in (sc.HeuristicPolicy(), sc.RandomPolicy(seed=3)):
        rec = sc.run_shadow_episode(ohlcv, ep, pol, sc.Budget(6), 5.0)
        assert all(s.action.params in GRID for s in rec.steps)


def test_out_of_space_action_is_rejected(setup):
    ohlcv, ep = setup

    class Rogue:
        name, version = "rogue", "1"

        def config(self):
            return {}

        def act(self, obs, candidates):
            return sc.Action(sc.ProposalSource.GRID, 0, {"fast_window": 3, "slow_window": 7}, 1.0)

    with pytest.raises(sc.ActionOutOfSpace):
        sc.run_shadow_episode(ohlcv, ep, Rogue(), sc.Budget(2), 5.0)


def test_random_and_heuristic_use_identical_budget(setup, monkeypatch):
    ohlcv, ep = setup
    counts = {}
    orig = search_arms._evaluate_params
    for pol in (sc.HeuristicPolicy(), sc.RandomPolicy(seed=1)):
        n = []
        monkeypatch.setattr(search_arms, "_evaluate_params",
                            lambda df, p, c, n=n: (n.append(1), orig(df, p, c))[1])
        rec = sc.run_shadow_episode(ohlcv, ep, pol, sc.Budget(5), 5.0)
        holdout_calls = 1 if rec.winner_params is not None else 0  # holdout is graded only for a gated winner
        counts[pol.name] = (len(n) - holdout_calls, rec.dev_proposals_evaluated)
    assert counts["heuristic_priority"] == counts["random_uniform"] == (5, 5)


def test_replay_reproduces_logged_decisions(setup):
    ohlcv, ep = setup
    for pol in (sc.HeuristicPolicy(), sc.RandomPolicy(seed=11)):
        rec = sc.run_shadow_episode(ohlcv, ep, pol, sc.Budget(5), 5.0)
        assert sc.replay_matches(rec, pol)
    assert not sc.replay_matches(rec, sc.RandomPolicy(seed=12))


def test_quality_gate_cannot_be_bypassed(setup):
    ohlcv, ep = setup
    rec = sc.run_shadow_episode(ohlcv, ep, sc.HeuristicPolicy(), sc.Budget(5), 5.0, min_sharpe=1e9)
    assert rec.winner_params is None and rec.status == "missing"
    assert rec.reward.composite is None  # missing is never coerced to 0


def test_final_holdout_is_graded_once(setup, tmp_path):
    ohlcv, ep = setup
    guard = FinalHoldoutGuard(tmp_path / "used.json")
    sc.run_shadow_episode(ohlcv, ep, sc.HeuristicPolicy(), sc.Budget(2), 5.0,
                          final_holdout=True, holdout_guard=guard)
    with pytest.raises(RuntimeError, match="already consumed"):
        sc.run_shadow_episode(ohlcv, ep, sc.HeuristicPolicy(), sc.Budget(2), 5.0,
                              final_holdout=True, holdout_guard=guard)


def test_record_is_serializable_with_required_fields(setup, tmp_path):
    ohlcv, ep = setup
    rec = sc.run_shadow_episode(ohlcv, ep, sc.RandomPolicy(seed=2), sc.Budget(3), 5.0,
                                evidence_tier="fixture")
    d = json.loads(rec.write(tmp_path / "r.json").read_text())
    assert d["schema_version"] == sc.SCHEMA_VERSION and d["evidence_tier"] == "fixture"
    assert d["sidecar_hash"] and d["budget"]["max_proposals"] == 3
    assert set(d["reward"]) >= {"holdout_net_return", "holdout_max_drawdown", "holdout_turnover",
                                "n_failed", "budget_used_frac", "composite"}
    step = d["trajectory"][0]
    assert "propensity" in step["action"] and step["rejected_candidates"]
    with pytest.raises(ValueError):
        sc.run_shadow_episode(ohlcv, ep, sc.HeuristicPolicy(), sc.Budget(1), 5.0, evidence_tier="great")
