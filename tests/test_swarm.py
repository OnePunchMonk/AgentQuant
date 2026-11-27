"""Tests: swarm."""

import numpy as np
import pandas as pd

from src.agent.proposal_generator import Proposal


def _data(n=620):
    idx = pd.date_range("2020-01-01", periods=n)
    close = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, n))
    spy = pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                        "Close": close, "Volume": 1_000_000}, index=idx)
    vix = pd.DataFrame({"Close": np.random.default_rng(1).uniform(15, 35, n)}, index=idx)
    return {"SPY": spy, "^VIX": vix}


def _run_swarm(monkeypatch, tmp_path, keep):
    from src.agent.swarm import orchestrator as orch

    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.chdir(tmp_path)
    called = []

    def critic(state):
        proposals = state.get("all_proposals", [])
        state["approved_proposals"] = proposals[:keep]
        state["rejected_count"] = len(proposals) - len(state["approved_proposals"])
        return state

    real = orch.run_backtest_coordinator

    def coordinator(state):
        called.append(len(state.get("approved_proposals", [])))
        return real(state)

    monkeypatch.setattr(orch, "run_critic_agent", critic)
    monkeypatch.setattr(orch, "run_backtest_coordinator", coordinator)
    result = orch.SwarmOrchestrator(strategy_types=["momentum"]).run(_data(), assets=["SPY"])
    return result, called


def _make_ohlcv(n: int = 620, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _make_data(n: int = 620) -> dict:
    return {
        "SPY": _make_ohlcv(n, seed=0),
        "^VIX": pd.DataFrame(
            {"Close": np.random.default_rng(1).uniform(15, 35, n)},
            index=pd.date_range("2020-01-01", periods=n),
        ),
    }


def test_swarm_does_not_overturn_a_critic_that_approves_nothing(monkeypatch, tmp_path):
    result, called = _run_swarm(monkeypatch, tmp_path, keep=0)
    assert result.status == "no_approved_proposals"
    assert result.proposals_approved == 0
    assert result.best_params == {}
    assert called == []  # nothing was backtested


def test_swarm_does_not_pad_approvals_up_to_the_minimum(monkeypatch, tmp_path):
    result, called = _run_swarm(monkeypatch, tmp_path, keep=1)
    assert result.status == "below_min_approved"
    assert result.proposals_approved == 1  # it used to be topped up to 2
    assert called == [1]


def test_swarm_reports_ok_when_enough_are_approved(monkeypatch, tmp_path):
    result, called = _run_swarm(monkeypatch, tmp_path, keep=3)
    assert result.status == "ok"
    assert called == [3]


def test_trend_following_needs_both_the_grid_and_the_ordering():
    from src.agent.context_builder import RegimeContext
    from src.agent.proposal_generator import Proposal
    from src.agent.swarm.critic_agent import CriticAgent

    on_grid = Proposal(params={"short_window": 10, "medium_window": 30, "long_window": 90}, confidence=0.5)
    ordered_but_off_grid = Proposal(params={"short_window": 7, "medium_window": 40, "long_window": 150}, confidence=0.5)
    on_grid_unordered = Proposal(params={"short_window": 90, "medium_window": 30, "long_window": 10}, confidence=0.5)
    approved, rejected = CriticAgent().review(
        [on_grid, ordered_but_off_grid, on_grid_unordered], "trend_following",
        RegimeContext(regime_label="LowVol-Bull"))
    assert [p.params for p in approved] == [on_grid.params]
    assert len(rejected) == 2


def test_regime_analyst_produces_narrative():
    from src.agent.swarm.regime_analyst import run_regime_analyst

    state = {"ohlcv_data": _make_data(), "assets": ["SPY"], "strategy_types": ["momentum"], "run_log": []}
    result = run_regime_analyst(state)

    assert result["regime_context"].regime_label
    assert "VIX" in result["regime_narrative"]


def test_critic_rejects_duplicate_proposals():
    from src.agent.context_builder import RegimeContext
    from src.agent.swarm.critic_agent import CriticAgent

    critic = CriticAgent()
    proposal = Proposal(params={"fast_window": 10, "slow_window": 30}, confidence=0.5)
    approved, rejected = critic.review(
        [proposal, Proposal(params={"fast_window": 10, "slow_window": 30}, confidence=0.4)],
        "momentum",
        RegimeContext(regime_label="LowVol-Bull"),
    )

    assert len(approved) == 1
    assert len(rejected) == 1


def test_swarm_runs_synthetic_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.chdir(tmp_path)

    from src.agent.swarm.orchestrator import SwarmOrchestrator
    from src.agent.swarm.state import SwarmResult

    result = SwarmOrchestrator(strategy_types=["momentum"]).run(_make_data(), assets=["SPY"])

    assert isinstance(result, SwarmResult)
    assert result.regime_label
    assert result.total_proposals_generated > 0
