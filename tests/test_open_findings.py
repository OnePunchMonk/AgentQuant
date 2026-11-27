"""Regression tests for three open findings from the sweep.

  A7  the final-holdout guard was check-then-write, so two processes could both pass
  N8  the swarm critic's rejections were overturned to meet a quota of two
  R1  the RL shadow policy logged propensity 1.0, so no other policy could be valued
"""

import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agent import idea_sidecar as sc
from src.agent import ope
from src.agent.episode_splits import build_episodes, synthetic_ohlcv
from src.agent.parameter_grid import ParameterGrid
from src.agent.policy_mutation import FinalHoldoutGuard

GRID = ParameterGrid().get_grid("momentum")


# ------------------------------------------------------------------ A7: the guard
def test_exactly_one_of_many_concurrent_callers_gets_the_holdout(tmp_path):
    path = tmp_path / "guard.json"
    barrier = threading.Barrier(12)
    won, lost = [], []

    def attempt(i):
        barrier.wait()
        try:
            FinalHoldoutGuard(path).check_and_mark_used(f"run-{i}")
            won.append(i)
        except RuntimeError:
            lost.append(i)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(won) == 1 and len(lost) == 11
    assert f"run-{won[0]}" in path.read_text()


def test_second_use_names_the_first_run(tmp_path):
    guard = FinalHoldoutGuard(tmp_path / "g.json")
    guard.check_and_mark_used("first")
    with pytest.raises(RuntimeError, match="first"):
        guard.check_and_mark_used("second")


@pytest.mark.parametrize("content", ["", "not json", "{}"])
def test_any_existing_file_counts_as_spent(tmp_path, content):
    path = tmp_path / "g.json"
    path.write_text(content)
    with pytest.raises(RuntimeError, match="already consumed"):
        FinalHoldoutGuard(path).check_and_mark_used("late")


def test_ledger_anchoring_survives_a_fresh_checkout(tmp_path):
    from agentquant_gate import Ledger

    ledger = Ledger(tmp_path / "ledger.jsonl")
    FinalHoldoutGuard(tmp_path / "one.json", ledger=ledger).check_and_mark_used("first")
    assert any(e["type"] == "holdout_read" for e in ledger.entries())
    # A different receipt path (a fresh clone) is still refused, because the ledger remembers.
    with pytest.raises(RuntimeError):
        FinalHoldoutGuard(tmp_path / "two.json", ledger=ledger).check_and_mark_used("second")
    assert not (tmp_path / "two.json").exists()


# ------------------------------------------------------------------ N8: the swarm fails closed
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


# ------------------------------------------------------------------ R1: exploration logging
@pytest.fixture
def episode():
    ohlcv = synthetic_ohlcv(seed=7, n_days=900, asset="SIM")
    return ohlcv, build_episodes(ohlcv, "SIM", n_episodes=2, dev_days=300, holdout_days=60)[1]


def test_epsilon_greedy_logs_the_exact_action_probability(episode):
    ohlcv, ep = episode
    policy = sc.EpsilonGreedyPolicy(epsilon=0.2, seed=3)
    rec = sc.run_shadow_episode(ohlcv, ep, policy, sc.Budget(6), 5.0)
    assert rec.steps
    for step in rec.steps:
        n = len(step.candidates)
        greedy = sc.HeuristicPolicy().act(step.observation, step.candidates)
        is_greedy = (step.action.source, step.action.grid_index) == (greedy.source, greedy.grid_index)
        expected = 0.2 / n + (0.8 if is_greedy else 0.0)
        assert step.action.propensity == pytest.approx(expected)
        assert step.action.propensity >= 0.2 / n  # every candidate keeps support


def test_epsilon_greedy_decisions_replay_from_the_log(episode):
    ohlcv, ep = episode
    policy = sc.EpsilonGreedyPolicy(epsilon=0.3, seed=5)
    rec = sc.run_shadow_episode(ohlcv, ep, policy, sc.Budget(6), 5.0)
    assert sc.replay_matches(rec, policy)
    assert not sc.replay_matches(rec, sc.EpsilonGreedyPolicy(epsilon=0.3, seed=6))


def test_epsilon_greedy_probabilities_sum_to_one(episode):
    ohlcv, ep = episode
    policy = sc.EpsilonGreedyPolicy(epsilon=0.25)
    rec = sc.run_shadow_episode(ohlcv, ep, policy, sc.Budget(1), 5.0)
    step = rec.steps[0]
    greedy = sc.HeuristicPolicy().act(step.observation, step.candidates)
    n = len(step.candidates)
    total = sum(0.25 / n + (0.75 if (c.source, c.grid_index) == (greedy.source, greedy.grid_index) else 0.0)
                for c in step.candidates)
    assert total == pytest.approx(1.0)


def test_zero_epsilon_is_rejected():
    with pytest.raises(ValueError):
        sc.EpsilonGreedyPolicy(epsilon=0.0)


def _bandit(rng, n, epsilon):
    means = np.array([0.1, 0.3, 0.2, 0.5, 0.0])
    greedy = 1  # the logging policy's favourite arm
    probs = np.full(5, epsilon / 5)
    probs[greedy] += 1 - epsilon
    arms = rng.choice(5, size=n, p=probs)
    rewards = means[arms] + rng.normal(0, 0.3, n)
    return arms, rewards, probs[arms]


def test_off_policy_estimate_recovers_the_value_of_another_policy_when_the_log_explored():
    arms, rewards, behavior = _bandit(np.random.default_rng(0), 40_000, epsilon=0.5)
    target = (arms == 3).astype(float)  # a policy that always plays arm 3 (true value 0.5)
    assert ope.snips_value(rewards, behavior, target) == pytest.approx(0.5, abs=0.03)
    assert ope.ips_value(rewards, behavior, target) == pytest.approx(0.5, abs=0.04)
    report = ope.support_report(behavior, target)
    assert report["mass_below_floor"] == 0.0 and report["effective_sample_size"] > 500


def test_a_deterministic_log_cannot_value_another_policy():
    """The old behaviour: propensity 1.0, always the same action."""
    rewards = np.random.default_rng(1).normal(0.3, 0.3, 1000)
    behavior = np.ones(1000)
    target = np.zeros(1000)  # a policy that wants arm 3 never takes the logged arm 1
    for estimator in (ope.ips_value, ope.snips_value):
        with pytest.raises(ValueError, match="no overlap"):  # IPS used to report a confident 0.0
            estimator(rewards, behavior, target)
    assert ope.support_report(behavior, target)["effective_sample_size"] == 0.0


def test_ope_rejects_impossible_propensities():
    with pytest.raises(ValueError):
        ope.ips_value([1.0], [0.0], [1.0])


# ------------------------------------------------------------------ the production observation is complete
def test_sidecar_observation_shows_what_the_run_already_tried(monkeypatch, tmp_path):
    from src.agent import agent_graph, rl_shadow_tool
    from src.utils.config import config

    monkeypatch.setenv("AGENTQUANT_ASSISTANT_TOOLS", "1")
    monkeypatch.delenv("AGENTQUANT_IDEA_SIDECAR_EPSILON", raising=False)
    monkeypatch.setattr(config, "results_db_path", str(tmp_path / "m.db"))
    seen = {}
    real = rl_shadow_tool.advise

    def spy(observation, candidates, policy=None, **kwargs):
        seen["obs"], seen["policy"] = observation, policy
        return real(observation, candidates, policy, **kwargs)

    monkeypatch.setattr(rl_shadow_tool, "advise", spy)
    state = {"as_of": "2024-01-31", "run_id": "r1", "asset": "SPY", "max_iterations": 3,
             "all_results": [{"params": dict(GRID[0])}, {"params": dict(GRID[3])}]}
    agent_graph._run_assistant_sidecar_context(state, "momentum", SimpleNamespace(regime_label="Bull"), 2)
    assert seen["obs"].tried_grid_indices == (0, 3)
    assert seen["obs"].remaining_proposals == 10  # rounds 2 and 3 left, 5 proposals each
    assert seen["policy"] is None  # deterministic unless exploration is switched on


def test_exploration_is_opt_in_through_the_environment(monkeypatch):
    from src.agent import agent_graph

    monkeypatch.delenv("AGENTQUANT_IDEA_SIDECAR_EPSILON", raising=False)
    assert agent_graph._sidecar_policy("r1") is None
    monkeypatch.setenv("AGENTQUANT_IDEA_SIDECAR_EPSILON", "0.15")
    policy = agent_graph._sidecar_policy("r1")
    assert isinstance(policy, sc.EpsilonGreedyPolicy) and policy.epsilon == 0.15
    monkeypatch.setenv("AGENTQUANT_IDEA_SIDECAR_EPSILON", "0")
    assert agent_graph._sidecar_policy("r1") is None
    monkeypatch.setenv("AGENTQUANT_IDEA_SIDECAR_EPSILON", "banana")
    assert agent_graph._sidecar_policy("r1") is None


# ------------------------------------------------------------------ fixes from review
def test_ledger_claim_comes_first_so_a_crash_leaves_the_split_spent(tmp_path, monkeypatch):
    import os

    from agentquant_gate import Ledger

    from src.agent import policy_mutation

    ledger = Ledger(tmp_path / "ledger.jsonl")
    receipt = tmp_path / "receipt.json"
    real_open = os.open

    def failing_open(path, flags, *args, **kwargs):
        if str(path) == str(receipt):
            raise OSError("disk full")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(policy_mutation.os, "open", failing_open)
    with pytest.raises(OSError):
        FinalHoldoutGuard(receipt, ledger=ledger).check_and_mark_used("crashy")
    assert any(e["type"] == "holdout_read" for e in ledger.entries())  # still recorded as spent
    monkeypatch.undo()
    with pytest.raises(RuntimeError):
        FinalHoldoutGuard(tmp_path / "other.json", ledger=ledger).check_and_mark_used("retry")


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


def test_epsilon_greedy_refuses_inputs_that_would_log_a_wrong_propensity(episode):
    ohlcv, ep = episode
    rec = sc.run_shadow_episode(ohlcv, ep, sc.HeuristicPolicy(), sc.Budget(1), 5.0)
    step = rec.steps[0]
    with pytest.raises(ValueError, match="deterministic"):
        sc.EpsilonGreedyPolicy(base=sc.RandomPolicy(seed=1), epsilon=0.1)
    with pytest.raises(ValueError, match="duplicate"):
        sc.EpsilonGreedyPolicy(epsilon=0.1).act(step.observation, step.candidates + step.candidates[:1])

    class Wrong:
        name, version, deterministic = "wrong", "1", True

        def config(self):
            return {}

        def act(self, obs, candidates):
            return sc.Action(sc.ProposalSource.GRID, 999, {"fast_window": 1, "slow_window": 2}, 1.0)

    with pytest.raises(ValueError, match="outside the candidate menu"):
        sc.EpsilonGreedyPolicy(base=Wrong(), epsilon=0.1).act(step.observation, step.candidates)


def test_ope_validates_clip_and_sees_unlogged_target_mass():
    with pytest.raises(ValueError):
        ope.snips_value([1.0, 2.0], [0.5, 0.5], [1.0, 0.0], clip=-1.0)
    with pytest.raises(ValueError):
        ope.importance_weights([0.5], [1.0], clip=float("nan"))
    # The logger only ever took action "a"; the target puts all its mass on "b", which was never logged.
    assert ope.unsupported_mass_from_distributions({"a": 1.0}, {"b": 1.0}) == 1.0
    assert ope.unsupported_mass_from_distributions({"a": 0.5, "b": 0.5}, {"b": 1.0}) == 0.0


def test_exploration_is_logged_but_never_shown_to_the_proposer(monkeypatch, tmp_path):
    from src.agent import agent_graph
    from src.utils.config import config

    monkeypatch.setenv("AGENTQUANT_ASSISTANT_TOOLS", "1")
    monkeypatch.setenv("AGENTQUANT_IDEA_SIDECAR_EPSILON", "1.0")  # always explore
    monkeypatch.setattr(config, "results_db_path", str(tmp_path / "m.db"))
    state = {"as_of": "2024-01-31", "run_id": "r1", "asset": "SPY", "max_iterations": 3, "all_results": []}
    prompt = agent_graph._run_assistant_sidecar_context(
        state, "momentum", SimpleNamespace(regime_label="Bull"), 1)
    # The prompt carries the deterministic advisory (propensity 1.0).
    assert '"propensity": 1.0' in prompt
    logged = next(item for item in state["assistant_tool_results"] if item.get("tool_name") == "rl_shadow_advisory")
    data = logged["data"]
    n = len(data["candidates"])
    assert data["action"]["propensity"] == pytest.approx(1.0 / n)  # epsilon = 1: uniform over the menu
    assert data["observation"]["tried_grid_indices"] == []
    assert data["policy_name"] == "epsilon_greedy"
