"""Tests: sidecars."""
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.agent import idea_sidecar as sc
from src.agent import idea_sidecar as sidecar
from src.agent import ope, rl_shadow_tool, search_arms
from src.agent.agent_graph import _run_assistant_sidecar_context
from src.agent.assistant_tool_runner import AssistantToolRunner, AssistantToolRunRequest
from src.agent.capabilities import (
    AllocationRequest,
    CapabilityRegistry,
    Observation,
    ProposalRequest,
    RunBudget,
    RunContext,
    RunVisibility,
)
from src.agent.episode_splits import build_episodes, synthetic_ohlcv
from src.agent.parameter_grid import ParameterGrid
from src.agent.policy_mutation import FinalHoldoutGuard
from src.agent.sidecar_tools import (
    AssistantTool,
    SidecarToolRegistry,
    ToolBudget,
    ToolRequest,
    make_idea_sidecar_registry,
    register_hyde_retrieval,
    register_memory_search,
)
from src.research.alpha_store import FailureRecord


@pytest.fixture
def setup():
    ohlcv = synthetic_ohlcv(seed=7, n_days=600, asset="SIM")
    ep = build_episodes(ohlcv, "SIM", n_episodes=1, dev_days=300, holdout_days=60)[0]
    return ohlcv, ep


def _failure(ts, params=None):
    return FailureRecord(timestamp=ts, regime="x", strategy_type="momentum",
                         params=params or GRID[0], failure_mode="below_threshold")


GRID = ParameterGrid().get_grid("momentum")


@pytest.fixture
def episode():
    ohlcv = synthetic_ohlcv(seed=7, n_days=900, asset="SIM")
    return ohlcv, build_episodes(ohlcv, "SIM", n_episodes=2, dev_days=300, holdout_days=60)[1]


def _bandit(rng, n, epsilon):
    means = np.array([0.1, 0.3, 0.2, 0.5, 0.0])
    greedy = 1  # the logging policy's favourite arm
    probs = np.full(5, epsilon / 5)
    probs[greedy] += 1 - epsilon
    arms = rng.choice(5, size=n, p=probs)
    rewards = means[arms] + rng.normal(0, 0.3, n)
    return arms, rewards, probs[arms]


def _inputs():
    candidates = [
        sidecar.Candidate(sidecar.ProposalSource.RETRIEVED_PRIOR, 0, {"x": 1}),
        sidecar.Candidate(sidecar.ProposalSource.UNEXPLORED_REGION, 1, {"x": 2}),
    ]
    obs = sidecar.Observation(
        schema_version=sidecar.SCHEMA_VERSION,
        episode_id="fixture",
        decision_time="2024-01-01",
        step=0,
        asset="SIM",
        regime="unknown",
        strategy_family="fixture",
        grid_size=2,
        tried_grid_indices=(),
        visible_failures=(),
        attempted=(),
        remaining_proposals=1,
        remaining_tool_calls=0,
    )
    return obs, candidates


def test_assistant_sidecar_lookup_is_opt_in(monkeypatch):
    monkeypatch.delenv("AGENTQUANT_ASSISTANT_TOOLS", raising=False)
    state = {"as_of": "2024-01-31", "run_id": "r1", "asset": "SPY"}
    assert _run_assistant_sidecar_context(
        state, "momentum", SimpleNamespace(regime_label="Bull"), 1
    ) == ""
    assert "assistant_tool_results" not in state


def test_assistant_sidecar_lookup_runs_bounded_memory_tool(monkeypatch):
    monkeypatch.setenv("AGENTQUANT_ASSISTANT_TOOLS", "1")
    calls = []

    class FakeMemoryService:
        def graph_search(self, query, **kwargs):
            calls.append((query, kwargs))
            return []

    monkeypatch.setattr("src.agent.memory_tool.MemoryService", FakeMemoryService)
    state = {"as_of": "2024-01-31", "run_id": "r1", "asset": "SPY"}
    context = SimpleNamespace(regime_label="Bull")
    prompt_context = _run_assistant_sidecar_context(state, "momentum", context, 2)
    assert "Bounded read-only sidecar context" in prompt_context
    # The registry runs handlers in an isolated child, so observe the call
    # through its structured result rather than shared mutable test state.
    assert len(calls) == 0
    payload = state["assistant_tool_results"][0]
    assert payload["status"] == "ok"
    assert payload["selected_tools"] == ["search_memory_graph"]
    assert payload["authority"] == {
        "can_access_holdout": False, "can_promote": False, "can_evaluate": False
    }
    assert payload["results"][0]["provenance"]["source"] == "src.agent.memory_tool.MemorySearchTool"
    assert payload["results"][0]["data"]["cutoffs"]["market_as_of"] == "2024-01-31"
    assert {item["intent"] for item in state["assistant_tool_results"] if "intent" in item} >= {
        "memory_lookup", "candidate_options", "hypothesis_retrieval"
    }


def test_runner_discovers_selects_and_stamps_decision_time():
    registry = make_idea_sidecar_registry()
    runner = AssistantToolRunner(registry)
    result = runner.run(AssistantToolRunRequest(
        "run-1", "failure_lookup", "2024-01-10",
        {"failures": [
            {"failure_id": "old", "timestamp": "2024-01-09"},
            {"failure_id": "future", "timestamp": "2024-01-11"},
        ]},
    ))
    assert result.status == "ok"
    assert result.selected_tools == ("idea_visible_failures",)
    assert [f["failure_id"] for f in result.results[0].data] == ["old"]
    assert result.results[0].request_id == "run-1:1"
    assert {tool["name"] for tool in result.discovery} == {
        "idea_visible_failures", "idea_candidate_options"
    }
    assert result.to_dict()["authority"] == {
        "can_access_holdout": False, "can_promote": False, "can_evaluate": False
    }


def test_runner_records_isolated_tool_error_and_respects_per_run_limit():
    registry = make_idea_sidecar_registry(budget=ToolBudget(max_calls=1))
    runner = AssistantToolRunner(registry, {"two": (
        "idea_visible_failures", "idea_candidate_options")})
    result = runner.run(AssistantToolRunRequest(
        "run-2", "two", "2024-01-10",
        {"failures": []},
        max_tool_calls=2,
    ))
    assert result.status == "partial"
    assert len(result.attempts) == 2
    assert result.attempts[0].ok
    assert not result.attempts[1].ok
    assert result.attempts[1].error == "call_budget_exceeded"


def test_unknown_intent_does_not_invoke_tools():
    result = AssistantToolRunner(make_idea_sidecar_registry()).run(
        AssistantToolRunRequest("run-3", "unknown", "2024-01-10"))
    assert result.status == "no_matching_tool"
    assert not result.attempts


def test_reused_runner_has_independent_run_budgets():
    runner = AssistantToolRunner(make_idea_sidecar_registry(budget=ToolBudget(max_calls=1)))
    for index in range(3):
        result = runner.run(AssistantToolRunRequest(
            f"reuse-{index}", "failure_lookup", "2024-01-10", {"failures": []}))
        assert result.status == "ok"


def test_concurrent_runs_do_not_consume_each_others_budgets():
    from concurrent.futures import ThreadPoolExecutor

    runner = AssistantToolRunner(make_idea_sidecar_registry(budget=ToolBudget(max_calls=1)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: runner.run(AssistantToolRunRequest(
            f"parallel-{index}", "failure_lookup", "2024-01-10", {"failures": []})), range(8)))
    assert all(result.status == "ok" for result in results)


def test_runner_rejects_conflicting_decision_time():
    runner = AssistantToolRunner(make_idea_sidecar_registry())
    result = runner.run(AssistantToolRunRequest(
        "run-4", "failure_lookup", "2024-01-10",
        {"decision_time": "2024-01-11", "failures": []},
    ))
    assert result.status == "error"
    assert "match" in result.attempts[0].error


def test_hypothesis_tool_cannot_raise_knowledge_cutoff_past_decision_time():
    class Retriever:
        def run(self, request):
            raise AssertionError("future knowledge must be rejected before invocation")

    registry = make_idea_sidecar_registry()
    register_hyde_retrieval(registry, Retriever())
    result = AssistantToolRunner(registry).run(AssistantToolRunRequest(
        "run-5", "hypothesis_retrieval", "2024-01-10",
        {"regime": "trend", "strategy_types": ["momentum"],
         "knowledge_as_of": "2024-01-11"},
    ))
    assert result.status == "error"
    assert result.attempts[0].error == "request_rejected: knowledge_as_of cannot exceed decision_time"


def test_non_integer_tool_limit_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="non-negative integer"):
        AssistantToolRunRequest("run-6", "failure_lookup", "2024-01-10", max_tool_calls=1.5)


def test_selector_ranks_relevant_registered_tool_for_unmapped_intent():
    registry = make_idea_sidecar_registry()
    runner = AssistantToolRunner(registry)
    result = runner.run(AssistantToolRunRequest(
        "run-7", "candidate options", "2024-01-10",
        {"grid": [], "tried_grid_indices": [], "failures": [], "prior_params": []},
    ))
    assert result.selected_tools == ("idea_candidate_options",)
    assert result.selection_trace
    assert result.selection_trace[0]["score"] > 0
    assert "selection_trace" in result.to_dict()


def test_selector_abstains_and_injected_scorer_failure_falls_back():
    registry = make_idea_sidecar_registry()
    unknown = AssistantToolRunner(registry).run(AssistantToolRunRequest(
        "run-8", "weather forecast", "2024-01-10"))
    assert unknown.status == "no_matching_tool"
    assert not unknown.selected_tools

    def broken_scorer(request, tool):
        raise RuntimeError("scorer unavailable")

    fallback = AssistantToolRunner(registry, scorer=broken_scorer,
                                   scorer_version="local-test-v1").run(
        AssistantToolRunRequest("run-9", "failure_lookup", "2024-01-10",
                                {"failures": []}))
    assert fallback.selected_tools == ("idea_visible_failures",)
    assert "scorer_failure_fallback" in fallback.selection_trace[0]


def test_context_carries_budget_visibility_and_immutable_metadata():
    metadata = {"epoch": 2}
    context = RunContext(
        "run-1", "momentum", RunBudget(max_proposals=3, max_backtests=7),
        RunVisibility(allow_holdout_data=False), metadata,
    )
    metadata["epoch"] = 9

    assert context.budget.max_backtests == 7
    assert context.visibility.allow_holdout_data is False
    assert context.metadata["epoch"] == 2
    with pytest.raises(TypeError):
        context.metadata["epoch"] = 4
    with pytest.raises(FrozenInstanceError):
        context.run_id = "changed"


def test_requests_enforce_budget_limits():
    context = RunContext("r", "momentum", RunBudget(max_proposals=2, max_backtests=4))
    ProposalRequest(context, 2)
    AllocationRequest(context, [], 4)
    with pytest.raises(ValueError):
        ProposalRequest(context, 3)
    with pytest.raises(ValueError):
        AllocationRequest(context, [], 5)


def test_registry_is_separated_by_capability_and_rejects_duplicate_names():
    class Provider:
        def propose(self, request):
            return []

    registry = CapabilityRegistry()
    provider = Provider()
    assert registry.register_proposal_provider("grid", provider) is provider
    assert registry.proposal_provider("grid") is provider
    with pytest.raises(ValueError, match="already registered"):
        registry.register_proposal_provider("grid", Provider())
    with pytest.raises(KeyError, match="unknown capability"):
        registry.allocator("grid")


def test_observation_payload_is_immutable_snapshot():
    data = {"count": 1}
    observation = Observation("r", "proposals_created", data)
    data["count"] = 5

    assert observation.payload["count"] == 1
    with pytest.raises(TypeError):
        observation.payload["count"] = 2


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
    seen = []
    orig = search_arms._evaluate_params
    monkeypatch.setattr(search_arms, "_evaluate_params",
                        lambda df, p, c, **kw: (seen.append(df), orig(df, p, c, **kw))[1])
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
                            lambda df, p, c, n=n, **kw: (n.append(1), orig(df, p, c, **kw))[1])
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


def test_advisory_includes_hash_source_counts_and_propensity():
    obs, candidates = _inputs()
    result = rl_shadow_tool.advise(obs, candidates)
    payload = result.to_dict()
    assert payload["policy_hash"] == sidecar.policy_hash(sidecar.HeuristicPolicy())
    assert payload["considered_candidates"] == 2
    assert payload["source_allocation"] == {"retrieved_prior": 1, "unexplored_region": 1}
    assert payload["action"]["source"] == "retrieved_prior"
    assert payload["action"]["propensity"] == 1.0


def test_candidate_cap_applies_before_policy_call():
    obs, candidates = _inputs()
    result = rl_shadow_tool.advise(obs, candidates, max_candidates=1)
    assert result.considered_candidates == 1
    assert result.action["grid_index"] == 0


@pytest.mark.parametrize("cap", [0, -1, 257, 1.5, True])
def test_invalid_cap_rejected(cap):
    obs, candidates = _inputs()
    with pytest.raises(ValueError):
        rl_shadow_tool.advise(obs, candidates, max_candidates=cap)


def test_policy_action_outside_bounded_menu_is_rejected():
    obs, candidates = _inputs()

    class RoguePolicy:
        name, version = "rogue", "1"

        def config(self):
            return {}

        def act(self, observation, menu):
            return sidecar.Action(sidecar.ProposalSource.UNEXPLORED_REGION, 1, {"x": 2}, 1.0)

    with pytest.raises(sidecar.ActionOutOfSpace):
        rl_shadow_tool.advise(obs, candidates, RoguePolicy(), max_candidates=1)


def test_empty_candidate_menu_rejected():
    obs, _ = _inputs()
    with pytest.raises(ValueError, match="must not be empty"):
        rl_shadow_tool.advise(obs, [])


def test_idea_sidecar_tools_filter_future_failures_and_emit_provenance():
    registry = make_idea_sidecar_registry()
    result = registry.call(ToolRequest(
        request_id="r1", tool_name="idea_visible_failures",
        arguments={"decision_time": "2024-01-10", "failures": [
            {"failure_id": "past", "timestamp": "2024-01-09", "params": {"x": 1}},
            {"failure_id": "future", "timestamp": "2024-01-11", "params": {"x": 2}},
            {"failure_id": "unknown", "timestamp": "not-a-date", "params": {"x": 3}},
        ]}, capabilities=frozenset({"prior_results"}),
    ))
    assert result.ok
    assert [failure["failure_id"] for failure in result.data] == ["past"]
    assert result.provenance["source"] == "src.agent.idea_sidecar.filter_visible_failures"


def test_candidate_tool_is_advisory_and_returns_only_untried_grid_points():
    registry = make_idea_sidecar_registry()
    result = registry.call(ToolRequest(
        request_id="r2", tool_name="idea_candidate_options",
        arguments={"decision_time": "2024-01-10", "grid": [{"window": 2}, {"window": 4}],
                   "tried_grid_indices": [0], "failures": [], "prior_params": [{"window": 4}]},
        capabilities=frozenset({"prior_results"}),
    ))
    assert result.ok
    assert result.data == [{"source": "retrieved_prior", "grid_index": 1, "params": {"window": 4}}]
    assert "promote_candidate" not in result.provenance.get("capabilities_used", [])


def test_registry_isolates_contract_capability_and_handler_failures():
    registry = make_idea_sidecar_registry(budget=ToolBudget(max_calls=1))
    denied = registry.call(ToolRequest(
        "r1", "idea_visible_failures", {"decision_time": "2024-01-10", "failures": []}
    ))
    assert not denied.ok and denied.error == "missing_capability"
    bad = registry.call(ToolRequest(
        "r2", "idea_visible_failures", {"decision_time": 10, "failures": []},
        frozenset({"prior_results"}),
    ))
    assert not bad.ok and "field_type" in bad.error
    exhausted = registry.call(ToolRequest(
        "r3", "idea_visible_failures", {"decision_time": "2024-01-10", "failures": []},
        frozenset({"prior_results"}),
    ))
    assert exhausted.ok
    limited = registry.call(ToolRequest(
        "r4", "idea_visible_failures", {"decision_time": "2024-01-10", "failures": []},
        frozenset({"prior_results"}),
    ))
    assert not limited.ok and limited.error == "call_budget_exceeded"


def test_registry_rejects_holdout_and_promotion_tools():
    registry = SidecarToolRegistry()
    tool = AssistantTool("unsafe", "unsafe", {"type": "object"}, {"type": "object"},
                         frozenset({"holdout_data"}), lambda _: {}, "test")
    try:
        registry.register(tool)
    except ValueError as exc:
        assert "holdout" in str(exc) or "promotion" in str(exc)
    else:
        raise AssertionError("holdout access must be rejected")


def test_hyde_retrieval_registers_as_bounded_assistant_tool():
    from src.agent.hyde_retrieval import Evidence, HydeRetrievalTool, SyntheticHypothesis

    class Generator:
        def generate(self, request):
            return [SyntheticHypothesis("momentum breakout", "momentum", {"fast_window": 10})]

    registry = SidecarToolRegistry()
    register_hyde_retrieval(registry, HydeRetrievalTool(
        Generator(), [Evidence("e1", "momentum breakout", "fixture", "2020-01-01",
                               "2020-01-02", "momentum", {"fast_window": 10}, 0.8)],
    ))
    result = registry.call(ToolRequest(
        "hyde", "retrieve_hypothesis_evidence",
        {"regime": "bull", "strategy_types": ["momentum"], "as_of": "2021-01-01"},
        frozenset({"prior_results"}),
    ))
    assert result.ok
    assert result.data[0]["evaluation_status"] == "unevaluated"
    assert result.data[0]["evidence"][0]["evidence_id"] == "e1"
    assert result.provenance["source"] == "src.agent.hyde_retrieval.HydeRetrievalTool"


def test_memory_search_registers_with_cutoffs_and_read_only_result():
    from src.agent.memory_tool import MemorySearchTool

    class ReadOnlyMemory:
        calls = []

        def graph_search(self, query, **kwargs):
            self.calls.append((query, kwargs))
            return []

    memory = ReadOnlyMemory()
    registry = SidecarToolRegistry()
    register_memory_search(registry, MemorySearchTool(memory))
    result = registry.call(ToolRequest(
        "mem", "search_memory_graph",
        {"query": "momentum failure", "as_of": "2022-01-01", "knowledge_as_of": "2022-02-01",
         "strategy_type": "momentum", "top_k": 3},
        frozenset({"prior_results"}),
    ))
    assert result.ok and result.data["read_only"] is True
    assert result.data["promoted_to_belief"] is False
    assert memory.calls[0][1]["knowledge_as_of"] == "2022-02-01"
    assert result.provenance["source"] == "src.agent.memory_tool.MemorySearchTool"


def test_registry_uses_strict_json_and_cumulative_payload_limits():
    registry = make_idea_sidecar_registry(budget=ToolBudget(max_total_input_bytes=60))
    args = {"decision_time": "2024-01-10", "failures": []}
    first = registry.call(ToolRequest("one", "idea_visible_failures", args,
                                      frozenset({"prior_results"})))
    second = registry.call(ToolRequest("two", "idea_visible_failures", args,
                                       frozenset({"prior_results"})))
    assert first.ok
    assert not second.ok and second.error == "total_input_budget_exceeded"

    unsafe = SidecarToolRegistry()
    unsafe.register(AssistantTool("strict", "strict", {"type": "object"},
                                  {"type": "object"}, frozenset(),
                                  lambda _: {"bad": object()}, "fixture"))
    result = unsafe.call(ToolRequest("x", "strict", {}))
    assert not result.ok and "non_json_value" in result.error


def test_opt_in_process_isolation_enforces_handler_deadline():
    import time

    def slow(_):
        time.sleep(0.2)
        return {}

    registry = SidecarToolRegistry(ToolBudget(max_wall_seconds=0.03, isolate_process=True))
    registry.register(AssistantTool("slow", "slow", {"type": "object"},
                                    {"type": "object"}, frozenset(), slow, "fixture"))
    result = registry.call(ToolRequest("x", "slow", {}))
    assert not result.ok and "wall_budget_exceeded" in result.error


def test_dispatch_uses_json_snapshot_without_mutating_caller():
    def mutate(args):
        args["nested"]["values"].append(2)
        return args

    registry = SidecarToolRegistry()
    registry.register(AssistantTool("mutate", "fixture", {"type": "object"},
                                   {"type": "object"}, frozenset(), mutate, "fixture"))
    arguments = {"nested": {"values": [1]}}
    result = registry.call(ToolRequest("snapshot", "mutate", arguments))
    assert result.ok and result.data == {"nested": {"values": [1, 2]}}
    assert arguments == {"nested": {"values": [1]}}


def test_budget_rejects_non_finite_and_non_integer_limits():
    import pytest

    for value in (float("nan"), float("inf"), True, -1):
        with pytest.raises(ValueError):
            ToolBudget(max_wall_seconds=value)
    for value in (True, 1.5, -1):
        with pytest.raises(ValueError):
            ToolBudget(max_calls=value)


def test_session_snapshots_registrations_and_keeps_direct_call_budget():
    registry = make_idea_sidecar_registry(budget=ToolBudget(max_calls=1))
    session = registry.new_session()
    registry.register(AssistantTool("later", "fixture", {"type": "object"},
                                   {"type": "object"}, frozenset(), lambda _: {}, "fixture"))
    assert "later" not in {tool["name"] for tool in session.describe()}
    request = ToolRequest("direct", "idea_visible_failures",
                          {"decision_time": "2024-01-10", "failures": []},
                          frozenset({"prior_results"}))
    assert registry.call(request).ok
    assert registry.call(request).error == "call_budget_exceeded"
    assert session.call(request).ok
