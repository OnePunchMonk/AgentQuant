"""Tests: search loop."""
import os

import numpy as np
import pandas as pd
import pytest

from src.agent import search_arms
from src.agent.attempt_history import fold_attempt_history
from src.agent.base_planner import FallbackPlanner
from src.agent.context_builder import RegimeContext
from src.agent.parameter_grid import ParameterGrid
from src.agent.proposal_generator import Proposal, ProposalGenerator, ProposalValidator


def _fixture(seed=3, n=80):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(0.0002 + 0.01 * rng.standard_normal(n)))
    return {
        "DEMO": pd.DataFrame(
            {"Open": close * 0.998, "High": close * 1.005, "Low": close * 0.995,
             "Close": close, "Volume": 1_000_000},
            index=dates,
        )
    }


GRID = ParameterGrid().get_grid("momentum")


class _OffGridPlanner(FallbackPlanner):
    """Pretends to be an LLM that ignores the grid."""

    def is_available(self):
        return True

    def generate_proposals(self, prompt, n=5):
        return [
            {"fast_window": 3, "slow_window": 17, "confidence": 0.9},
            {"fast_window": 10, "slow_window": 30, "confidence": 0.8},
        ]


def _generator(tmp_path):
    from src.research.alpha_store import AlphaStore

    return ProposalGenerator(
        planner=FallbackPlanner(),
        alpha_store=AlphaStore(str(tmp_path / "a.db")),
        use_alpha_memory=False,
    )


def _state(proposals):
    return {
        "ohlcv_data": {"SPY": pd.DataFrame({"Close": [1.0, 2.0]})},
        "proposals": proposals,
        "iteration": 1,
        "run_log": [],
        "strategy_type": "momentum",
        "asset": "SPY",
        "trace": None,
    }


def test_folding_keeps_every_digest_and_expands_best_worst_and_recent():
    attempts = [
        {"params": {"fast_window": i}, "sharpe": score, "reasoning": f"try {i}"}
        for i, score in enumerate([0.1, -0.8, 0.7, 0.2, 0.3, 0.4])
    ]
    folded = fold_attempt_history(attempts, char_budget=3000, max_full_records=3)

    assert len(folded) == len(attempts)
    assert [record["attempt_index"] for record in folded] == list(range(1, 7))
    expanded = {record["attempt_index"] for record in folded if record.get("expanded_text")}
    assert expanded == {2, 3, 6}
    prompt = ProposalGenerator._format_prior_results(folded)
    assert "attempt 1:" in prompt and "attempt 6:" in prompt
    assert "try 2" in prompt and "try 4" not in prompt


def test_folding_empty_and_invalid_budget():
    assert fold_attempt_history([]) == []
    with pytest.raises(ValueError, match="non-negative"):
        fold_attempt_history([], char_budget=-1)


def test_folding_fails_instead_of_dropping_digests_when_catalog_exceeds_budget():
    attempts = [{"params": {"long": "x" * 100}, "sharpe": 0.1}]
    with pytest.raises(ValueError, match="exceeding budget"):
        fold_attempt_history(attempts, char_budget=10)


def test_agent_prompt_history_policy_is_opt_in():
    from src.agent.agent_graph import _prior_attempts_for_prompt

    attempts = [{"params": {"fast_window": 5}, "sharpe": 0.2}]
    recent = _prior_attempts_for_prompt({"all_results": attempts, "attempt_history_policy": "recent"})
    folded = _prior_attempts_for_prompt({"all_results": attempts, "attempt_history_policy": "folding"})

    assert recent == attempts
    assert folded[0]["attempt_index"] == 1
    assert folded[0]["digest"]


def test_fallback_without_api_key(monkeypatch):
    """Without an API key, generator should use grid search."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    gen = ProposalGenerator()
    ctx = RegimeContext(regime_label="LowVol-Bull")
    proposals = gen.generate(ctx, n_proposals=3, strategy_type="momentum")
    assert len(proposals) > 0
    assert all(isinstance(p, Proposal) for p in proposals)
    methods = {p.generation_method for p in proposals}
    assert "llm" not in methods  # no LLM key → grid_search or random


def test_proposals_have_valid_params(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    gen = ProposalGenerator()
    ctx = RegimeContext(regime_label="Crisis-Bear")
    proposals = gen.generate(ctx, n_proposals=5, strategy_type="momentum")
    for p in proposals:
        assert "fast_window" in p.params
        assert "slow_window" in p.params
        assert p.params["fast_window"] < p.params["slow_window"]


def test_proposals_count(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    gen = ProposalGenerator()
    ctx = RegimeContext(regime_label="MidVol-Neutral")
    proposals = gen.generate(ctx, n_proposals=4, strategy_type="momentum")
    assert len(proposals) == 4


def test_offline_default_strips_any_present_llm_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-should-be-stripped")
    captured = {}

    def fake_run_agent(data, **kwargs):
        captured["anthropic_key_present"] = "ANTHROPIC_API_KEY" in os.environ
        captured["offline_flag"] = os.environ.get("AGENTQUANT_OFFLINE")
        return {"all_results": [], "best_result": None, "trace": None}

    monkeypatch.setattr("src.agent.agent_graph.run_agent", fake_run_agent)

    search_arms._run_agent_offline(
        _fixture(), "DEMO", seed=1, memory_db_path=str(tmp_path / "m.db"), live=False,
    )
    assert captured["anthropic_key_present"] is False
    assert captured["offline_flag"] == "1"


def test_live_mode_does_not_strip_keys_and_unsets_offline_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-should-survive")
    monkeypatch.setenv("AGENTQUANT_OFFLINE", "1")
    captured = {}

    def fake_run_agent(data, **kwargs):
        captured["anthropic_key_present"] = "ANTHROPIC_API_KEY" in os.environ
        captured["offline_flag"] = os.environ.get("AGENTQUANT_OFFLINE")
        return {"all_results": [], "best_result": None, "trace": None}

    monkeypatch.setattr("src.agent.agent_graph.run_agent", fake_run_agent)

    search_arms._run_agent_offline(
        _fixture(), "DEMO", seed=1, memory_db_path=str(tmp_path / "m.db"), live=True,
    )
    assert captured["anthropic_key_present"] is True
    assert captured["offline_flag"] is None


def test_live_mode_without_any_key_fails_loudly_instead_of_falling_back(tmp_path):
    with pytest.raises(RuntimeError, match="requires at least one of"):
        search_arms._run_agent_offline(
            _fixture(), "DEMO", seed=1, memory_db_path=str(tmp_path / "m.db"), live=True,
        )


def test_validator_rejects_off_grid_momentum():
    assert ProposalValidator.validate({"fast_window": 3, "slow_window": 17}, "momentum", GRID) is None
    assert ProposalValidator.validate({"fast_window": 7, "slow_window": 999}, "momentum", GRID) is None


def test_validator_accepts_on_grid_momentum_even_with_float_ints():
    proposal = ProposalValidator.validate({"fast_window": 10.0, "slow_window": 30.0}, "momentum", GRID)
    assert proposal is not None
    assert proposal.params == {"fast_window": 10, "slow_window": 30}


def test_validator_without_grid_keeps_old_behaviour():
    assert ProposalValidator.validate({"fast_window": 3, "slow_window": 17}, "momentum") is not None


def test_generator_never_returns_off_grid_params(tmp_path):
    from src.research.alpha_store import AlphaStore

    gen = ProposalGenerator(
        planner=_OffGridPlanner(),
        alpha_store=AlphaStore(str(tmp_path / "a.db")),
        use_alpha_memory=False,
    )
    proposals = gen.generate(RegimeContext(regime_label="LowVol-Bull"), n_proposals=5)
    assert proposals
    assert all(p.params in GRID for p in proposals)
    assert {"fast_window": 3, "slow_window": 17} not in [p.params for p in proposals]


@pytest.mark.parametrize("label", ["LowVol-Bull", "Crisis-Bear", "MidVol-Neutral"])
def test_generate_excludes_already_tried_configs(tmp_path, label):
    gen = _generator(tmp_path)
    ctx = RegimeContext(regime_label=label)
    tried = []
    seen = set()
    for _ in range(2):
        batch = gen.generate(ctx, n_proposals=5, strategy_type="momentum", exclude_params=tried)
        keys = {tuple(sorted(p.params.items())) for p in batch}
        assert len(batch) == 5
        assert not (keys & seen), "a retry re-proposed a config that was already evaluated"
        seen |= keys
        tried = [dict(k) for k in seen]
    assert len(seen) == 10  # the two rounds covered the whole 10-point grid
    assert gen.generate(ctx, n_proposals=5, strategy_type="momentum", exclude_params=tried) == []


def test_best_result_is_best_across_iterations(monkeypatch):
    import src.backtest.runner as runner
    from src.agent import agent_graph

    sharpe_by_fast = {5: 1.5, 10: 0.2}

    def fake_run_backtest(ohlcv, assets, strategy, params, **kwargs):
        return {
            "equity_curve": None,
            "metrics": {"sharpe_ratio": sharpe_by_fast[params["fast_window"]], "total_return": 0.1,
                        "max_drawdown": 0.05, "calmar": 1.0, "sortino": 1.0, "num_trades": 3},
        }

    monkeypatch.setattr(runner, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(agent_graph, "_record_trials", lambda *a, **k: None)

    state = _state([Proposal(params={"fast_window": 5, "slow_window": 20})])
    state = agent_graph.backtest_node(state)
    assert state["best_result"]["sharpe"] == 1.5

    state["iteration"] = 2
    state["proposals"] = [Proposal(params={"fast_window": 10, "slow_window": 30})]
    state = agent_graph.backtest_node(state)
    # The worse second iteration must not replace the earlier winner.
    assert state["best_result"]["sharpe"] == 1.5
    assert state["best_result"]["params"] == {"fast_window": 5, "slow_window": 20}


def test_hypothesize_stops_when_grid_is_exhausted(monkeypatch, tmp_path):
    from src.agent import agent_graph
    from src.utils.config import config

    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENTQUANT_OFFLINE", "1")
    monkeypatch.setattr(config, "results_db_path", str(tmp_path / "mem.db"))

    best = {"params": GRID[0], "sharpe": 0.1}
    state = _state([])
    state["context"] = RegimeContext(regime_label="LowVol-Bull")
    state["all_results"] = [{"params": dict(g), "sharpe": 0.1} for g in GRID]
    state["best_result"] = best
    state["should_continue"] = True

    state = agent_graph.hypothesize_node(state)
    assert state["space_exhausted"] is True
    assert state["iteration"] == 1  # the empty round is not counted as an iteration
    assert state["should_continue"] is False
    assert state["run_status"] == agent_graph.STATUS_BUDGET_EXHAUSTED
    assert state["best_result"] is best


def test_offline_arm_restores_environment(monkeypatch):
    import os

    from src.agent import agent_graph, search_arms

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k" * 24)
    monkeypatch.setenv("TAVILY_API_KEY", "t" * 24)
    monkeypatch.delenv("AGENTQUANT_OFFLINE", raising=False)

    seen_inside = {}

    def fake_run_agent(*args, **kwargs):
        seen_inside["anthropic"] = os.environ.get("ANTHROPIC_API_KEY")
        seen_inside["offline"] = os.environ.get("AGENTQUANT_OFFLINE")
        return {}

    monkeypatch.setattr(agent_graph, "run_agent", fake_run_agent)
    search_arms._run_agent_offline({}, "SIM", 0, "unused.db", live=False)

    # Inside the call the agent is forced offline...
    assert seen_inside == {"anthropic": None, "offline": "1"}
    # ...and afterwards the caller's environment is back.
    assert os.environ.get("ANTHROPIC_API_KEY") == "k" * 24
    assert os.environ.get("TAVILY_API_KEY") == "t" * 24
    assert "AGENTQUANT_OFFLINE" not in os.environ


def test_tool_path_parser_rejects_off_grid_proposals():
    from src.agent.tools.orchestrator import _parse_proposals_from_text

    text = '[{"fast_window": 3, "slow_window": 17}, {"fast_window": 10, "slow_window": 30, "confidence": 0.7}]'
    parsed = _parse_proposals_from_text(text, "momentum")
    assert [p.params for p in parsed] == [{"fast_window": 10, "slow_window": 30}]
