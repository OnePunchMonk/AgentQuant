"""Regression tests for the correctness fixes found in the code sweep.

Each test pins one bug that was reproduced before it was fixed:
  1. bootstrap Sharpe was non-deterministic (unseeded RNG)
  2. num_trades truncated an enter-once-and-hold signal to 0
  3. LLM proposals were not required to be points on the canonical grid
  4. retries re-proposed the exact configs already tested; best_result was the
     latest iteration's best instead of the best across iterations
  5. an offline benchmark call permanently deleted API keys from os.environ
"""

import numpy as np
import pandas as pd
import pytest

from src.agent.base_planner import FallbackPlanner
from src.agent.context_builder import RegimeContext
from src.agent.parameter_grid import ParameterGrid
from src.agent.proposal_generator import Proposal, ProposalGenerator, ProposalValidator
from src.backtest.metrics import PerformanceMetrics
from src.backtest.runner import run_backtest


# ---------------------------------------------------------------- 1. bootstrap
def test_bootstrap_sharpe_is_deterministic():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0004, 0.01, 400))
    first = PerformanceMetrics.bootstrap_sharpe(returns)
    second = PerformanceMetrics.bootstrap_sharpe(returns)
    assert first == second


def test_bootstrap_sharpe_explicit_seed_controls_result():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.0004, 0.01, 400))
    a = PerformanceMetrics.bootstrap_sharpe(returns, seed=1)
    b = PerformanceMetrics.bootstrap_sharpe(returns, seed=1)
    c = PerformanceMetrics.bootstrap_sharpe(returns, seed=2)
    assert a == b
    assert a != c


# ---------------------------------------------------------------- 2. num_trades
def test_num_trades_counts_enter_once_and_hold():
    idx = pd.bdate_range("2022-01-03", periods=80)
    close = pd.Series(np.linspace(100, 160, 80), index=idx)  # steady uptrend
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6})
    result = run_backtest({"X": df}, ["X"], "momentum", {"fast_window": 3, "slow_window": 10})
    # Flat during warmup, then long for the rest of the window: one position change.
    assert result["metrics"]["num_trades"] == 1


# ---------------------------------------------------------------- 3. grid membership
GRID = ParameterGrid().get_grid("momentum")


def test_validator_rejects_off_grid_momentum():
    assert ProposalValidator.validate({"fast_window": 3, "slow_window": 17}, "momentum", GRID) is None
    assert ProposalValidator.validate({"fast_window": 7, "slow_window": 999}, "momentum", GRID) is None


def test_validator_accepts_on_grid_momentum_even_with_float_ints():
    proposal = ProposalValidator.validate({"fast_window": 10.0, "slow_window": 30.0}, "momentum", GRID)
    assert proposal is not None
    assert proposal.params == {"fast_window": 10, "slow_window": 30}


def test_validator_without_grid_keeps_old_behaviour():
    assert ProposalValidator.validate({"fast_window": 3, "slow_window": 17}, "momentum") is not None


class _OffGridPlanner(FallbackPlanner):
    """Pretends to be an LLM that ignores the grid."""

    def is_available(self):
        return True

    def generate_proposals(self, prompt, n=5):
        return [
            {"fast_window": 3, "slow_window": 17, "confidence": 0.9},
            {"fast_window": 10, "slow_window": 30, "confidence": 0.8},
        ]


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


# ---------------------------------------------------------------- 4a. no repeats
def _generator(tmp_path):
    from src.research.alpha_store import AlphaStore

    return ProposalGenerator(
        planner=FallbackPlanner(),
        alpha_store=AlphaStore(str(tmp_path / "a.db")),
        use_alpha_memory=False,
    )


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


# ---------------------------------------------------------------- 4b/4c. agent loop
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


# ---------------------------------------------------------------- 5. environment
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


# ---------------------------------------------------------------- tool path is grid-checked too
def test_tool_path_parser_rejects_off_grid_proposals():
    from src.agent.tools.orchestrator import _parse_proposals_from_text

    text = '[{"fast_window": 3, "slow_window": 17}, {"fast_window": 10, "slow_window": 30, "confidence": 0.7}]'
    parsed = _parse_proposals_from_text(text, "momentum")
    assert [p.params for p in parsed] == [{"fast_window": 10, "slow_window": 30}]


# ---------------------------------------------------------------- num_trades unit
def test_num_trades_keeps_round_trip_unit():
    idx = pd.bdate_range("2022-01-03", periods=120)
    up = np.linspace(100, 140, 60)
    down = np.linspace(140, 100, 60)
    close = pd.Series(np.concatenate([up, down]), index=idx)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6})
    result = run_backtest({"X": df}, ["X"], "momentum", {"fast_window": 3, "slow_window": 10})
    # long after warmup (1 unit), then a flip to short when the trend reverses (2 units):
    # 3 units of turnover = 2 round trips under the historical unit (ceil(3 / 2)).
    assert result["metrics"]["num_trades"] == 2
