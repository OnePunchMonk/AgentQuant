"""Tests: evaluator."""
import numpy as np
import pandas as pd
import pytest

from src.agent import agent_graph, search_arms
from src.agent.agent_graph import _split_search_and_holdout
from src.agent.episode_splits import build_episodes, synthetic_ohlcv
from src.agent.parameter_grid import ParameterGrid
from src.backtest.core import common_warmup_bars, evaluate_window
from src.backtest.runner import run_backtest
from src.utils.config import config

GRID = ParameterGrid().get_grid("momentum")


def _runner_cost_bps() -> float:
    b = config.backtest
    return (b.commission + b.slippage) * 1e4 + b.market_impact_bps


def _episodes(n=3):
    ohlcv = synthetic_ohlcv(seed=5, n_days=900, asset="SIM")
    return ohlcv, build_episodes(ohlcv, "SIM", n_episodes=n, dev_days=220, holdout_days=60)


def _frame(n):
    idx = pd.bdate_range("2018-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0003, 0.01, n))), index=idx)
    return pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1e6})


def _make_ohlcv(n=500):
    idx = pd.bdate_range("2022-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0 + pd.Series(range(n), index=idx) * 0.01,
            "Volume": 1_000_000,
        },
        index=idx,
    )
    return {"SPY": df, "^VIX": df.copy()}


@pytest.mark.parametrize("params", GRID[:6])
def test_runner_and_benchmark_evaluator_agree(params):
    ohlcv, _ = _episodes()
    df = ohlcv["SIM"]
    eval_start = df.index[400]
    arms = search_arms._evaluate_params(df, params, _runner_cost_bps(), eval_start=eval_start)
    runner = run_backtest({"SIM": df}, ["SIM"], "momentum", params, eval_start=eval_start)["metrics"]
    assert arms["sharpe"] == pytest.approx(runner["sharpe_ratio"], abs=1e-9)
    assert arms["net_return"] == pytest.approx(runner["total_return"], abs=1e-9)
    assert arms["max_drawdown"] == pytest.approx(runner["max_drawdown"], abs=1e-9)


def test_history_beyond_the_warmup_does_not_change_the_score():
    ohlcv, _ = _episodes()
    df = ohlcv["SIM"]
    params = {"fast_window": 20, "slow_window": 60}
    start = df.index[500]
    short_history = df.loc[df.index >= df.index[400]]  # 100 bars of history before `start`
    long_history = df  # 500 bars of history
    a = evaluate_window(short_history, "momentum", params, 0.0005, eval_start=start)
    b = evaluate_window(long_history, "momentum", params, 0.0005, eval_start=start)
    assert a["sharpe"] == pytest.approx(b["sharpe"], abs=1e-12)
    assert a["net_return"] == pytest.approx(b["net_return"], abs=1e-12)


def test_holdout_is_scored_with_history_not_on_the_slice_alone():
    ohlcv, episodes = _episodes()
    ep = episodes[1]  # has data before its dev window
    params = {"fast_window": 50, "slow_window": 150}  # slow window longer than the 60-bar holdout
    holdout_df = ohlcv["SIM"].loc[ep.holdout_start:ep.holdout_end]
    history = search_arms._history_upto(ohlcv, ep, ep.holdout_end)
    assert len(holdout_df) == 60 and len(history) > 150

    expected = evaluate_window(history, "momentum", params, 5.0 / 1e4, eval_start=ep.holdout_start)
    graded = search_arms._grade_on_holdout(
        "grid_search", ep, 1, [], params, holdout_df, 5.0, 0, 0.0, history_df=history,
    )
    slice_only = search_arms._grade_on_holdout("grid_search", ep, 1, [], params, holdout_df, 5.0, 0, 0.0)

    assert graded.status == "ok"
    assert graded.holdout_net_return == pytest.approx(expected["net_return"], abs=1e-12)
    # On the slice alone a 150-bar average has no history, so the strategy is flat.
    assert slice_only.holdout_net_return == 0.0
    assert graded.holdout_net_return != 0.0


def test_dev_scoring_gets_history_but_never_future_rows(monkeypatch):
    ohlcv, episodes = _episodes()
    ep = episodes[1]
    calls = []
    original = search_arms._evaluate_params

    def spy(df, params, cost_bps, **kwargs):
        calls.append((df, kwargs))
        return original(df, params, cost_bps, **kwargs)

    monkeypatch.setattr(search_arms, "_evaluate_params", spy)
    search_arms.run_grid_search_arm(ohlcv, ep, seed=1, cost_bps=5.0)

    dev_calls = calls[:-1]  # the last call is the sealed holdout grading
    assert dev_calls
    for df, kwargs in dev_calls:
        assert df.index.max() <= pd.Timestamp(ep.dev_end)
        assert df.index.min() < pd.Timestamp(ep.dev_start)  # history before the window
        assert pd.Timestamp(kwargs["eval_start"]) == pd.Timestamp(ep.dev_start)


def test_random_search_arm_stays_on_the_canonical_grid():
    ohlcv, episodes = _episodes()
    result = search_arms.run_random_search_arm(ohlcv, episodes[1], seed=3, cost_bps=5.0)
    assert result.candidates
    assert all(c.params in GRID for c in result.candidates)
    assert len({tuple(sorted(c.params.items())) for c in result.candidates}) == len(result.candidates)


def test_common_warmup_uses_the_longest_grid_window():
    assert common_warmup_bars(GRID) == 253  # longest window plus one bar
    assert common_warmup_bars(GRID, n_bars=1000) == 253
    assert common_warmup_bars(GRID, n_bars=500) == 0  # would eat more than 40% of the window


def test_common_eval_start_only_when_the_window_can_afford_it():
    long_df, short_df = _frame(1000), _frame(300)
    assert agent_graph._common_eval_start({"SPY": long_df}, "SPY", "momentum") == long_df.index[253]
    assert agent_graph._common_eval_start({"SPY": short_df}, "SPY", "momentum") is None


def test_backtest_node_scores_every_candidate_from_the_same_start(monkeypatch):
    import src.backtest.runner as runner
    from src.agent.proposal_generator import Proposal

    seen = []

    def fake_run_backtest(ohlcv, assets, strategy, params, **kwargs):
        seen.append(kwargs.get("eval_start"))
        return {"equity_curve": None, "metrics": {"sharpe_ratio": 0.5, "total_return": 0.1,
                                                   "max_drawdown": 0.05, "num_trades": 2}}

    monkeypatch.setattr(runner, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(agent_graph, "_record_trials", lambda *a, **k: None)
    df = _frame(1000)
    state = {
        "ohlcv_data": {"SPY": df}, "iteration": 1, "run_log": [], "strategy_type": "momentum",
        "asset": "SPY", "trace": None,
        "proposals": [Proposal(params=GRID[0]), Proposal(params=GRID[-1])],
    }
    agent_graph.backtest_node(state)
    assert seen == [df.index[253], df.index[253]]


def test_whole_frame_scoring_matches_the_runner_without_a_base_bar():
    ohlcv, _ = _episodes()
    df = ohlcv["SIM"].iloc[:300]
    params = {"fast_window": 10, "slow_window": 50}
    arms = search_arms._evaluate_params(df, params, _runner_cost_bps())
    runner = run_backtest({"SIM": df}, ["SIM"], "momentum", params)["metrics"]
    assert arms["sharpe"] == pytest.approx(runner["sharpe_ratio"], abs=1e-9)
    assert arms["net_return"] == pytest.approx(runner["total_return"], abs=1e-9)


def test_offline_agent_arms_charge_the_same_total_cost_as_the_other_arms(monkeypatch):
    from src.agent import agent_graph

    seen = {}

    def fake_run_agent(*args, **kwargs):
        b = config.backtest
        seen["total_bps"] = (b.commission + b.slippage) * 1e4 + b.market_impact_bps
        return {}

    monkeypatch.setattr(agent_graph, "run_agent", fake_run_agent)
    before = (config.backtest.commission, config.backtest.slippage, config.backtest.market_impact_bps)
    search_arms._run_agent_offline({}, "SIM", 0, "unused.db", cost_bps=5.0)
    assert seen["total_bps"] == pytest.approx(5.0)
    assert (config.backtest.commission, config.backtest.slippage, config.backtest.market_impact_bps) == before


def test_split_reserves_trailing_fraction():
    ohlcv = _make_ohlcv(500)
    search, holdout_start = _split_search_and_holdout(ohlcv, "SPY", 0.2)

    assert holdout_start is not None
    assert (search["SPY"].index < holdout_start).all()
    # Roughly 80% of history remains for search.
    assert 350 <= len(search["SPY"]) <= 410
    # Holdout data (>= holdout_start) is excluded from every ticker, not
    # just the primary asset.
    assert (search["^VIX"].index < holdout_start).all()





@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.0, 1.5])  # 0.0 means disabled
def test_split_disabled_or_invalid_fraction_is_noop(fraction):
    ohlcv = _make_ohlcv(500)
    search, holdout_start = _split_search_and_holdout(ohlcv, "SPY", fraction)

    assert holdout_start is None
    assert search is ohlcv


def test_split_short_history_is_noop():
    ohlcv = _make_ohlcv(3)
    search, holdout_start = _split_search_and_holdout(ohlcv, "SPY", 0.2)

    # Not enough bars to carve out a non-trivial search window.
    assert holdout_start is None
    assert search is ohlcv


def test_split_missing_asset_is_noop():
    ohlcv = _make_ohlcv(500)
    search, holdout_start = _split_search_and_holdout(ohlcv, "MISSING", 0.2)

    assert holdout_start is None
    assert search is ohlcv
