"""Tests for the P1 fair search benchmark: leakage sentinels, future-dated
memory, and missing-outcome handling."""

import pandas as pd

from src.agent.episode_splits import (
    apply_transaction_costs,
    build_episodes,
    slice_dev,
    slice_holdout,
    synthetic_ohlcv,
)
from src.agent.search_arms import (
    Candidate,
)


def _episodes():
    ohlcv = synthetic_ohlcv(seed=1, n_days=400, asset="SIM")
    episodes = build_episodes(ohlcv, "SIM", n_episodes=1, dev_days=200, holdout_days=60)
    assert episodes
    return ohlcv, episodes[0]


def test_dev_slice_never_contains_holdout_rows():
    """Leakage sentinel: the dev-window slice handed to a proposal/selection
    step must not contain any row at or after the holdout start."""
    ohlcv, ep = _episodes()
    dev = slice_dev(ohlcv, ep)["SIM"]
    assert (dev.index <= pd.Timestamp(ep.dev_end)).all()
    assert (dev.index < pd.Timestamp(ep.holdout_start)).all()


def test_holdout_slice_never_contains_dev_rows():
    ohlcv, ep = _episodes()
    holdout = slice_holdout(ohlcv, ep)["SIM"]
    assert (holdout.index >= pd.Timestamp(ep.holdout_start)).all()


def test_grid_search_proposal_step_never_receives_holdout_rows(monkeypatch):
    """The grid-search arm's evaluation step (the "proposal" path) must only
    ever be called with rows inside the dev window."""
    ohlcv, ep = _episodes()
    dev_cutoff = pd.Timestamp(ep.dev_end)

    from src.agent import search_arms

    original = search_arms._evaluate_params
    seen_frames = []

    def spy(df, params, cost_bps):
        seen_frames.append(df)
        return original(df, params, cost_bps)

    monkeypatch.setattr(search_arms, "_evaluate_params", spy)
    search_arms.run_grid_search_arm(ohlcv, ep, seed=1, cost_bps=5.0)

    # The first N-1 calls are the dev-window search calls; the LAST call is
    # the sealed holdout grading call. Every call except the final grading
    # call must be within the dev window.
    assert len(seen_frames) >= 2
    for df in seen_frames[:-1]:
        assert (df.index <= dev_cutoff).all(), "search step saw data past the dev cutoff"


def test_missing_outcome_reported_as_missing_not_coerced():
    """A failed/empty evaluation must surface as status='missing', never as
    a fallback numeric score like 0.0."""
    ohlcv, ep = _episodes()
    empty_holdout = {"SIM": ohlcv["SIM"].iloc[0:0]}  # simulate a grading failure: no holdout rows

    from src.agent import search_arms

    res = search_arms._grade_on_holdout(
        "grid_search", ep, seed=1,
        candidates=[Candidate(params={"fast_window": 5, "slow_window": 20}, dev_sharpe=0.4,
                               status="ok", generation_method="grid_search")],
        winner_params={"fast_window": 5, "slow_window": 20},
        holdout_df=empty_holdout["SIM"], cost_bps=5.0, tool_calls=0, started=0.0,
    )
    assert res.status == "missing"
    d = res.to_dict()
    assert d["holdout_net_return"] == "missing"
    assert d["holdout_max_drawdown"] == "missing"


def test_missing_outcome_when_no_winner_selected():
    """If every candidate fails, there's no winner_params, and grading must
    report missing rather than a fallback."""
    ohlcv, ep = _episodes()
    holdout_df = ohlcv["SIM"].iloc[-10:]
    from src.agent import search_arms

    res = search_arms._grade_on_holdout(
        "random_search", ep, seed=1, candidates=[], winner_params=None,
        holdout_df=holdout_df, cost_bps=5.0, tool_calls=0, started=0.0,
    )
    assert res.status == "missing"


def test_success_rate_denominator_counts_failures():
    """The benchmark's success-rate-style stat must count failed attempts
    in the denominator, not just successes."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from scripts.fair_search_benchmark import _summarize

    rows = [
        {"status": "ok", "holdout_net_return": 0.1, "holdout_max_drawdown": 0.05,
         "holdout_turnover": 2.0, "n_candidates": 3, "n_failed": 0, "tool_calls": 0, "wall_time_s": 0.01},
        {"status": "missing", "holdout_net_return": "missing", "holdout_max_drawdown": "missing",
         "holdout_turnover": "missing", "n_candidates": 3, "n_failed": 3, "tool_calls": 0, "wall_time_s": 0.01},
    ]
    summary = _summarize(rows)
    assert summary["n_episode_seed_runs"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["n_missing"] == 1


def test_buy_and_hold_charges_entry_trade_cost():
    """The fixed/buy-and-hold arm's initial position entry must be treated
    as a trade and charged transaction cost: turnover must be nonzero (at
    least the one entry trade), and changing cost_bps must change net
    return -- previously the entry trade was never charged because the
    first bar's position diff was filled with 0 instead of treated as
    entering from flat."""
    idx = pd.bdate_range("2020-01-01", periods=30)
    signal = pd.Series(1.0, index=idx)  # always in market, no further trades
    returns = pd.Series(0.001, index=idx)

    zero_cost = apply_transaction_costs(returns, signal, cost_bps=0.0)
    high_cost = apply_transaction_costs(returns, signal, cost_bps=100.0)

    assert zero_cost["turnover"] > 0
    assert high_cost["net_returns"].sum() < zero_cost["net_returns"].sum()


def test_fixed_arm_turnover_is_nonzero_and_cost_sensitive():
    from src.agent.search_arms import run_fixed_arm

    ohlcv = synthetic_ohlcv(seed=2, n_days=400, asset="SIM")
    episodes = build_episodes(ohlcv, "SIM", n_episodes=1, dev_days=200, holdout_days=60)
    ep = episodes[0]

    low = run_fixed_arm(ohlcv, ep, seed=1, cost_bps=0.0)
    high = run_fixed_arm(ohlcv, ep, seed=1, cost_bps=200.0)

    assert low.holdout_turnover > 0
    assert low.holdout_net_return != high.holdout_net_return


def test_transaction_cost_reduces_returns_when_trading():
    idx = pd.bdate_range("2020-01-01", periods=50)
    signal = pd.Series([i % 2 for i in range(50)], index=idx, dtype=float)  # flips every bar
    returns = pd.Series(0.01, index=idx)
    costed = apply_transaction_costs(returns, signal, cost_bps=50.0)
    assert costed["net_returns"].sum() < returns.sum()
    assert costed["turnover"] > 0
