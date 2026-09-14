"""Tests for the P1 fair search benchmark: leakage sentinels, future-dated
memory, and missing-outcome handling."""

import json
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Mutation arms (#31): evidence-conditioned mutation vs. random vs.
# shuffled-evidence controls, wired into the fair benchmark.
# ---------------------------------------------------------------------------

def test_derive_structured_failure_reads_prior_episode_drawdown():
    from src.agent.search_arms import EpisodeResult, derive_structured_failure

    prior = EpisodeResult(
        arm="x", episode_id="ep00", seed=1, candidates=[], winner_params=None,
        holdout_net_return=0.05, holdout_max_drawdown=0.35, holdout_turnover=0.1,
        tool_calls=0, wall_time_s=0.0, status="ok",
    )
    failures = derive_structured_failure(prior)
    assert len(failures) == 1
    assert failures[0].tag == "high_drawdown"
    assert "0.35" in failures[0].evidence


def test_derive_structured_failure_empty_for_no_prior_episode():
    from src.agent.search_arms import derive_structured_failure

    assert derive_structured_failure(None) == []


def test_derive_structured_failure_missing_grading_yields_no_evidence():
    from src.agent.search_arms import EpisodeResult, derive_structured_failure

    missing = EpisodeResult(
        arm="x", episode_id="ep00", seed=1, candidates=[], winner_params=None,
        holdout_net_return="missing", holdout_max_drawdown="missing", holdout_turnover="missing",
        tool_calls=0, wall_time_s=0.0, status="missing",
    )
    assert derive_structured_failure(missing) == []


def test_evidence_conditioned_arm_runs_and_carries_mutated_policy_forward():
    from src.agent.harness_config import harness_v1_base
    from src.agent.search_arms import EpisodeResult, run_mutation_agent_arm

    ohlcv = synthetic_ohlcv(seed=3, n_days=400, asset="SIM")
    episodes = build_episodes(ohlcv, "SIM", n_episodes=1, dev_days=200, holdout_days=60)
    ep = episodes[0]
    incumbent = harness_v1_base()

    prior = EpisodeResult(
        arm="x", episode_id="ep_prior", seed=1, candidates=[], winner_params=None,
        holdout_net_return=0.05, holdout_max_drawdown=0.40, holdout_turnover=0.1,
        tool_calls=0, wall_time_s=0.0, status="ok",
    )
    result, record, child = run_mutation_agent_arm(
        ohlcv, ep, seed=1, cost_bps=5.0, incumbent=incumbent, prior_result=prior,
        mode="evidence_conditioned",
    )
    assert result.arm == "evidence_conditioned_mutation"
    assert "high_drawdown" in record["diagnosis"]
    assert child.prompt_template == "tool_aware_tuned_v2_learnings"
    assert child.prompt_template != incumbent.prompt_template


def test_random_mode_ignores_prior_result():
    """The random-mutation arm's diagnosis must never reference the prior
    episode's failure -- it's a control precisely because it ignores it."""
    from src.agent.harness_config import harness_v1_base
    from src.agent.search_arms import EpisodeResult, run_mutation_agent_arm

    ohlcv = synthetic_ohlcv(seed=3, n_days=400, asset="SIM")
    episodes = build_episodes(ohlcv, "SIM", n_episodes=1, dev_days=200, holdout_days=60)
    ep = episodes[0]
    incumbent = harness_v1_base()
    prior = EpisodeResult(
        arm="x", episode_id="ep_prior", seed=1, candidates=[], winner_params=None,
        holdout_net_return=0.05, holdout_max_drawdown=0.40, holdout_turnover=0.1,
        tool_calls=0, wall_time_s=0.0, status="ok",
    )
    _, record, _ = run_mutation_agent_arm(
        ohlcv, ep, seed=1, cost_bps=5.0, incumbent=incumbent, prior_result=prior, mode="random",
    )
    assert "random_baseline" in record["diagnosis"]
    assert "high_drawdown" not in record["diagnosis"]


def test_shuffled_evidence_mode_uses_shuffle_source_not_true_prior():
    """The shuffled-evidence control must derive its failure tag from
    `shuffle_source_result`, not `prior_result` -- otherwise it's
    indistinguishable from evidence_conditioned."""
    from src.agent.harness_config import harness_v1_base
    from src.agent.search_arms import EpisodeResult, run_mutation_agent_arm

    ohlcv = synthetic_ohlcv(seed=3, n_days=400, asset="SIM")
    episodes = build_episodes(ohlcv, "SIM", n_episodes=1, dev_days=200, holdout_days=60)
    ep = episodes[0]
    incumbent = harness_v1_base()

    true_prior = EpisodeResult(  # high drawdown -- must NOT drive the mutation
        arm="x", episode_id="ep_true_prior", seed=1, candidates=[], winner_params=None,
        holdout_net_return=0.05, holdout_max_drawdown=0.40, holdout_turnover=0.1,
        tool_calls=0, wall_time_s=0.0, status="ok",
    )
    shuffle_source = EpisodeResult(  # within bounds -- SHOULD drive the mutation
        arm="x", episode_id="ep_wrong_episode", seed=1, candidates=[], winner_params=None,
        holdout_net_return=0.02, holdout_max_drawdown=0.05, holdout_turnover=0.1,
        tool_calls=0, wall_time_s=0.0, status="ok",
    )
    _, record, child = run_mutation_agent_arm(
        ohlcv, ep, seed=1, cost_bps=5.0, incumbent=incumbent, prior_result=true_prior,
        mode="shuffled_evidence", shuffle_source_result=shuffle_source,
    )
    assert "unspecified" in record["diagnosis"] or "no non-no-op" in record["diagnosis"]
    assert "high_drawdown" not in record["diagnosis"]
    # unspecified's mapped patch (grid_search_default) equals the incumbent's
    # own template, so it falls through to the first distinct pool entry.
    assert child.prompt_template == "tool_aware_default"


def test_offline_benchmark_run_documents_that_mutation_arms_collapse_to_frozen_agent(tmp_path, monkeypatch):
    """End-to-end honesty check for #31 acceptance: with no LLM key set, the
    mutated prompt_template never reaches a live proposal backend (offline
    FallbackPlanner ignores it), so the mutation arms MUST match
    frozen_agent numerically -- and the report MUST say so explicitly rather
    than let identical numbers pass as a silent null result."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fair_search_benchmark

    out = tmp_path / "bench.json"
    monkeypatch.setattr(sys, "argv", [
        "fair_search_benchmark.py", "--episodes", "1", "--seeds", "1", "2", "3",
        "--max-iterations", "1", "--output", str(out),
        "--splits-path", str(tmp_path / "splits.json"),
    ])
    fair_search_benchmark.main()

    report = json.loads(out.read_text())
    assert "mutation_arms_caveat" in report
    assert "does not masquerade" in report["mutation_arms_caveat"]

    frozen = report["arms"]["frozen_agent"]["summary"]["net_return_mean"]
    for arm in ("evidence_conditioned_mutation", "random_mutation", "shuffled_evidence_mutation"):
        assert report["arms"][arm]["summary"]["net_return_mean"] == frozen

    # Offline: mutation never reaches the proposal backend, so every paired
    # delta must be exactly zero with a CI that (degenerately) pins at zero.
    for comparison in report["paired_comparisons"].values():
        assert comparison["mean_delta"] == 0.0
        assert comparison["n_dropped_missing_coverage"] == 0


# ---------------------------------------------------------------------------
# Paired-uncertainty comparison
# ---------------------------------------------------------------------------

def test_paired_comparison_matches_by_episode_and_seed_not_position():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fair_search_benchmark

    # Deliberately out-of-order / mismatched-length rows: pairing must be by
    # (episode_id, seed), not list position.
    rows_a = [
        {"episode_id": "ep00", "seed": 1, "status": "ok", "holdout_net_return": 0.10},
        {"episode_id": "ep01", "seed": 1, "status": "ok", "holdout_net_return": 0.20},
    ]
    rows_b = [
        {"episode_id": "ep01", "seed": 1, "status": "ok", "holdout_net_return": 0.25},
        {"episode_id": "ep00", "seed": 1, "status": "ok", "holdout_net_return": 0.12},
    ]
    result = fair_search_benchmark._paired_comparison(rows_a, rows_b)
    assert result["n_paired"] == 2
    # deltas: ep00 0.12-0.10=0.02, ep01 0.25-0.20=0.05 -> mean 0.035
    assert abs(result["mean_delta"] - 0.035) < 1e-9
    assert result["ci95_low"] <= result["mean_delta"] <= result["ci95_high"]


def test_paired_comparison_drops_cells_missing_on_either_side():
    rows_a = [
        {"episode_id": "ep00", "seed": 1, "status": "ok", "holdout_net_return": 0.10},
        {"episode_id": "ep01", "seed": 1, "status": "missing", "holdout_net_return": "missing"},
    ]
    rows_b = [
        {"episode_id": "ep00", "seed": 1, "status": "ok", "holdout_net_return": 0.15},
        {"episode_id": "ep01", "seed": 1, "status": "ok", "holdout_net_return": 0.50},
    ]
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fair_search_benchmark

    result = fair_search_benchmark._paired_comparison(rows_a, rows_b)
    assert result["n_paired"] == 1
    assert result["n_dropped_missing_coverage"] == 1


def test_paired_comparison_deterministic_ci_across_runs():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fair_search_benchmark

    rows_a = [{"episode_id": f"ep{i:02d}", "seed": 1, "status": "ok", "holdout_net_return": 0.1 + 0.01 * i}
              for i in range(10)]
    rows_b = [{"episode_id": f"ep{i:02d}", "seed": 1, "status": "ok", "holdout_net_return": 0.12 + 0.01 * i}
              for i in range(10)]
    r1 = fair_search_benchmark._paired_comparison(rows_a, rows_b)
    r2 = fair_search_benchmark._paired_comparison(rows_a, rows_b)
    assert r1 == r2
