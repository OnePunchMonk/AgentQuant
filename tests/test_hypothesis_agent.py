import csv

from scripts.benchmark_hypothesis_techniques import (
    HISTORICAL_REGIMES,
    REGIMES,
    load_historical_prices,
    run_benchmark,
    run_historical_benchmark,
)
from src.agent.generators import (
    AnalyticalGenerator,
    EnsembleGenerator,
    HydeGenerator,
    LiteratureGenerator,
)
from src.agent.hyde_retrieval import Evidence, HypothesisRequest
from src.agent.hypothesis_agent import HypothesisAgent


def test_agent_runs_all_generator_contracts_with_cutoffs():
    evidence = [
        Evidence(
            "visible",
            "bull_trend momentum trend study",
            "fixture",
            "2024-01-01",
            "2024-01-02",
            "momentum",
            {"fast_window": 5, "slow_window": 20},
            0.8,
        ),
        Evidence(
            "late",
            "bull_trend momentum trend study",
            "fixture",
            "2024-01-01",
            "2025-01-02",
            "momentum",
            {"fast_window": 5, "slow_window": 20},
            0.8,
        ),
    ]
    literature = LiteratureGenerator(evidence)
    generators = {
        "literature": literature,
        "hyde": HydeGenerator(),
        "analytical": AnalyticalGenerator(),
        "ensemble": EnsembleGenerator((literature, HydeGenerator(), AnalyticalGenerator())),
    }
    agent = HypothesisAgent(generators, evidence)
    request = HypothesisRequest("bull_trend", ("momentum",), "2024-12-31", 3, 2)
    for method in generators:
        run = agent.run(method, request)
        assert run.method == method
        assert run.regime == "bull_trend"
        assert all(c.evaluation_status == "unevaluated" for c in run.candidates)
        assert all(all(e.evidence_id != "late" for e in c.evidence) for c in run.candidates)


def test_benchmark_is_reproducible_and_labels_fixture_only():
    a, b = run_benchmark(seed=13), run_benchmark(seed=13)
    assert a == b
    assert a["regime_count"] == 6
    assert a["claim_scope"].startswith("synthetic deterministic fixtures only")
    assert set(a["results"][0]["metrics"]) == {"literature", "hyde", "analytical", "ensemble"}
    for regime in REGIMES:
        record = next(r for r in a["results"] if r["regime"] == regime)
        assert record["data_label"] == "synthetic_fixture_not_real_historical_data"
        for metrics in record["metrics"].values():
            assert "fixture_sharpe_mean" in metrics
            assert "fixture_sharpe_gap" in metrics
            assert "novelty_rate" in metrics
            assert "false_positive_rate" in metrics
            assert "cost_units" in metrics
            assert "learning_curve_fixture_sharpe" in metrics


def test_unknown_method_rejected():
    agent = HypothesisAgent({}, [])
    request = HypothesisRequest("bull", ("momentum",), "2024-12-31")
    try:
        agent.run("unknown", request)
    except ValueError as exc:
        assert "unknown generation method" in str(exc)
    else:
        raise AssertionError("unknown generator should fail")


def test_historical_mode_uses_prices_and_fixed_dates_and_budgets(tmp_path):
    path = tmp_path / "prices.csv"
    # Build enough actual observations for all fixed windows; the test asserts
    # plumbing and causal split behavior, not any expected financial outcome.
    import pandas as pd

    dates = pd.bdate_range("2000-01-01", "2025-01-01")
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Adj Close"])
        for i, day in enumerate(dates):
            writer.writerow([day.date(), 100 + i * 0.02 + (i % 31) * 0.1])
    loaded, provenance = load_historical_prices(path)
    assert len(loaded) > 1000
    assert provenance["provider"] == "user_csv"
    result = run_historical_benchmark(csv_path=path, candidate_budget=2)
    assert result["data_label"] == "public_historical_daily_prices"
    assert result["regime_count"] == 8
    assert len(result["results"]) == len(HISTORICAL_REGIMES)
    for record in result["results"]:
        assert record["date_start"] <= record["split_date"] < record["date_end"]
        assert record["holdout_observations"] > 0
        for metrics in record["metrics"].values():
            assert metrics["budget"] == 2
            assert metrics["candidate_count"] <= 2
            assert metrics["cost_bps_per_unit_turnover"] == 5.0
            assert all("train" in c and "holdout" in c for c in metrics["candidates"])
