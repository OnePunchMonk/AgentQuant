from scripts.benchmark_hyde_retrieval import evaluate, run_benchmark


def test_benchmark_has_two_regimes_common_evidence_ceiling_and_scoped_metrics():
    result = run_benchmark(k=4, seed=7)
    assert len(result["results"]) == 2
    assert "no backtest utility measured" in result["claim_scope"]
    for regime in result["results"]:
        assert regime["candidate_budget_ceiling"] == 4
        assert regime["metrics"]["hyde"]["candidate_count"] == 1
        assert regime["evidence_budget_per_candidate"] == 4
        assert regime["evidence_tier"] == "fixture_relevance_label"
        assert set(regime["metrics"]) == {
            "hyde", "literature_keyword", "grid_order", "shuffled_evidence", "no_memory"
        }
        assert regime["hyde_candidate"]["evaluation_status"] == "unevaluated"


def test_time_cutoff_and_shuffled_no_memory_ablations_are_explicit():
    result = evaluate("trend", k=5)
    assert result["time_cutoff_excluded_relevant"] == ["trend-05", "trend-late-known"]
    assert result["metrics"]["no_memory"]["relevant_retrieved"] == 0
    assert result["metrics"]["shuffled_evidence"]["candidate_count"] == 5
