from scripts.benchmark_tool_decision_protocol import run_benchmark


def test_tool_communication_benchmark_is_deterministic_and_exercises_all_modes():
    first = run_benchmark()
    second = run_benchmark()
    assert first["model_quality_claim"] is False
    assert first["training_fixture_count"] == 3
    assert first["frozen_evaluation_fixture_count"] == 5
    assert first["frozen_tool_candidates"] == ["lookup_facts", "lookup_status"]
    assert {row["mode"] for row in first["rows"]} == {
        "json_one_shot", "compact_one_shot", "json_refined", "compact_refined"
    }
    for mode in ("json_refined", "compact_refined"):
        summary = first["summaries"][mode]
        assert summary["selection_valid_rate"] == 1.0
        assert summary["argument_valid_rate"] == 0.8
        assert summary["execution_success_rate"] == 0.8
        assert summary["result_integrity_rate"] == 1.0
        assert summary["expected_result_rate"] == 0.8
    one_shot = [r for r in first["rows"] if r["mode"].endswith("one_shot")]
    for mode in ("json_one_shot", "compact_one_shot"):
        summary = first["summaries"][mode]
        assert summary["selection_valid_rate"] == 0.4
        assert summary["expected_result_rate"] == 0.2
    assert any(not row["arguments_valid"] for row in one_shot)
    assert any(not row["selected_tool_valid"] for row in one_shot)
    unverified = [r for r in first["rows"] if r["fixture_id"] == "eval_05" and r["mode"].endswith("refined")]
    assert all(not row["verified"] and row["registry_calls"] == 0 for row in unverified)
    if first["token_counts_available"]:
        assert first["tokenizer"] == "tiktoken:cl100k_base"
        assert all(isinstance(row["token_count"], int) for row in first["rows"])
    else:
        assert all(row["token_count"] is None for row in first["rows"])
    assert first["summaries"]["compact_one_shot"]["mean_total_bytes"] > 0
    assert first["summaries"]["json_one_shot"]["mean_total_bytes"] > 0
    assert [r["fixture_id"] for r in first["rows"]] == [r["fixture_id"] for r in second["rows"]]
    assert [r["total_bytes"] for r in first["rows"]] == [r["total_bytes"] for r in second["rows"]]
