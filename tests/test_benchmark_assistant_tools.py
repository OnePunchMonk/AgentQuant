import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("assistant_tool_benchmark", ROOT / "scripts" / "benchmark_assistant_tools.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_frozen_fixture_digest_and_suite_size():
    fixture = benchmark._fixture()
    assert fixture["manifest"]["version"] == benchmark.MANIFEST_VERSION
    assert len(fixture["tasks"]) == 6


def test_benchmark_reports_routing_quality_cost_and_replay():
    result = benchmark.run_benchmark()
    variants = result["variants"]
    assert set(variants) == {"no_tools", "fixed_intent", "selector", "memory_only", "hyde_only", "rl_shadow_only", "all_tools"}
    assert variants["fixed_intent"]["quality_rate"] == 1
    assert variants["no_tools"]["quality_rate"] == 0
    assert variants["no_tools"]["abstain_rate"] == 1
    assert variants["all_tools"]["calls_per_task"] == 3
    assert variants["fixed_intent"]["calls_per_task"] == 1
    assert all(row["replay_parity"] for row in variants.values())
    assert variants["all_tools"]["synthetic_cost_units"] > variants["fixed_intent"]["synthetic_cost_units"]


def test_selector_is_a_transparent_deterministic_baseline():
    fixture = benchmark._fixture()
    task = fixture["tasks"][0]
    result = benchmark._run_real(task, "selector")
    assert result["selected_tools"] == ["search_memory_graph"]
    assert result["selection_trace"]
    assert result["results"][0]["data"] == {"evidence_source": "memory"}


def test_replay_parity_uses_serialized_real_runner_results():
    fixture = benchmark._fixture()
    task = fixture["tasks"][1]
    first = benchmark._run_real(task, "selector")
    second = benchmark._run_real(task, "selector")
    assert benchmark._canonical(first) == benchmark._canonical(second)
    assert first["results"][0]["provenance"]["elapsed_seconds"] >= 0
