"""Tests for the platform research workspace layer."""

import pandas as pd

from src.research.workspace import (
    build_research_memo,
    load_research_workspace,
    runs_to_dataframe,
    summarize_workspace,
)


def test_load_research_workspace_aggregates_csv_artifacts(tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()

    pd.DataFrame({
        "test_start": ["2024-01-01", "2024-07-01", "2025-01-01"],
        "test_end": ["2024-06-30", "2024-12-31", "2025-06-30"],
        "sharpe": [1.0, 0.5, -0.25],
        "return": [0.10, 0.04, -0.02],
        "drawdown": [0.04, 0.07, 0.10],
        "params": ["{}", "{}", "{}"],
    }).to_csv(exp_dir / "walk_forward_results.csv", index=False)

    pd.DataFrame({
        "strategy": ["Buy and Hold", "Golden Cross"],
        "sharpe": ["Ticker\nSPY    0.80\ndtype: float64", "0.55"],
        "return": ["Ticker\nSPY    0.25\ndtype: float64", "0.08"],
        "drawdown": ["Ticker\nSPY   -0.20\ndtype: float64", "0.05"],
    }).to_csv(exp_dir / "static_baseline_results.csv", index=False)

    runs = load_research_workspace(exp_dir, tmp_path / "missing.db")

    assert len(runs) == 3
    assert any(run.run_id == "wf-momentum" for run in runs)
    assert any(run.name == "Buy and Hold benchmark" for run in runs)

    walk_forward = next(run for run in runs if run.run_id == "wf-momentum")
    buy_hold = next(run for run in runs if run.name == "Buy and Hold benchmark")
    assert walk_forward.metrics["n_windows"] == 3
    assert walk_forward.validation_status == "pass"
    assert buy_hold.sharpe == 0.8
    assert buy_hold.total_return == 0.25


def test_workspace_summary_selects_best_robustness(tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()

    pd.DataFrame({
        "strategy": ["Low Drawdown", "High Sharpe"],
        "sharpe": [0.7, 0.9],
        "return": [0.08, 0.12],
        "drawdown": [0.03, 0.35],
    }).to_csv(exp_dir / "static_baseline_results.csv", index=False)

    runs = load_research_workspace(exp_dir, tmp_path / "missing.db")
    summary = summarize_workspace(runs)

    assert summary["run_count"] == 2
    assert summary["best_sharpe"] == 0.9
    assert summary["best_run"].name == "Low Drawdown benchmark"


def test_dataframe_and_memo_are_dashboard_ready(tmp_path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()

    pd.DataFrame({
        "type": ["With Context", "With Context", "No Context"],
        "sharpe": [0.4, 0.6, 0.2],
    }).to_csv(exp_dir / "ablation_results.csv", index=False)

    runs = load_research_workspace(exp_dir, tmp_path / "missing.db")
    df = runs_to_dataframe(runs)
    memo = build_research_memo(runs[0])

    assert {"Run ID", "Name", "Sharpe", "Robustness", "Validation"}.issubset(df.columns)
    assert "Validation Checks" in memo
    assert runs[0].name in memo
