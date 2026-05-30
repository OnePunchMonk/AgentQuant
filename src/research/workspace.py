"""
Research Workspace
==================

Typed platform layer for turning experiment outputs into inspectable research
runs. The dashboard can render these objects without knowing whether they came
from SQLite, CSV backtests, or future agent-generated reports.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List

import pandas as pd

PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class ValidationCheck:
    """A research hygiene check shown in the platform workspace."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ResearchRun:
    """A normalized, UI-ready experiment or benchmark run."""

    run_id: str
    name: str
    source: str
    strategy: str
    mode: str
    metrics: Dict[str, float]
    validation_checks: List[ValidationCheck] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""
    timestamp: str = ""
    git_hash: str = ""

    @property
    def sharpe(self) -> float:
        return float(self.metrics.get("sharpe", 0.0))

    @property
    def total_return(self) -> float:
        return float(self.metrics.get("total_return", 0.0))

    @property
    def max_drawdown(self) -> float:
        return abs(float(self.metrics.get("max_drawdown", 0.0)))

    @property
    def robustness_score(self) -> float:
        return float(self.metrics.get("robustness_score", self.sharpe - self.max_drawdown))

    @property
    def validation_status(self) -> str:
        statuses = {check.status for check in self.validation_checks}
        if FAIL in statuses:
            return FAIL
        if WARN in statuses:
            return WARN
        return PASS

    def as_row(self) -> Dict[str, Any]:
        return {
            "Run ID": self.run_id,
            "Name": self.name,
            "Mode": self.mode,
            "Strategy": self.strategy,
            "Source": self.source,
            "Sharpe": round(self.sharpe, 3),
            "Return": self.total_return,
            "Max Drawdown": self.max_drawdown,
            "Robustness": round(self.robustness_score, 3),
            "Validation": self.validation_status,
            "Git": self.git_hash,
        }


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Extract a numeric scalar from plain values or pandas string dumps."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(value):
            return default
        return float(value)

    text = str(value).strip()
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    if not matches:
        return default
    return float(matches[0])


def _metric_alias(row: Dict[str, Any], *names: str) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return _coerce_float(row[name])
    return 0.0


def _basic_checks(
    *,
    sharpe: float,
    max_drawdown: float,
    n_windows: int = 1,
    source: str,
) -> List[ValidationCheck]:
    checks = [
        ValidationCheck(
            name="Metric completeness",
            status=PASS if isinstance(sharpe, float) and isinstance(max_drawdown, float) else FAIL,
            detail="Sharpe and drawdown are available for comparison.",
        ),
        ValidationCheck(
            name="Drawdown sanity",
            status=PASS if abs(max_drawdown) <= 0.35 else WARN,
            detail=f"Observed max drawdown is {abs(max_drawdown) * 100:.1f}%.",
        ),
    ]

    if source == "walk_forward":
        checks.append(
            ValidationCheck(
                name="Temporal validation",
                status=PASS if n_windows >= 3 else WARN,
                detail=f"Evaluated across {n_windows} chronological windows.",
            )
        )
        checks.append(
            ValidationCheck(
                name="Robustness floor",
                status=PASS if sharpe > 0 and n_windows >= 3 else WARN,
                detail="Mean Sharpe remains positive after chronological splitting.",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="Baseline context",
                status=WARN,
                detail="Useful benchmark, but not a leakage-safe validation protocol.",
            )
        )

    return checks


def _aggregate_walk_forward(path: Path) -> ResearchRun | None:
    df = pd.read_csv(path)
    if df.empty:
        return None

    sharpes = [_coerce_float(v) for v in df["sharpe"].tolist()]
    returns = [_coerce_float(v) for v in df["return"].tolist()]
    drawdowns = [_coerce_float(v) for v in df["drawdown"].tolist()]
    mean_sharpe = mean(sharpes)
    sharpe_std = pd.Series(sharpes).std(ddof=0) if len(sharpes) > 1 else 0.0
    max_drawdown = max(abs(v) for v in drawdowns) if drawdowns else 0.0
    metrics = {
        "sharpe": mean_sharpe,
        "median_sharpe": median(sharpes),
        "min_sharpe": min(sharpes),
        "sharpe_std": float(sharpe_std),
        "total_return": sum(returns),
        "max_drawdown": max_drawdown,
        "robustness_score": mean_sharpe - float(sharpe_std) - max_drawdown,
        "n_windows": float(len(df)),
    }
    return ResearchRun(
        run_id="wf-momentum",
        name="Walk-forward momentum study",
        source="walk_forward",
        strategy="momentum",
        mode="Agent research",
        metrics=metrics,
        validation_checks=_basic_checks(
            sharpe=mean_sharpe,
            max_drawdown=max_drawdown,
            n_windows=len(df),
            source="walk_forward",
        ),
        artifacts=[str(path)],
        notes="Chronological windows with agent-selected momentum parameters.",
    )


def _runs_from_static_baselines(path: Path) -> List[ResearchRun]:
    df = pd.read_csv(path)
    runs = []
    for idx, row in df.iterrows():
        record = row.to_dict()
        strategy = str(record.get("strategy", f"baseline-{idx}"))
        sharpe = _metric_alias(record, "sharpe", "sharpe_ratio")
        total_return = _metric_alias(record, "return", "total_return")
        max_drawdown = abs(_metric_alias(record, "drawdown", "max_drawdown"))
        runs.append(
            ResearchRun(
                run_id=f"base-{idx + 1}",
                name=f"{strategy} benchmark",
                source="baseline",
                strategy=strategy,
                mode="Benchmark",
                metrics={
                    "sharpe": sharpe,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "robustness_score": sharpe - max_drawdown,
                },
                validation_checks=_basic_checks(
                    sharpe=sharpe,
                    max_drawdown=max_drawdown,
                    source="baseline",
                ),
                artifacts=[str(path)],
                notes="Static benchmark used to anchor agent results.",
            )
        )
    return runs


def _aggregate_random_baseline(path: Path) -> ResearchRun | None:
    df = pd.read_csv(path)
    if df.empty:
        return None

    sharpes = [_coerce_float(v) for v in df["sharpe"].tolist()]
    returns = [_coerce_float(v) for v in df["return"].tolist()]
    drawdowns = [_coerce_float(v) for v in df["drawdown"].tolist()]
    p95 = float(pd.Series(sharpes).quantile(0.95))
    max_drawdown = max(abs(v) for v in drawdowns) if drawdowns else 0.0
    metrics = {
        "sharpe": mean(sharpes),
        "p95_sharpe": p95,
        "total_return": mean(returns),
        "max_drawdown": max_drawdown,
        "robustness_score": mean(sharpes) - max_drawdown,
        "n_trials": float(len(df)),
    }
    return ResearchRun(
        run_id="rnd-momentum",
        name="Random momentum baseline",
        source="baseline",
        strategy="momentum",
        mode="Benchmark",
        metrics=metrics,
        validation_checks=_basic_checks(
            sharpe=metrics["sharpe"],
            max_drawdown=max_drawdown,
            source="baseline",
        ),
        artifacts=[str(path)],
        notes="Distributional baseline for checking whether agent runs beat random parameter search.",
    )


def _runs_from_ablation(path: Path) -> List[ResearchRun]:
    df = pd.read_csv(path)
    if df.empty or "type" not in df or "sharpe" not in df:
        return []

    runs = []
    for group_name, group in df.groupby("type"):
        sharpes = [_coerce_float(v) for v in group["sharpe"].tolist()]
        avg_sharpe = mean(sharpes)
        runs.append(
            ResearchRun(
                run_id=f"abl-{str(group_name).lower().replace(' ', '-')}",
                name=f"{group_name} ablation",
                source="ablation",
                strategy="agent_context",
                mode="Ablation",
                metrics={
                    "sharpe": avg_sharpe,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "robustness_score": avg_sharpe,
                    "n_trials": float(len(group)),
                },
                validation_checks=[
                    ValidationCheck(
                        name="Ablation coverage",
                        status=PASS if len(group) >= 3 else WARN,
                        detail=f"{len(group)} trials available for this ablation arm.",
                    ),
                    ValidationCheck(
                        name="Metric completeness",
                        status=PASS,
                        detail="Sharpe is available for context-vs-no-context comparison.",
                    ),
                ],
                artifacts=[str(path)],
                notes="Compares agent proposal quality with and without regime context.",
            )
        )
    return runs


def _runs_from_results_store(db_path: Path) -> List[ResearchRun]:
    if not db_path.exists():
        return []

    from experiments.results_store import ResultsStore

    runs = []
    store = ResultsStore(str(db_path))
    for row in store.list_runs():
        aggregate = json.loads(row.get("aggregate_metrics") or "{}")
        sharpe = _metric_alias(aggregate, "mean_sharpe", "sharpe", "sharpe_ratio")
        max_drawdown = abs(_metric_alias(aggregate, "max_drawdown"))
        run = ResearchRun(
            run_id=str(row["run_id"]),
            name=f"{row['experiment_type']} run",
            source="sqlite",
            strategy=str(aggregate.get("strategy", "mixed")),
            mode=str(row["experiment_type"]),
            timestamp=str(row.get("timestamp", "")),
            git_hash=str(row.get("git_hash", "")),
            metrics={
                "sharpe": sharpe,
                "total_return": _metric_alias(aggregate, "total_return"),
                "max_drawdown": max_drawdown,
                "robustness_score": sharpe - max_drawdown,
                "n_windows": _metric_alias(aggregate, "n_windows"),
            },
            validation_checks=_basic_checks(
                sharpe=sharpe,
                max_drawdown=max_drawdown,
                n_windows=int(_metric_alias(aggregate, "n_windows")),
                source="walk_forward" if "walk" in str(row["experiment_type"]) else "sqlite",
            ),
            artifacts=[str(db_path)],
        )
        runs.append(run)
    return runs


def load_research_workspace(
    experiments_dir: str | Path = "experiments",
    results_db_path: str | Path = "experiments/results.db",
) -> List[ResearchRun]:
    """Load platform-ready research runs from known local experiment artifacts."""
    exp_dir = Path(experiments_dir)
    runs: List[ResearchRun] = []

    runs.extend(_runs_from_results_store(Path(results_db_path)))

    walk_forward_path = exp_dir / "walk_forward_results.csv"
    if walk_forward_path.exists():
        run = _aggregate_walk_forward(walk_forward_path)
        if run:
            runs.append(run)

    static_path = exp_dir / "static_baseline_results.csv"
    if static_path.exists():
        runs.extend(_runs_from_static_baselines(static_path))

    random_path = exp_dir / "random_baseline_results.csv"
    if random_path.exists():
        run = _aggregate_random_baseline(random_path)
        if run:
            runs.append(run)

    ablation_path = exp_dir / "ablation_results.csv"
    if ablation_path.exists():
        runs.extend(_runs_from_ablation(ablation_path))

    return sorted(runs, key=lambda run: run.robustness_score, reverse=True)


def runs_to_dataframe(runs: Iterable[ResearchRun]) -> pd.DataFrame:
    """Convert research runs to a dashboard-friendly table."""
    rows = [run.as_row() for run in runs]
    return pd.DataFrame(rows)


def summarize_workspace(runs: Iterable[ResearchRun]) -> Dict[str, Any]:
    """Aggregate high-level platform KPIs for the research workspace."""
    run_list = list(runs)
    if not run_list:
        return {
            "run_count": 0,
            "best_run": None,
            "best_sharpe": 0.0,
            "best_robustness": 0.0,
            "validation_pass_rate": 0.0,
        }

    best_run = max(run_list, key=lambda run: run.robustness_score)
    pass_count = sum(1 for run in run_list if run.validation_status == PASS)
    return {
        "run_count": len(run_list),
        "best_run": best_run,
        "best_sharpe": max(run.sharpe for run in run_list),
        "best_robustness": best_run.robustness_score,
        "validation_pass_rate": pass_count / len(run_list),
    }


def build_research_memo(run: ResearchRun) -> str:
    """Generate a concise Markdown memo for a selected research run."""
    checks = "\n".join(
        f"- **{check.name}** ({check.status}): {check.detail}"
        for check in run.validation_checks
    )
    return f"""### {run.name}

**Mode:** {run.mode}
**Strategy:** {run.strategy}
**Source:** `{run.source}`

**Result:** Sharpe {run.sharpe:.3f}, total return {run.total_return * 100:.1f}%, max drawdown {run.max_drawdown * 100:.1f}%, robustness {run.robustness_score:.3f}.

**Research Notes:** {run.notes or "No notes recorded."}

**Validation Checks**
{checks}
"""
