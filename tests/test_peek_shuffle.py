import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

import peek  # noqa: E402


def _df_with_real_signal(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3 * x1 - 2 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "x1": x1,
        "x2": x2,
        "target": y,
    })


def _df_with_no_signal(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "target": rng.normal(size=n),
    })


def test_shuffle_passes_when_real_signal_exists():
    df = _df_with_real_signal()
    report = peek.audit(
        df, time_col="date", target="target",
        pipeline=LinearRegression(), cv=KFold(n_splits=5), scorer=r2_score,
    )
    shuffle_findings = [f for f in report.findings if f.check == "shuffle"]
    assert shuffle_findings
    assert shuffle_findings[0].severity.value == "PASS"


def test_shuffle_flags_when_no_real_signal():
    df = _df_with_no_signal()
    report = peek.audit(
        df, time_col="date", target="target",
        pipeline=LinearRegression(), cv=KFold(n_splits=5), scorer=r2_score,
    )
    shuffle_findings = [f for f in report.findings if f.check == "shuffle"]
    assert shuffle_findings
    assert shuffle_findings[0].severity.value == "CRITICAL"


def test_shuffle_only_runs_with_full_pipeline_cv_scorer():
    df = _df_with_real_signal(n=50)
    report = peek.audit(df, time_col="date", target="target", pipeline=LinearRegression())
    assert "shuffle" not in report.checks_run
