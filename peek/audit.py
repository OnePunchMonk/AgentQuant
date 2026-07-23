"""
Peek Audit Orchestrator
========================

`audit()` is the single public entry point. It builds an `AuditContext` from
whatever the caller supplies, runs every check that applies given the
available inputs, and returns an `AuditReport`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from peek.checks import ALL_CHECKS, AuditContext
from peek.report import AuditReport


def audit(
    df: pd.DataFrame,
    time_col: str,
    target: str,
    *,
    horizon: int = 1,
    feature_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    splits: Optional[list] = None,
    splitter: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    cv: Optional[Any] = None,
    scorer: Optional[Callable[..., float]] = None,
    embargo: int = 0,
) -> AuditReport:
    """
    Audit a time-series dataset (and optionally a feature function, CV split,
    or full pipeline) for look-ahead bias and data leakage.

    Always runs (needs only df/time_col/target):
        - target_leak: flags features that are near-exact copies of the
          (possibly shifted) target.

    Runs when `feature_fn` is given:
        - causality: the flagship check. Recomputes features on truncated
          data and compares against the full computation to prove a feature
          only used information available at the time.

    Runs when `splits` or `splitter` is given:
        - split: flags train/test temporal overlap and missing embargo gaps.

    Runs when `pipeline`, `cv`, and `scorer` are all given:
        - shuffle: permutation sanity test comparing the real score against
          scores achievable on randomly shuffled labels.

    Args:
        df: the full time-ordered dataset.
        time_col: name of the timestamp/ordering column.
        target: name of the label column.
        horizon: forecast horizon in rows, used by the target_leak check.
        feature_fn: callable(df) -> DataFrame of features, recomputed under
            truncation by the causality check.
        splits: an explicit list of (train_idx, test_idx) index arrays.
        splitter: a sklearn-style object exposing `.split(df)`.
        pipeline: a fit/predict object, cloned per fold by the shuffle check.
        cv: a sklearn-style splitter used by the shuffle check.
        scorer: callable(y_true, y_pred) -> float, higher is better.
        embargo: minimum gap required between train and test windows.

    Returns:
        AuditReport with a `.verdict` of "LEAKING", "SUSPICIOUS", or "CLEAN".
    """
    ctx = AuditContext(
        df=df,
        time_col=time_col,
        target=target,
        horizon=horizon,
        feature_fn=feature_fn,
        splits=splits,
        splitter=splitter,
        pipeline=pipeline,
        cv=cv,
        scorer=scorer,
        embargo=embargo,
    )

    report = AuditReport()
    for check in ALL_CHECKS:
        if not check.applies(ctx):
            continue
        report.checks_run.append(check.name)
        report.findings.extend(check.run(ctx))

    return report
