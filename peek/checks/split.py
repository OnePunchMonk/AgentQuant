"""
Split Check
===========

Audits train/test splits (either passed directly as `splits`, or produced by
a sklearn-style `splitter.split(df)`) for temporal leakage:

- Any test row whose timestamp is <= a training row's timestamp is fine only
  if the training row comes strictly before it; overlap or future-dated
  training rows are a leak.
- Missing purge/embargo gap between train and test (López de Prado): rows
  immediately adjacent to the test window with labels that depend on future
  data can still leak signal across the boundary.
"""

from __future__ import annotations

import numpy as np

from peek.checks.base import AuditContext
from peek.report import Finding, Severity


class SplitCheck:
    name = "split"

    def applies(self, ctx: AuditContext) -> bool:
        return ctx.splits is not None or ctx.splitter is not None

    def _iter_splits(self, ctx: AuditContext):
        if ctx.splits is not None:
            yield from ctx.splits
            return
        yield from ctx.splitter.split(ctx.df)

    def run(self, ctx: AuditContext) -> list[Finding]:
        times = ctx.df[ctx.time_col].to_numpy()
        findings: list[Finding] = []
        any_issue = False

        for fold_i, (train_idx, test_idx) in enumerate(self._iter_splits(ctx)):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            train_times = times[train_idx]
            test_times = times[test_idx]
            test_start = test_times.min()

            future_train_mask = train_times >= test_start
            n_future_train = int(future_train_mask.sum())
            if n_future_train > 0:
                any_issue = True
                findings.append(Finding(
                    check=self.name,
                    severity=Severity.CRITICAL,
                    message=f"fold {fold_i}: {n_future_train} training row(s) are dated "
                            "at or after the test window starts",
                    detail=(
                        f"test window starts at {test_start}; those training rows "
                        "let the model train on data from the test period or later."
                    ),
                ))

            # Embargo: check in row-index space so this works for any time
            # dtype (datetime, int, float).  `ctx.embargo` is a number of rows.
            if ctx.embargo > 0:
                test_start_pos = int(test_idx.min())
                # Train rows that are within `embargo` rows before the test
                # window (excluding any already flagged as future-dated).
                embargo_violators = train_idx[
                    (train_idx < test_start_pos)
                    & (train_idx >= (test_start_pos - ctx.embargo))
                ]
                if len(embargo_violators) > 0:
                    boundary = times[train_idx[train_idx < test_start_pos].max()]
                    any_issue = True
                    findings.append(Finding(
                        check=self.name,
                        severity=Severity.WARNING,
                        message=f"fold {fold_i}: {len(embargo_violators)} training row(s) fall "
                                f"inside the requested embargo gap ({ctx.embargo} rows) before "
                                "the test window",
                        detail=f"last training timestamp before test: {boundary}",
                    ))

        if not any_issue:
            findings.append(Finding(
                check=self.name,
                severity=Severity.PASS,
                message="no train/test temporal overlap found across the provided splits",
            ))
        return findings
