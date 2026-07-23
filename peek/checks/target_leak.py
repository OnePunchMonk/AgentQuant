"""
Target Leak Check
=================

Flags features that are near-exact copies of the target, shifted from the
future. This is the highest-precision, cheapest check: it needs nothing but
the dataframe and always runs.

Example of what it catches: a "next_day_return" column accidentally left in
the feature set used to predict "next_day_return".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from peek.checks.base import AuditContext
from peek.report import Finding, Severity

CORR_THRESHOLD = 0.98
MAX_SHIFT_PROBE = 5


class TargetLeakCheck:
    name = "target_leak"

    def applies(self, ctx: AuditContext) -> bool:
        return True

    def run(self, ctx: AuditContext) -> list[Finding]:
        findings: list[Finding] = []
        target = ctx.df[ctx.target]
        feature_cols = [
            c for c in ctx.df.columns
            if c not in (ctx.target, ctx.time_col) and pd.api.types.is_numeric_dtype(ctx.df[c])
        ]

        any_leak = False
        for col in feature_cols:
            feature = ctx.df[col]
            for shift in range(0, MAX_SHIFT_PROBE + 1):
                shifted_target = target.shift(-shift)
                valid = feature.notna() & shifted_target.notna()
                if valid.sum() < max(10, len(ctx.df) * 0.1):
                    continue
                corr = np.corrcoef(feature[valid], shifted_target[valid])[0, 1]
                if abs(corr) >= CORR_THRESHOLD:
                    any_leak = True
                    when = "the same row as" if shift == 0 else f"{shift} step(s) into the future of"
                    findings.append(Finding(
                        check=self.name,
                        severity=Severity.CRITICAL,
                        message=f"feature '{col}' is a near-copy of the target shifted {shift} step(s)",
                        detail=(
                            f"corr(feature['{col}'], target.shift(-{shift})) = {corr:.4f} "
                            f"(feature matches {when} the target)."
                        ),
                    ))
                    break  # one finding per feature is enough

        if not any_leak:
            findings.append(Finding(
                check=self.name,
                severity=Severity.PASS,
                message="no feature is a near-exact copy of the (shifted) target",
            ))
        return findings
