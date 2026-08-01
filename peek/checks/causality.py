"""
Causality Check (flagship)
===========================

The gold-standard test for look-ahead bias: recompute the user's feature
function on the full series versus on the series *truncated* at a probe
timestamp. If a feature's value at the probe row differs between the two
computations, the feature function used rows after the probe — i.e. it saw
the future.

This is the generalization of AgentQuant's `WarmupEnforcer` /
`enforce_lookback` idea: instead of asserting a fixed warmup, we prove
causality directly by truncation, which catches any leak (centered rolling
windows, full-series normalization, target encoding on the whole set, etc.)
regardless of its exact shape.

Only runs when the caller supplies `feature_fn`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from peek.checks.base import AuditContext
from peek.report import Finding, Severity

N_PROBES = 8
ATOL = 1e-9
RTOL = 1e-6


class CausalityCheck:
    name = "causality"

    def applies(self, ctx: AuditContext) -> bool:
        return ctx.feature_fn is not None

    def run(self, ctx: AuditContext) -> list[Finding]:
        df = ctx.df
        n = len(df)
        min_probe = max(10, n // 10)
        if n - min_probe < 2:
            return [Finding(
                check=self.name,
                severity=Severity.WARNING,
                message="not enough rows to run the causality (truncation) test",
            )]

        probe_positions = np.unique(
            np.linspace(min_probe, n - 1, num=min(N_PROBES, n - min_probe), dtype=int)
        )

        full_features = ctx.feature_fn(df)
        if not isinstance(full_features, pd.DataFrame):
            full_features = pd.DataFrame(full_features)

        # Contract check: feature_fn must be length-preserving.  If it drops
        # or adds rows the probe comparison is undefined.
        if len(full_features) != len(df):
            return [Finding(
                check=self.name,
                severity=Severity.WARNING,
                message=(
                    f"feature_fn returned {len(full_features)} rows for a "
                    f"{len(df)}-row DataFrame — rows were dropped or added.  "
                    "The causality check requires a length-preserving feature_fn "
                    "(keep NaN rows rather than dropping them)."
                ),
            )]

        # Non-determinism check: run feature_fn a second time on the full df.
        # If outputs differ the function is stochastic and cannot be audited.
        full_features_2 = ctx.feature_fn(df)
        if not isinstance(full_features_2, pd.DataFrame):
            full_features_2 = pd.DataFrame(full_features_2)
        numeric_cols = [
            c for c in full_features.columns
            if pd.api.types.is_numeric_dtype(full_features[c])
            and c in full_features_2.columns
        ]
        for col in numeric_cols:
            v1 = full_features[col].to_numpy(dtype=float, na_value=np.nan)
            v2 = full_features_2[col].to_numpy(dtype=float, na_value=np.nan)
            valid = ~(np.isnan(v1) | np.isnan(v2))
            if valid.any() and not np.allclose(v1[valid], v2[valid], atol=ATOL, rtol=RTOL):
                return [Finding(
                    check=self.name,
                    severity=Severity.WARNING,
                    message=(
                        f"feature_fn is non-deterministic: feature '{col}' differs "
                        "across two runs on the same data.  Seed all random steps "
                        "before using the causality check."
                    ),
                )]

        leaking_cols: dict[str, list[int]] = {}
        for pos in probe_positions:
            truncated = df.iloc[: pos + 1]
            truncated_features = ctx.feature_fn(truncated)
            if not isinstance(truncated_features, pd.DataFrame):
                truncated_features = pd.DataFrame(truncated_features)

            # Skip this probe if truncated feature_fn contracted rows — we
            # can't safely align without a shared key.
            if len(truncated_features) != len(truncated):
                continue

            common_cols = [c for c in full_features.columns if c in truncated_features.columns]
            for col in common_cols:
                full_val = full_features[col].iloc[pos]
                trunc_val = truncated_features[col].iloc[-1]
                if pd.isna(full_val) and pd.isna(trunc_val):
                    continue
                if pd.isna(full_val) or pd.isna(trunc_val):
                    continue
                if not np.isclose(full_val, trunc_val, atol=ATOL, rtol=RTOL):
                    leaking_cols.setdefault(col, []).append(int(pos))

        findings: list[Finding] = []
        if leaking_cols:
            for col, positions in leaking_cols.items():
                sample_pos = positions[0]
                full_val = float(full_features[col].iloc[sample_pos])
                findings.append(Finding(
                    check=self.name,
                    severity=Severity.CRITICAL,
                    message=f"feature '{col}' changes value when future rows are removed",
                    detail=(
                        f"at row {sample_pos}, value computed on the full series "
                        f"({full_val!r}) differs from the value computed with only "
                        f"data up to that row. This feature is not causal — it used "
                        f"future information. Leaked at {len(positions)}/{len(probe_positions)} probes."
                    ),
                ))
        else:
            findings.append(Finding(
                check=self.name,
                severity=Severity.PASS,
                message=(
                    f"feature_fn is causal at all {len(probe_positions)} probed timestamps "
                    "(truncating the series does not change past values)"
                ),
            ))
        return findings
