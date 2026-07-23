"""
Shuffle Check
=============

A permutation sanity test: refit the exact same pipeline + cross-validation
procedure on randomly shuffled labels, several times, to build a null
distribution of achievable scores. Compare the real score against that null.

If the real score is not clearly better than what random labels can achieve
under the *same* pipeline and CV splitter, something is off — either there is
no real signal, or (more interestingly for leakage-hunting) part of the
pipeline is exploiting information that has nothing to do with the true
label (e.g. row identity, leaked features, or a splitter that lets identical
rows appear in both train and test).

Honesty note: this is a statistical sanity check (similar in spirit to
`sklearn.model_selection.permutation_test_score`), not a proof of any specific
leak. Pair it with the `causality` and `split` checks for a stronger case.

Only runs when the caller supplies `pipeline`, `cv`, and `scorer`.
"""

from __future__ import annotations

import copy

import numpy as np

from peek.checks.base import AuditContext
from peek.report import Finding, Severity

# n=24 keeps the achievable p-value floor (1/(n+1)) comfortably below the
# significance threshold even when the real score beats every shuffle.
N_SHUFFLES = 24
P_VALUE_THRESHOLD = 0.05


class ShuffleCheck:
    name = "shuffle"

    def applies(self, ctx: AuditContext) -> bool:
        return ctx.pipeline is not None and ctx.cv is not None and ctx.scorer is not None

    def _cv_score(self, ctx: AuditContext, X, y: np.ndarray) -> float:
        fold_scores = []
        for train_idx, test_idx in ctx.cv.split(X, y):
            model = copy.deepcopy(ctx.pipeline)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_scores.append(ctx.scorer(y_test, preds))
        return float(np.mean(fold_scores))

    def run(self, ctx: AuditContext) -> list[Finding]:
        X = ctx.df.drop(columns=[ctx.target, ctx.time_col])
        y = ctx.df[ctx.target].to_numpy()

        real_score = self._cv_score(ctx, X, y)

        rng = np.random.default_rng(0)
        shuffled_scores = np.array([
            self._cv_score(ctx, X, rng.permutation(y)) for _ in range(N_SHUFFLES)
        ])

        p_value = (np.sum(shuffled_scores >= real_score) + 1) / (len(shuffled_scores) + 1)
        mean_shuffled = float(shuffled_scores.mean())
        std_shuffled = float(shuffled_scores.std())

        detail = (
            f"real score = {real_score:.4f} | shuffled-label scores: "
            f"mean={mean_shuffled:.4f}, std={std_shuffled:.4f}, "
            f"max={shuffled_scores.max():.4f} (n={N_SHUFFLES}) | p-value={p_value:.4f}"
        )

        if p_value > P_VALUE_THRESHOLD:
            return [Finding(
                check=self.name,
                severity=Severity.CRITICAL,
                message="model's real score is not statistically distinguishable "
                        "from scores achieved on randomly shuffled labels",
                detail=detail + (
                    "\nEither there is no real signal, or the pipeline/splitter is "
                    "letting the model exploit something other than the true label "
                    "(duplicate rows across folds, a leaked identity feature, etc.)."
                ),
            )]

        return [Finding(
            check=self.name,
            severity=Severity.PASS,
            message="real score is significantly better than the shuffled-label null "
                    f"(p={p_value:.4f})",
            detail=detail,
        )]
