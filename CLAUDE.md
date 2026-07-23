# CLAUDE.md — Build Plan: Peek (Data-Leakage Auditor)

> This file is the working plan for the pivot of this repo. It is the source of
> truth for what we are building and why. Update it as the plan evolves.

## 1. The pivot in one sentence

Repurpose this repo from "AgentQuant, an autonomous LLM quant agent" into
**Peek** — a Python library + CLI that catches **look-ahead bias and data
leakage** in *any* time-series ML pipeline, not just trading.

The repo (GitHub: `OnePunchMonk/AgentQuant`, ~160 stars) keeps its URL and stars.
AgentQuant becomes the **origin story / case study**: "we built an LLM quant
research agent, then discovered it was fooling us with a leaky backtest — so we
extracted the guardrails into a general tool."

## 2. Why this category

- **Open water.** Agent-eval and backtesting-in-Claude (MCP) are saturated
  (Braintrust, Confident AI, QuantConnect MCP, etc.). Time-series *leakage
  detection* is not: the only dedicated package (`tsdataleaks`) is R-only and
  scoped to forecasting competitions. `sklearn.TimeSeriesSplit` only *splits*,
  it does not *detect* leakage. No go-to Python auditor exists.
- **Universal, dreaded pain.** "Look-ahead bias: the invisible killer." Models
  score 0.95 offline and die in prod because a feature secretly peeked at the
  future. Every data scientist doing time-series ML hits this.
- **Reuses our real strength.** `src/features/lookback_guard.py`
  (`WarmupEnforcer`, `enforce_lookback`), walk-forward validation, and the whole
  "prove it isn't cheating" DNA generalize directly.
- **Star-magnet shape.** Focused, single-purpose libraries with a memorable hook
  ("is your model peeking at the future?") star better than sprawling platforms.

## 3. Product hook

> **Peek — is your model peeking at the future?**
> Point it at a dataset, a feature function, or a CV split. It screams when
> something leaks, with the proof. Not "here's your score" — "**don't trust this
> score, and here's why.**"

## 4. Architecture

New top-level package `peek/` alongside the existing `src/` (AgentQuant stays as
the documented case study; nothing is deleted).

```
peek/
├── __init__.py          # public API: audit(), AuditReport, Severity, checks
├── report.py            # Finding, Severity, AuditReport (verdict + rich render)
├── audit.py             # audit() orchestrator + AuditContext
├── datasets.py          # make_leaky_dataset() / make_clean_dataset() for demos+tests
├── cli.py               # `peek demo`, `peek audit <csv>`
└── checks/
    ├── __init__.py
    ├── base.py          # Check base class + AuditContext
    ├── target_leak.py   # feature == target shifted into the future (definitive)
    ├── causality.py     # recompute-on-truncation test (GOLD STANDARD, needs feature_fn)
    ├── split.py         # train/test temporal overlap, future-in-train, embargo/purge gap
    └── shuffle.py       # permutation/target-shuffle sanity test (needs pipeline+cv)
```

Dependencies: **numpy, pandas, scipy, rich only** (all already in pyproject).
No hard sklearn dependency — splitters/pipelines are duck-typed and optional.

## 5. The checks (must be statistically honest — no fake science)

| Check | Mode | Needs | Verdict | How it works |
|---|---|---|---|---|
| **TargetLeak** | DataFrame | df+target | CRITICAL | Flags a feature that is a near-exact copy of the target shifted from the *future* (`corr(feature, target.shift(-k)) ≈ 1`, k≥0). Definitive, high precision. |
| **Causality** | Feature-fn | `feature_fn(df)->DataFrame` | CRITICAL | Recompute features on the full series vs on the series truncated at probe time t. If the value AT t differs, the feature used future rows → look-ahead leak. This is the flagship, deterministic test. |
| **Split** | Split | `splits`/`splitter` + time_col | CRITICAL/WARN | Detects train/test timestamp overlap, training data dated after test start, and missing purge/embargo gap (López de Prado). |
| **Shuffle** | Pipeline | `pipeline`+`cv`+`scorer` | CRITICAL/WARN | Permutation test: shuffle the target, rerun the same CV pipeline. If the score stays well above chance, the *evaluation harness itself* leaks (e.g. preprocessing fit on all data). |

Honesty rule baked into docs: DataFrame mode gives **high-precision definitive**
catches (future-copy) but cannot prove absence of leakage; the **Causality**
(feature_fn) and **Shuffle** (pipeline) modes are the rigorous ones. Say so
plainly — that candor is on-brand.

## 6. Public API (target)

```python
import peek

report = peek.audit(
    df,
    time_col="date",
    target="y",
    feature_fn=build_features,   # optional -> enables Causality check
    splits=my_cv_splits,          # optional -> enables Split check
    pipeline=model, cv=tscv, scorer=roc_auc,  # optional -> enables Shuffle check
    horizon=1,
)

print(report)          # rich verdict + findings table
report.has_leak        # bool
report.verdict         # "LEAKING" | "SUSPICIOUS" | "CLEAN"
report.to_dict()       # for CI / JSON
```

CLI:
```
peek demo                     # runs on built-in leaky dataset -> instant wow
peek audit data.csv --time date --target y
```

## 7. Deliverables for the PR

- [ ] `peek/` package (files above), working end-to-end.
- [ ] `peek/datasets.py` synthetic leaky + clean generators.
- [ ] `tests/test_peek_*.py` — duplicate/future-copy caught, clean passes,
      causality catches centered-rolling/full-series-stat leaks and passes
      expanding-window features, split overlap caught, report verdict logic.
- [ ] Rewrite `README.md` to lead with Peek; AgentQuant demoted to a
      "Case study / origin story" section with a link to the finance code.
- [ ] `pyproject.toml`: add `peek*` to packages, add `peek = "peek.cli:main"`
      console script (keep existing `agentquant` script). Keep name/deps sane.
- [ ] Keep CI green (`pytest` + `ruff`). All new code must pass ruff (E,F,W,I).
- [ ] Commit **without** Claude as co-author. Open a PR against `main`.

## 8. Positioning / naming notes

- Repo name stays `AgentQuant` (preserves stars + inbound links). README makes
  the pivot explicit; no repo rename.
- Import package is `peek`. PyPI name may need a suffix later (e.g. `peek-ml`) if
  taken — do **not** promise a `pip install peek` that doesn't resolve; document
  editable install (`pip install -e .`) as the working path, PyPI "coming soon".

## 9. Launch (after merge — not part of this PR)

Honest, skeptic-flavored writeup: *"An LLM quant agent fooled me with a leaky
backtest. So I built a tool that catches it — for any time-series model."*
Post to r/MachineLearning ([P]), r/datascience, HN (Show HN), ML Twitter, with
the `peek demo` output as the hero image.
