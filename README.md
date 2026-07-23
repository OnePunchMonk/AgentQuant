# Peek: Is Your Model Peeking at the Future?

**A Python library that audits time-series ML pipelines for look-ahead bias and data leakage — and tells you exactly where the leak is, not just that your score looks suspicious.**

[![CI](https://github.com/OnePunchMonk/AgentQuant/actions/workflows/ci.yml/badge.svg)](https://github.com/OnePunchMonk/AgentQuant/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen)

---

## The problem

Your model scores 0.95 offline. It falls apart in production. Somewhere, a
feature saw data it shouldn't have — a centered rolling window, a
normalization fit on the whole dataset, a target-encoded column, a CV split
with the wrong dates on the wrong side. This is **look-ahead bias / data
leakage**, and it is one of the most common, most expensive mistakes in
applied ML — and almost universally under-tooled. `sklearn.TimeSeriesSplit`
only *splits* your data; it does not check whether your *features* or your
*evaluation harness* are honest.

Peek is the check that's missing. Point it at a dataframe, a feature
function, a CV split, or a full pipeline, and it screams — with proof —
when something isn't causal.

```bash
$ peek demo

🔍 peek report ──────────────────────────
✗ CRITICAL  feature 'future_return_leak' is a near-copy of the target shifted 0 step(s)
            corr(feature['future_return_leak'], target.shift(-0)) = 1.0000
✗ CRITICAL  feature 'centered_ma_5' changes value when future rows are removed
            at row 49, value computed on the full series (106.998...) differs
            from the value computed with only data up to that row. This
            feature is not causal — it used future information.

Verdict: LEAKING — 2 critical issue(s) found. This score is fiction until fixed.
```

## Install

```bash
git clone https://github.com/OnePunchMonk/AgentQuant.git
cd AgentQuant
pip install -e .
peek demo
```

*(PyPI package coming soon — for now, install from source.)*

## Quickstart

```python
import peek

report = peek.audit(
    df,                       # your full time-ordered dataframe
    time_col="date",
    target="y",
    feature_fn=build_features,          # optional -> enables the causality check
    splits=my_cv_splits,                # optional -> enables the split check
    pipeline=model, cv=tscv, scorer=r2_score,  # optional -> enables the shuffle check
)

print(report)
report.has_leak     # bool
report.verdict       # "LEAKING" | "SUSPICIOUS" | "CLEAN"
report.to_dict()      # for CI / JSON output
```

## What it checks

| Check | Needs | Catches |
|---|---|---|
| **target_leak** | just `df` | A feature that's a near-exact copy of the (possibly shifted) target — always runs. |
| **causality** *(flagship)* | `feature_fn` | Recomputes your features on a truncated series and compares against the full computation. If a value changes, the feature saw the future — catches centered rolling windows, full-series normalization, whole-dataset target encoding, anything. |
| **split** | `splits` or `splitter` | Train/test temporal overlap, training rows dated after the test window starts, missing purge/embargo gaps. |
| **shuffle** | `pipeline` + `cv` + `scorer` | Permutation sanity check: refits your exact pipeline+CV on randomly shuffled labels. If the real score isn't clearly better than the shuffled-label null, the harness — not just a feature — may be leaking. |

**Honesty note:** the `target_leak` check is definitive but narrow (it only
catches direct future-copies). The `causality` and `shuffle` checks are the
rigorous ones — they don't just look for suspicious correlations, they prove
(or disprove) causality by truncation and permutation. No check here proves
the *absence* of leakage; each one proves the *presence* of a specific,
well-defined failure mode. Run all four you can.

## CLI

```bash
peek demo                                       # instant leaky-vs-clean walkthrough
peek audit data.csv --time date --target y      # audit a CSV (target_leak check)
```

## Why this exists

This project started as **AgentQuant**, an autonomous LLM research agent that
proposed and backtested trading strategies. While building its rigorous
walk-forward validation, we kept catching our *own* pipeline quietly cheating
— a rolling feature with a centered window, a warmup period that was one bar
short. We built `WarmupEnforcer` and a lookback guard to stop lying to
ourselves. Then we realized: this problem isn't specific to finance. Any
time-series ML pipeline can leak the same way. Peek is that guard,
generalized and pulled out into its own library.

The original agent is preserved as a case study in
[`docs/AGENTQUANT.md`](docs/AGENTQUANT.md) and still lives in `src/` — a real
ReAct research loop that we used, ironically, to prove that a
context-aware LLM agent can *lose* to a dumb static baseline once you audit
the backtest properly (see [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md)). That
honesty is what led here.

## Project layout

```
peek/                    # the library
├── audit.py             # audit() orchestrator
├── report.py            # Finding / Severity / AuditReport
├── datasets.py           # synthetic leaky/clean datasets used by `peek demo`
├── cli.py                # `peek demo`, `peek audit`
└── checks/
    ├── target_leak.py    # future-copy-of-target detector
    ├── causality.py       # truncation-based causality proof (flagship)
    ├── split.py           # train/test temporal overlap + embargo
    └── shuffle.py          # permutation sanity test on a full pipeline

src/                     # AgentQuant — the LLM research agent (case study)
docs/                    # AGENTQUANT.md, PAPER_DRAFT.md, EXPERIMENTAL_DETAILS.md
tests/                   # peek + AgentQuant test suites
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

**80 tests passing** — 17 for `peek` (target-leak, causality, split, shuffle,
report/verdict logic) plus 63 covering the AgentQuant research agent
(backtest engine, metrics, regime detection, strategies, memory, swarm).

---

> Peek is a diagnostic tool, not a guarantee. It flags well-defined, provable
> failure modes; it cannot prove a pipeline is leak-free. AgentQuant is for
> educational and research purposes only — not financial advice.
