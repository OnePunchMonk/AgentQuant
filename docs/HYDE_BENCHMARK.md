# HyDE retrieval synthetic benchmark

Run the deterministic offline comparison from the repository root:

```bash
python scripts/benchmark_hyde_retrieval.py --k 5 --seed 23
```

The benchmark compares the `HydeRetrievalTool` ranking against a lexical
literature-keyword control, canonical parameter-grid order, shuffled visible
evidence, and a no-memory arm. It uses two synthetic regimes (trend and mean
reversion), with a common top-k evidence budget. Grid arms expose up to k
parameter candidates while HyDE maps its single generated hypothesis to one
canonical candidate; the JSON reports these counts explicitly. It reports
precision@k and reciprocal rank for fixture relevance labels, and whether the
HyDE-mapped parameter tuple appears anywhere in the canonical grid.

The fixture includes relevant evidence outside market-time and knowledge-time
cutoffs. Those rows are excluded before every evidence-backed arm is scored.
The shuffle is seeded. No-memory returns no evidence. Synthetic relevance
labels are an implementation check, not literature judgments or expert labels.

**Evidence tier: synthetic fixture.** Results support only deterministic
behavior and ranking against the labels embedded in this toy fixture. They do
not establish retrieval quality on real literature, candidate usefulness,
returns, Sharpe, or backtest improvement. Every candidate remains
`unevaluated`; no strategy is run. A financial utility claim requires a
separate preregistered, leakage-controlled walk-forward experiment using
verified real sources and an independent evaluator.
