# HyDE style hypothesis retrieval

The retrieval sidecar in `src/agent/hyde_retrieval.py` takes generated
strategy descriptions and ranks time-visible evidence for inspection. Its
default scorer is deterministic lexical cosine; generators and optional
similarity functions are injected. This path has no model or network
dependency and does not claim predicted or measured performance.

```python
from src.agent.hyde_retrieval import Evidence, HydeRetrievalTool, HypothesisRequest

tool = HydeRetrievalTool(generator, evidence_rows)
candidates = tool.run(HypothesisRequest(
    regime="high volatility", strategy_types=("momentum",), as_of="2024-06-30",
    max_hypotheses=3, max_evidence=8,
))
```

Each candidate returns evidence IDs as citations plus each record's source,
market period, known-at date, evidence tier, similarity, quality, and optional
explicit measured metric. Citation IDs are copied from retrieved records;
zero-similarity matches are excluded. `validate_citations` checks citations
against retrieved records and the same market and knowledge cutoffs, and
reports `grounded`, `stale`, or `unsupported` with the specific IDs. Missing,
malformed, future-known, or future-market evidence fails closed. An empty
retrieval yields an unsupported candidate citation status, not fabricated
support. Citation validity establishes provenance and visibility; it does not
prove a strategy claim.

`evaluation_status` starts as `unevaluated`. Retrieval and quality scores are
ranking aids only. They are not Sharpe estimates, backtest outcomes, or a
substitute for the normal backtest and quality gate. The synthetic benchmark
in `scripts/benchmark_hyde_retrieval.py` exercises two regimes against
lexical, literature-keyword, grid-order, shuffled-evidence, and no-memory
baselines under a shared evidence budget. Its fixture relevance labels measure
ranking behavior only; they do not establish real-world retrieval quality or
financial utility. Run it with:

```bash
python scripts/benchmark_hyde_retrieval.py --k 5 --seed 23
```
