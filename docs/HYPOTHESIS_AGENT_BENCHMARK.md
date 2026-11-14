# Hypothesis Agent Technique Benchmark

Status: **offline fixture plus public historical mode**. The four proposal methods are literature-keyword, HyDE-style, analytical rules, and a simple ensemble. Both use the `HypothesisRequest` / `SyntheticHypothesis` contract and feed the existing cutoff-aware `HydeRetrievalTool`.

## Run

```bash
python scripts/benchmark_hypothesis_techniques.py --seed 29
python scripts/benchmark_hypothesis_techniques.py --output results/hypothesis_techniques_fixture.json
pytest tests/test_hypothesis_agent.py -q
```

Historical mode fetches adjusted daily SPY closes from Yahoo Finance for pinned dates. It can instead read a public CSV with `Date` and `Adj Close` (or `Close`) columns:

```bash
python scripts/benchmark_hypothesis_techniques.py --mode historical --symbol SPY --output results/hypothesis_techniques_spy.json
python scripts/benchmark_hypothesis_techniques.py --mode historical --csv /path/to/spy.csv --candidate-budget 4 --cost-bps 5
```

The historical mode evaluates each candidate's momentum, mean-reversion, or volatility signal from its proposed parameters. Signals are lagged one bar; turnover incurs 5 bps per unit position change by default. Every fixed date window is split chronologically: first 60% for training and final 40% as holdout. Eight calendar windows and exact inclusive boundaries are embedded in the JSON output. Candidate count is capped at the same requested budget per technique. Output records train and holdout metrics, split dates, observation counts, price source, date span, and cost assumptions. The dates are pinned in source rather than inferred from holdout returns.

If Yahoo Finance is unavailable or a window lacks sufficient rows, the historical run stops with an error. It never substitutes fixture data. CSV `Close` inputs may be unadjusted; the provenance identifies the selected column, so use `Adj Close` when available.

Fixture mode covers six named synthetic regimes and a default budget of four candidates. It reports fixture Sharpe mean, fixture Sharpe gap, novelty rate versus fixture evidence, false-positive rate (candidate fixture Sharpe <= 0), method cost units, and cumulative learning curves.

## Evidence and interpretation

The default fixture mode generates synthetic, deterministic return series and evidence records. Its metrics validate reproducibility and metric plumbing only. The `fixture_sharpe_*` names and `data_label` fields distinguish those numbers from the optional historical mode. Cost units are relative bookkeeping units, not dollars or API invoices.

The literature method is an offline adapter over timestamped fixture evidence, not a literature search client. In historical mode it uses a fixed, predeclared parameter catalog rather than purported literature findings. The HyDE method emits a hypothetical description, analytical uses transparent regime rules, and ensemble aggregates those proposals. Historical backtests are descriptive public-data evidence: the static parameter catalog, fixed date windows, and many tested candidate variants limit any generalization. The historical mode does not claim prospective performance or control for survivorship, publication, or data-vendor revisions. It also does not yet measure real model-generated HyDE ideas or a learned technique selector. Do not infer a winning technique when candidate sets overlap.
