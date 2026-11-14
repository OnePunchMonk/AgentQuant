# Benchmark suite evolution contract

`src.agent.benchmark_suite` defines a static contract for benchmark identity and
report validation. It does not execute experiments, load price data, or inspect
sealed rows.

## Immutable manifests

A suite manifest has a schema version, stable suite ID, integer suite version,
and ordered episode records. Each episode identifies its partition (`train`,
`selection`, or `sealed`), data content hash, split hash, and optional asset.
`content_hash` is SHA-256 over canonical JSON of the complete manifest. Save the
returned manifest as a new artifact; do not rewrite a prior version. Version 1
has no parent. Later versions carry the previous manifest's hash and must append
new episode IDs. Verify a loaded child against an independently trusted parent
with `verify_evolution(parent, child)`; a hash string inside the child alone
cannot prove lineage. This establishes lineage while preserving all earlier records.

## Partition boundaries

Train and selection episodes are the only IDs allowed in a candidate report's
`partition_ids` input. They must match the partition named by the suite and be
disjoint. A report must not claim that it read sealed episodes. Sealed outcomes
may be attached separately after evaluation, but are outside this validator's
input and must not influence training or selection. The validator never opens a
manifest path or dataset path, so it cannot expose sealed data.

## Matched reports

Reports declare `suite_hash`, `candidate_id`, `partition_ids`, `budgets`, and
`evidence_tier`. `validate_matched_reports` checks each report and requires the
same budget mapping, evidence tier, and train/selection episode IDs across at
least two distinct candidates. The tier
is descriptive provenance (`unverified`, `fixture_demo`, `synthetic`,
`measured_historical`, or `paper`); it does not upgrade evidence or generate
metrics. The validator reports only contract errors and candidate count. It
never imputes missing outcomes or fabricates scores.

```python
from src.agent.benchmark_suite import BenchmarkSuite, SuiteEpisode, validate_matched_reports

suite = BenchmarkSuite(
    suite_id="example", version=1,
    episodes=(
        SuiteEpisode("train-01", "train", "data-sha256", "split-sha256"),
        SuiteEpisode("selection-01", "selection", "data-sha256", "split-sha256"),
        SuiteEpisode("sealed-01", "sealed", "data-sha256", "split-sha256"),
    ),
)
result = validate_matched_reports(reports, suite=suite)
```
