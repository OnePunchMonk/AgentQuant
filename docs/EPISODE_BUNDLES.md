# Episode bundles and replay

An episode bundle stores the episode result, provenance, dataset identity, and software identity. It does not contain OHLCV rows. Keep the referenced private data file in its controlled location; the bundle records a content hash and source metadata so a rerun can check that it is using the same input.

Version 2 bundles include SHA-256 hashes of the serialized payload and key references. They detect accidental edits and corruption; they are not signed attestations against someone who can rewrite both the content and hashes. Version 1 remains readable, without that integrity guarantee. The checked-in `experiments/episode_bundles/memo-v1_base.json` and `tests/fixtures/portable_splits.json` form a synthetic, zero-key portability example. Its scores are fixture wiring evidence only.

## Report replay

Use `--replay` to rebuild the memo from the persisted episode result. This is offline and does not need the original data:

```bash
python scripts/export_research_memo.py --replay experiments/episode_bundles/RUN.json --output results/replayed
```

## Fresh replay

Use `--fresh-replay` only when you want to execute the experiment again. The command reads the saved allowlisted run parameters, locates the original synthetic or user supplied dataset, hashes its normalized OHLCV content, and stops before execution if the hash differs. It never runs the stored command as shell code.
It also checks the saved episode split file hash when one is recorded. A fresh execution invokes the final holdout guard; report-only replay does not grade or consume any holdout.

```bash
python scripts/export_research_memo.py --fresh-replay experiments/episode_bundles/RUN.json --output results/fresh
```

For private data, restore the original CSV to the recorded path. If it is unavailable, the error identifies the missing path; use `--replay` to reproduce the report without the data. A dataset mismatch also stops execution and reports both expected and observed hashes. Fresh replay may require the same optional model credentials as the original run; otherwise the configured offline fallback is used.

Fresh replay is an execution attempt against matching data and saved run parameters. It does not promise byte-identical model output when an external model provider is involved. Report replay reproduces the original memo from the captured result.
