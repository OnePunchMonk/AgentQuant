# Reversible memory compaction

`src.memory.compaction.MemoryCompactor` stages the current Trial and Note
records together with a deterministic projection of the temporal evidence graph.
The bundle uses lossless JSON serialization plus zlib compression and includes
an integrity hash, source IDs, record types, market data dates, knowledge dates,
and the full graph nodes and edges. Edges retain source record IDs and the
graph's market-time and knowledge-time provenance fields, including OOS
visibility dates. Opposing assertions remain separate; compaction does not
resolve contradictions.

The API is intentionally separate from `MemoryService`: it never edits canonical
SQLite rows. Callers may persist an accepted bundle as an archive and use the
rollback token to restore source-shaped records. A future storage integration
must keep canonical rows or another verifiable copy until restoration has been
checked. There is no raw record deletion in this implementation.

## Pre-archive gate

`compact(trials, notes, probes=..., archive_path=...)` stages a candidate, then
runs each probe through `EvidenceGraph.retrieve` against both the original and
the candidate records. It compares the actual retrieved source ID sets under
query, scope, market cutoff, knowledge cutoff, and result limit. Optional
expected source IDs restrict which returned sources count as relevant. The gate
also checks explicit decision signal retention. All configured thresholds
default to 1.0. The source manifest and complete source record ID set are
invariant checks regardless of thresholds. A durable archive path is required:
the bundle is atomically written, reloaded, and verified before the result is
marked archived.

`candidate_transform` is an optional evaluation hook for testing or evaluating
another future compact representation. A transformed candidate that changes
retrieved sources, drops records, or loses decisions fails the gate. Production
callers should omit it and use the lossless bundle until they have a separately
verifiable representation and corresponding retrieval probes.

## Running compaction and the benchmark

Create a verified archive from an existing database with:

```bash
python scripts/compact_memory.py --db-path results/agentquant.db \
  --archive results/memory.archive.json
```

The snapshot read is read-only. The archive file must be retained and verified
before any future system removes canonical records. This command itself does
not remove records. To run the fixed fixture report without a database:

```bash
python scripts/compact_memory.py --benchmark
```

The report gives uncompressed vs compressed bytes and before/after relevance,
citation, and decision retention for fixed cutoff-aware retrieval probes.
`CompactedMemory.load(path)` verifies payload SHA-256 and the source ID, record
type, market date, knowledge date, and bundle ID manifests. `save(path)` writes
to a temporary file in the same directory, fsyncs it, atomically replaces the
destination, and verifies the result by reloading it. Rollback accepts either
the cached bundle token or the archive path.

## Deterministic evaluation fixtures

`tests/test_memory_compaction.py` exercises fixed cases: contradictory
accepted/rejected evidence and provenance preservation; a false-positive
decision mutation that must fail the gate; stale evidence beyond a market cutoff;
loss of a currently visible citation; query execution under market and knowledge
cutoffs; archive corruption detection; and exact source-record restoration
through rollback. Run with:

```bash
pytest tests/test_memory_compaction.py
```

The gate measures only the supplied probes and explicit decision signals. It is
not an estimate of all possible future queries; callers should include probes
for each important decision class and temporal boundary before archiving.
