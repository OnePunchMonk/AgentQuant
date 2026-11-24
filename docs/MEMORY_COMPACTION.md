# Memory compaction

`src.memory.compaction.MemoryCompactor` creates a compressed, lossless archive of Trial and Note records and a projection of their temporal evidence graph. The bundle keeps source IDs, record types, dates, graph provenance, and opposing assertions. It does not resolve contradictions or change canonical SQLite rows.

## Archive gate

Before accepting an archive, `compact()` compares graph retrieval from the original and candidate records using the supplied probes. It checks source IDs, decision signals, and the complete source manifest. Thresholds default to 1.0. The archive is written atomically, reloaded, and verified before it is marked archived.

`candidate_transform` is an evaluation hook for alternative representations. Production callers should omit it unless they have independently verified their representation and retrieval probes. The gate covers only the supplied probes and decision signals; include important decision classes and temporal boundaries.

```bash
python scripts/compact_memory.py --db-path results/agentquant.db \
  --archive results/memory.archive.json
python scripts/compact_memory.py --benchmark
pytest tests/test_memory_compaction.py
```

The first command reads a database snapshot and creates an archive; it does not delete records. The benchmark reports size and retrieval retention on fixed fixtures. `CompactedMemory.load()` verifies the payload and manifests; rollback accepts a cached bundle token or archive path. Any future deletion of canonical records must retain a verifiable copy and check restoration first.
