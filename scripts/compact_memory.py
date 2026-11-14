"""Read-only, gated memory archive creation and deterministic benchmark.

Examples:
  python scripts/compact_memory.py --db-path results/agentquant.db --archive results/memory.archive.json
  python scripts/compact_memory.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.memory.compaction import MemoryCompactor, RetrievalProbe  # noqa: E402
from src.memory.models import Note, Trial  # noqa: E402
from src.memory.service import MemoryService  # noqa: E402


def _fixtures() -> tuple[list[Trial], list[Note], list[RetrievalProbe]]:
    trials = [
        Trial(asset="SPY", strategy_type="momentum", params={"lookback": 20},
              trial_id="bench-accepted", run_id="fixture-a", created_at="2024-02-01T00:00:00Z",
              data_end="2024-01-31", outcome="accepted", is_sharpe=0.8,
              reasoning="January breakout accepted"),
        Trial(asset="SPY", strategy_type="momentum", params={"lookback": 20},
              trial_id="bench-rejected", run_id="fixture-b", created_at="2024-03-01T00:00:00Z",
              data_end="2024-02-29", outcome="rejected", is_sharpe=-0.2,
              reasoning="February breakout rejected"),
        Trial(asset="SPY", strategy_type="momentum", params={"lookback": 10},
              trial_id="bench-stale", run_id="fixture-c", created_at="2024-04-01T00:00:00Z",
              data_end="2024-03-31", outcome="accepted", is_sharpe=0.6,
              reasoning="March result unavailable in February replay"),
    ]
    notes = [Note(kind="memo", body="Holdout confirms February failure", note_id="bench-note",
                  created_at="2024-03-02T00:00:00Z", data_end="2024-02-29",
                  strategy_type="momentum", asset="SPY", source="benchmark:fixture")]
    probes = [
        RetrievalProbe("January accepted", query="outcome accepted", market_as_of="2024-02-15",
                       knowledge_as_of="2024-02-15", strategy_type="momentum", asset="SPY",
                       expected_source_ids=("bench-accepted", "bench-rejected", "bench-stale")),
        RetrievalProbe("February rejected", query="outcome rejected", market_as_of="2024-03-15",
                       knowledge_as_of="2024-03-15", strategy_type="momentum", asset="SPY",
                       expected_source_ids=("bench-accepted", "bench-rejected", "bench-stale")),
    ]
    return trials, notes, probes


def run_benchmark() -> dict:
    trials, notes, probes = _fixtures()
    compactor = MemoryCompactor()
    bundle = compactor.stage(trials, notes)
    evaluation = compactor.evaluate(trials, notes, bundle, probes)
    raw_bytes = len(json.dumps(bundle.unpack(), sort_keys=True, separators=(",", ":")).encode())
    return {
        "fixture": "fixed-memory-compaction-v1",
        "source_records": len(bundle.source_ids),
        "original_uncompressed_bytes": raw_bytes,
        "compacted_bytes": bundle.compressed_size,
        "size_reduction_ratio": round(1 - bundle.compressed_size / max(raw_bytes, 1), 6),
        "fixed_vs_compact_retention": {
            "relevance": evaluation.relevance_retention,
            "citation": evaluation.citation_retention,
            "decision": evaluation.decision_retention,
            "passed": evaluation.passed,
            "failures": list(evaluation.failures),
        },
        "probes": len(probes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="run fixed before/after fixtures")
    parser.add_argument("--db-path", help="SQLite memory database path (read-only snapshot)")
    parser.add_argument("--archive", help="required durable archive output path")
    parser.add_argument("--market-as-of", help="market cutoff for the baseline retrieval probe")
    parser.add_argument("--knowledge-as-of", help="knowledge cutoff for the baseline retrieval probe")
    args = parser.parse_args()
    if args.benchmark:
        print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
        return 0
    if not args.db_path or not args.archive:
        parser.error("--db-path and --archive are required unless --benchmark is used")
    service = MemoryService(db_path=args.db_path, mode="read")
    trials, notes = service.snapshot()
    probes = [RetrievalProbe("snapshot baseline", market_as_of=args.market_as_of,
                             knowledge_as_of=args.knowledge_as_of,
                             expected_source_ids=tuple(t.trial_id for t in trials) +
                                                  tuple(n.note_id for n in notes))]
    compactor = MemoryCompactor()
    result = compactor.compact(trials, notes, probes=probes, archive_path=args.archive)
    print(json.dumps({"archived": result.archived, "archive": str(Path(args.archive).resolve()),
                      "bundle_id": result.bundle.bundle_id, "source_records": len(result.bundle.source_ids),
                      "evaluation": result.evaluation.__dict__}, indent=2, default=list, sort_keys=True))
    return 0 if result.archived else 2


if __name__ == "__main__":
    raise SystemExit(main())
