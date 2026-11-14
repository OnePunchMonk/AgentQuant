"""Offline synthetic ranking benchmark for the HyDE retrieval prototype.

This evaluates evidence ranking and candidate novelty only. It does not run
strategies or estimate financial performance.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.hyde_retrieval import (  # noqa: E402
    Evidence,
    HydeRetrievalTool,
    HypothesisRequest,
    SyntheticHypothesis,
)
from src.agent.parameter_grid import ParameterGrid  # noqa: E402

AS_OF = "2024-12-31"
REGIMES = {
    "trend": {
        "query": "persistent upward trend momentum breakout long horizon",
        "keywords": ("trend", "momentum", "breakout", "upward", "long"),
        "params": {"fast_window": 10, "slow_window": 50},
        "irrelevant": "short horizon mean reversion rangebound oversold",
    },
    "reversion": {
        "query": "rangebound market mean reversion oversold short horizon",
        "keywords": ("rangebound", "reversion", "oversold", "short", "range"),
        "params": {"window": 20, "num_std": 2.0},
        "irrelevant": "persistent upward trend momentum breakout long horizon",
    },
}


@dataclass(frozen=True)
class RankingMetrics:
    precision_at_k: float
    reciprocal_rank: float
    relevant_retrieved: int
    novel_candidates: int
    candidate_count: int


class FixtureGenerator:
    def __init__(self, hypothesis: SyntheticHypothesis):
        self.hypothesis = hypothesis

    def generate(self, request: HypothesisRequest) -> list[SyntheticHypothesis]:
        del request
        return [self.hypothesis]


def make_fixture(regime: str) -> tuple[SyntheticHypothesis, list[Evidence], set[str]]:
    spec = REGIMES[regime]
    strategy = "momentum" if regime == "trend" else "mean_reversion"
    params = spec["params"]
    hypothesis = SyntheticHypothesis(spec["query"], strategy, params)
    evidence: list[Evidence] = []
    relevant_ids: set[str] = set()
    for i in range(12):
        is_relevant = i < 6
        evidence_id = f"{regime}-{i:02d}"
        if is_relevant:
            relevant_ids.add(evidence_id)
        text = (f"{spec['query']} {spec['query']} study {i}" if is_relevant
                else f"{spec['irrelevant']} unrelated note {i}")
        # Some relevant records are temporally invisible, exercising the cutoff.
        market_end = "2025-01-15" if i == 5 else "2024-06-30"
        known_at = "2024-07-01"
        evidence.append(Evidence(
            evidence_id, text, "synthetic_fixture", market_end, known_at,
            strategy, dict(params), quality=1.0 if is_relevant else 0.0,
            evidence_tier="fixture_relevance_label",
        ))
    # Later-known record must be excluded even though its market period is old.
    evidence.append(Evidence(
        f"{regime}-late-known", spec["query"], "synthetic_fixture",
        "2023-06-30", "2025-01-02", strategy, dict(params), 1.0,
        evidence_tier="fixture_relevance_label",
    ))
    relevant_ids.add(f"{regime}-late-known")
    return hypothesis, evidence, relevant_ids


def _rank(records: Iterable[Evidence], query: str, k: int) -> list[str]:
    from src.agent.hyde_retrieval import LexicalSimilarity

    scorer = LexicalSimilarity()
    ranked = sorted(records, key=lambda row: (-scorer(query, row.text), row.evidence_id))
    return [row.evidence_id for row in ranked[:k]]


def evaluate(regime: str, *, k: int = 5, seed: int = 23) -> dict[str, object]:
    hypothesis, evidence, relevant_ids = make_fixture(regime)
    request = HypothesisRequest(regime, (hypothesis.strategy_type,), AS_OF,
                                max_hypotheses=1, max_evidence=k)
    tool = HydeRetrievalTool(FixtureGenerator(hypothesis), evidence)
    candidate = tool.run(request)[0]
    visible_ids = {row.evidence_id for row in evidence
                   if row.market_end <= AS_OF and row.known_at <= AS_OF}
    relevant_visible = relevant_ids & visible_ids
    arms: dict[str, list[str]] = {
        "hyde": [row.evidence_id for row in candidate.evidence],
        "literature_keyword": [
            row.evidence_id for row in sorted(
                (row for row in evidence if row.evidence_id in visible_ids),
                key=lambda row: (-sum(word in row.text.lower()
                                      for word in REGIMES[regime]["keywords"]), row.evidence_id),
            )[:k]
        ],
        "grid_order": [row.evidence_id for row in evidence if row.evidence_id in visible_ids][:k],
    }
    shuffled = [row for row in evidence if row.evidence_id in visible_ids]
    random.Random(seed).shuffle(shuffled)
    arms["shuffled_evidence"] = [row.evidence_id for row in shuffled[:k]]
    arms["no_memory"] = []

    all_grid_params = [dict(p) for p in ParameterGrid().get_grid(hypothesis.strategy_type)]
    baseline_params = all_grid_params[:k]
    grid_set = {tuple(sorted(p.items())) for p in all_grid_params}
    output: dict[str, object] = {
        "regime": regime,
        "as_of": AS_OF,
        "candidate_budget_ceiling": k,
        "evidence_budget_per_candidate": k,
        "fixture": True,
        "evidence_tier": "fixture_relevance_label",
        "metrics": {},
        "candidate_novelty_vs_canonical_grid": int(
            tuple(sorted(candidate.params.items())) not in grid_set
        ),
        "time_cutoff_excluded_relevant": sorted(relevant_ids - visible_ids),
        "grid_candidates": baseline_params,
    }
    metrics: dict[str, object] = {}
    for name, ids in arms.items():
        hits = [i for i, evidence_id in enumerate(ids, start=1)
                if evidence_id in relevant_visible]
        metrics[name] = asdict(RankingMetrics(
            precision_at_k=len(hits) / k,
            reciprocal_rank=(1 / hits[0]) if hits else 0.0,
            relevant_retrieved=len(hits),
            novel_candidates=(1 if name == "hyde" and
                              tuple(sorted(candidate.params.items())) not in grid_set else 0),
            candidate_count=1 if name in ("hyde", "no_memory") else min(k, len(ids)),
        ))
    output["metrics"] = metrics
    output["hyde_candidate"] = {
        "params": dict(candidate.params),
        "retrieval_score": candidate.retrieval_score,
        "evaluation_status": candidate.evaluation_status,
        "evidence_ids": [item.evidence_id for item in candidate.evidence],
    }
    return output


def run_benchmark(k: int = 5, seed: int = 23) -> dict[str, object]:
    return {
        "benchmark": "hyde_retrieval_synthetic_fixture_v1",
        "seed": seed,
        "claim_scope": "retrieval ranking and candidate novelty only; no backtest utility measured",
        "results": [evaluate(regime, k=k, seed=seed) for regime in REGIMES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="shared top-k evidence limit")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    result = run_benchmark(k=args.k, seed=args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
