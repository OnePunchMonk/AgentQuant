"""Conservative, deterministic preflight for harness edit proposals.

This module scores descriptors only. It does not read, apply, or promote code
changes. The design is RRSI-inspired rather than a reproduction of that work.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EditProposal:
    component: str
    diff_paths: tuple[str, ...]
    content_metadata: str = ""
    estimated_cost: float = 0.0
    benchmark_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreflightResult:
    accepted_for_evaluation: bool
    reasons: tuple[str, ...]
    fingerprint: str
    novelty: float
    effective_budget: int
    adjusted_floor: float


@dataclass(frozen=True)
class RegularizerConfig:
    initial_edit_budget: int = 4
    final_edit_budget: int = 1
    leakage_tokens: frozenset[str] = frozenset()
    forbidden_path_prefixes: tuple[str, ...] = (
        ".git/", ".env", "secrets/", "data/", "experiments/",
    )
    min_effect_floor: float = 0.01
    cost_floor_scale: float = 0.001
    noise_floor_scale: float = 1.0
    max_cost: float = 100.0


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9_./-]+", "", token.lower()).strip("./-")


def _fingerprint(proposal: EditProposal) -> str:
    material = "\n".join((proposal.component.strip().lower(),
                            *sorted(p.strip().replace("\\", "/") for p in proposal.diff_paths),
                            proposal.content_metadata.strip()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _effective_budget(config: RegularizerConfig, progress: float) -> int:
    """Linearly anneal the maximum path count from initial to final budget."""
    p = min(1.0, max(0.0, progress))
    raw = config.initial_edit_budget + p * (config.final_edit_budget - config.initial_edit_budget)
    return max(0, math.floor(raw + 0.5))


def preflight_edit(
    proposal: EditProposal,
    *,
    progress: float = 0.0,
    history: Iterable[EditProposal] = (),
    measurement_noise: float = 0.0,
    config: RegularizerConfig = RegularizerConfig(),
) -> PreflightResult:
    """Decide whether a descriptor may proceed to isolated evaluation.

    `progress` is normalized to [0, 1]. The effect floor rises with estimated
    cost and observed noise. Benchmark tokens matching configured leakage
    markers, structural paths, empty descriptors, and repeated edits abstain.
    """
    past = tuple(history)
    fingerprint = _fingerprint(proposal)
    seen = {_fingerprint(item) for item in past}
    novelty = 0.0 if fingerprint in seen else 1.0
    budget = _effective_budget(config, progress)
    floor = max(config.min_effect_floor,
                config.cost_floor_scale * max(0.0, proposal.estimated_cost),
                config.noise_floor_scale * max(0.0, measurement_noise))
    reasons: list[str] = []
    paths = tuple(p.replace("\\", "/").lstrip("./") for p in proposal.diff_paths)
    if not proposal.component.strip():
        reasons.append("missing_component")
    if not paths:
        reasons.append("empty_diff_paths")
    if len(paths) > budget:
        reasons.append("annealed_edit_budget_exceeded")
    if proposal.estimated_cost < 0 or proposal.estimated_cost > config.max_cost:
        reasons.append("cost_out_of_bounds")
    if any(any(path == prefix.rstrip("/") or path.startswith(prefix)
               for prefix in config.forbidden_path_prefixes) for path in paths):
        reasons.append("structural_path_pruned")
    if novelty == 0.0:
        reasons.append("duplicate_in_history")
    markers = {_normalize_token(t) for t in config.leakage_tokens}
    candidate_tokens = {_normalize_token(t) for t in proposal.benchmark_tokens}
    if markers & candidate_tokens:
        reasons.append("benchmark_specific_leakage_marker")
    return PreflightResult(not reasons, tuple(reasons), fingerprint, novelty, budget, floor)


@dataclass(frozen=True)
class SelectionResult:
    selected: bool
    reason: str
    mean_paired_gain: float | None = None
    required_floor: float | None = None
    pair_count: int = 0


def select_with_paired_evidence(
    paired_differences: Sequence[float] | None,
    *,
    adjusted_floor: float,
    minimum_pairs: int = 5,
) -> SelectionResult:
    """Evidence helper; abstains unless paired gains clear a conservative bar.

    Selection here is a research decision signal only, never a promotion or
    filesystem mutation. Requiring every paired difference to be positive
    limits selection when evidence is small or heterogeneous.
    """
    if paired_differences is None or len(paired_differences) < minimum_pairs:
        return SelectionResult(False, "insufficient_paired_evidence", pair_count=0 if paired_differences is None else len(paired_differences))
    values = tuple(float(x) for x in paired_differences)
    if not all(math.isfinite(x) for x in values):
        return SelectionResult(False, "non_finite_evidence", pair_count=len(values))
    mean_gain = sum(values) / len(values)
    if any(x <= 0 for x in values):
        return SelectionResult(False, "paired_regression_present", mean_gain, adjusted_floor, len(values))
    if mean_gain <= adjusted_floor:
        return SelectionResult(False, "gain_below_cost_noise_floor", mean_gain, adjusted_floor, len(values))
    return SelectionResult(True, "paired_evidence_clears_floor", mean_gain, adjusted_floor, len(values))
