"""
Swarm State — Shared State TypedDict for the Agent Swarm
=========================================================

All agents read from and write to a single SwarmState dict.
This makes the system trivially convertible to LangGraph StateGraph
when Phase 4 is implemented (each node is already a pure function of state).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from src.agent.context_builder import RegimeContext
from src.agent.proposal_generator import Proposal


class SwarmState(TypedDict, total=False):
    """Shared state flowing through all swarm agents."""

    # ── Input ─────────────────────────────────────────────────────────────
    ohlcv_data: Dict[str, pd.DataFrame]
    assets: List[str]
    strategy_types: List[str]      # which specialists to run

    # ── Regime Analyst outputs ────────────────────────────────────────────
    features_df: pd.DataFrame
    regime_context: Optional[RegimeContext]
    regime_narrative: str           # rich natural-language regime description
    macro_summary: str              # FRED macro context (if available)

    # ── Strategy Specialists outputs ──────────────────────────────────────
    # key = specialist name e.g. "momentum", "mean_reversion"
    specialist_proposals: Dict[str, List[Proposal]]
    all_proposals: List[Proposal]   # aggregated pool from all specialists

    # ── Critic Agent outputs ──────────────────────────────────────────────
    approved_proposals: List[Proposal]
    rejected_count: int
    rejection_log: List[Dict[str, str]]   # {proposal_str, reason}

    # ── Backtest Coordinator outputs ──────────────────────────────────────
    # List of per-window results: [{proposal_key, window_label, sharpe, ...}, ...]
    window_results: List[Dict[str, Any]]
    # Final ranked proposals: [{params, mean_sharpe, sharpe_std, min_sharpe, robustness}, ...]
    final_ranking: List[Dict[str, Any]]
    best_result: Optional[Dict[str, Any]]

    # ── Memory Agent outputs ──────────────────────────────────────────────
    memory_patterns: List[str]      # natural-language patterns from history
    memory_context: str             # formatted context for LLM injection

    # ── Run metadata ──────────────────────────────────────────────────────
    run_log: List[str]
    iteration: int


@dataclass
class SwarmResult:
    """Structured result returned by SwarmOrchestrator.run()."""
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_strategy_type: str = ""
    mean_sharpe: float = 0.0
    min_sharpe: float = 0.0
    sharpe_std: float = 0.0
    robustness_score: float = 0.0
    total_proposals_generated: int = 0
    proposals_approved: int = 0
    proposals_rejected: int = 0
    n_windows_tested: int = 0
    regime_label: str = ""
    regime_confidence: float = 0.0
    generation_methods: Dict[str, int] = field(default_factory=dict)
    run_log: List[str] = field(default_factory=list)
    full_ranking: List[Dict[str, Any]] = field(default_factory=list)
    memory_patterns: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Swarm Result ===",
            f"Regime:       {self.regime_label} (conf={self.regime_confidence:.0%})",
            f"Best strategy:{self.best_strategy_type} {self.best_params}",
            f"Sharpe (mean/min/std): {self.mean_sharpe:.3f} / {self.min_sharpe:.3f} / {self.sharpe_std:.3f}",
            f"Robustness:   {self.robustness_score:.3f}",
            f"Windows:      {self.n_windows_tested}",
            f"Proposals:    {self.total_proposals_generated} generated → {self.proposals_approved} approved → {self.n_windows_tested}-window test",
            f"Methods:      {self.generation_methods}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_strategy_type": self.best_strategy_type,
            "mean_sharpe": self.mean_sharpe,
            "min_sharpe": self.min_sharpe,
            "sharpe_std": self.sharpe_std,
            "robustness_score": self.robustness_score,
            "total_proposals": self.total_proposals_generated,
            "proposals_approved": self.proposals_approved,
            "proposals_rejected": self.proposals_rejected,
            "n_windows": self.n_windows_tested,
            "regime_label": self.regime_label,
            "regime_confidence": self.regime_confidence,
            "generation_methods": self.generation_methods,
        }
