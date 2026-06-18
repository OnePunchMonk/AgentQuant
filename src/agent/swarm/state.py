"""Shared state and result objects for the AgentQuant swarm."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from src.agent.context_builder import RegimeContext
from src.agent.proposal_generator import Proposal


class SwarmState(TypedDict, total=False):
    """State passed between specialized swarm agents."""

    ohlcv_data: Dict[str, pd.DataFrame]
    assets: List[str]
    strategy_types: List[str]
    features_df: pd.DataFrame
    regime_context: Optional[RegimeContext]
    regime_narrative: str
    macro_summary: str
    specialist_proposals: Dict[str, List[Proposal]]
    all_proposals: List[Proposal]
    approved_proposals: List[Proposal]
    rejected_count: int
    rejection_log: List[Dict[str, str]]
    window_results: List[Dict[str, Any]]
    final_ranking: List[Dict[str, Any]]
    best_result: Optional[Dict[str, Any]]
    memory_patterns: List[str]
    memory_context: str
    run_log: List[str]


@dataclass
class SwarmResult:
    """Structured result returned by SwarmOrchestrator."""

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
    regime_label: str = "Unknown"
    regime_confidence: float = 0.0
    regime_narrative: str = ""
    generation_methods: Dict[str, int] = field(default_factory=dict)
    run_log: List[str] = field(default_factory=list)
    full_ranking: List[Dict[str, Any]] = field(default_factory=list)
    memory_patterns: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join(
            [
                "=== Swarm Result ===",
                f"Regime: {self.regime_label} (confidence={self.regime_confidence:.0%})",
                f"Best strategy: {self.best_strategy_type} {self.best_params}",
                (
                    "Sharpe mean/min/std: "
                    f"{self.mean_sharpe:.3f} / {self.min_sharpe:.3f} / {self.sharpe_std:.3f}"
                ),
                f"Robustness: {self.robustness_score:.3f}",
                f"Windows tested: {self.n_windows_tested}",
                (
                    "Proposals: "
                    f"{self.total_proposals_generated} generated, "
                    f"{self.proposals_approved} approved, "
                    f"{self.proposals_rejected} rejected"
                ),
                f"Methods: {self.generation_methods}",
            ]
        )

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
