"""
Evaluation Suite for Harness Evolution

Exposes an eval tool that agents can call to assess:
- Strategy performance (Sharpe, returns, drawdown)
- Generalization gap (train vs held-out performance)
- Falsifiable claim accuracy
- Tool effectiveness
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result from an evaluation run."""
    name: str
    score: float  # 0.0-1.0
    metrics: Dict[str, Any]
    timestamp: str
    passed: bool


class HarnessEvalSuite:
    """Evaluation suite for assessing harness quality and improvement."""

    def __init__(self):
        self.results: List[EvalResult] = []

    def run_benchmark_to_assess_quality(
        self,
        backtest_results: Optional[List[Dict[str, Any]]] = None,
        strategy_type: str = "",
        regime: str = "",
    ) -> Dict[str, Any]:
        """
        Run a quality assessment benchmark on recent harness performance.

        Measures:
        - Out-of-sample Sharpe ratio (primary metric)
        - Stability across regimes
        - Generalization gap (if train/val/test splits available)
        - Tool usage efficiency
        - Falsifiable claim accuracy

        Args:
            backtest_results: Recent backtest results to evaluate
            strategy_type: Strategy being evaluated
            regime: Market regime

        Returns:
            {
                "overall_score": 0.0-1.0,
                "passed": bool,
                "metrics": {
                    "oos_sharpe": float,
                    "stability": float,
                    "generalization_gap": float,
                    "tool_accuracy": float,
                },
                "thresholds": {
                    "min_acceptable_sharpe": float,
                    "max_acceptable_drawdown": float,
                },
                "recommendation": str,
            }
        """
        from src.utils.config import config
        from src.research.alpha_store import AlphaStore

        logger.info(f"Running quality assessment benchmark for {strategy_type} in {regime}")

        # Gather recent results from alpha store
        alpha_store = AlphaStore()
        recent_candidates = alpha_store.list_recent(n=25)

        # Compute metrics
        metrics = {}

        # 1. Out-of-sample Sharpe
        sharpe_scores = [
            c.metrics.get("sharpe", 0.0) for c in recent_candidates if c.metrics
        ]
        oos_sharpe = sum(sharpe_scores) / len(sharpe_scores) if sharpe_scores else 0.0
        metrics["oos_sharpe"] = oos_sharpe

        # 2. Stability (lower std = more stable)
        if len(sharpe_scores) > 1:
            import statistics

            stability = 1.0 - min(statistics.stdev(sharpe_scores) / (oos_sharpe + 0.01), 1.0)
            metrics["stability"] = stability
        else:
            metrics["stability"] = 0.5

        # 3. Generalization gap (train - test; lower is better)
        generalization_gaps = []
        for candidate in recent_candidates:
            if "generalization_gap" in candidate.metadata:
                generalization_gaps.append(candidate.metadata["generalization_gap"])

        if generalization_gaps:
            avg_gap = sum(generalization_gaps) / len(generalization_gaps)
            metrics["generalization_gap"] = 1.0 - min(avg_gap, 0.5) / 0.5  # Normalize to 0-1
        else:
            metrics["generalization_gap"] = 0.5  # Unknown

        # 4. Tool accuracy (% of falsifiable claims that materialized)
        tool_accuracy = _compute_tool_accuracy(recent_candidates)
        metrics["tool_accuracy"] = tool_accuracy

        # 5. Max drawdown check
        max_drawdowns = [
            c.metrics.get("max_drawdown", 1.0) for c in recent_candidates if c.metrics
        ]
        avg_drawdown = sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 1.0
        metrics["avg_max_drawdown"] = avg_drawdown

        # Compute overall score
        min_acceptable_sharpe = config.agent.min_acceptable_sharpe
        min_acceptable_dd = config.agent.risk.max_drawdown

        passed = oos_sharpe >= min_acceptable_sharpe and avg_drawdown <= min_acceptable_dd

        # Weighted score
        overall_score = (
            0.4 * min(oos_sharpe / max(min_acceptable_sharpe, 0.5), 1.0)
            + 0.2 * metrics["stability"]
            + 0.2 * metrics["generalization_gap"]
            + 0.2 * metrics["tool_accuracy"]
        )
        overall_score = min(overall_score, 1.0)

        return {
            "overall_score": overall_score,
            "passed": passed,
            "metrics": metrics,
            "thresholds": {
                "min_acceptable_sharpe": min_acceptable_sharpe,
                "max_acceptable_drawdown": min_acceptable_dd,
            },
            "recommendation": _generate_recommendation(
                passed, oos_sharpe, metrics, strategy_type, regime
            ),
        }

    def checkpoint_replay(
        self,
        harness_checkpoints: Optional[List[Dict[str, Any]]] = None,
        test_window: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Replay past harness checkpoints on held-out test data.

        Measures how much harness evolution actually generalizes.
        Returns the % of updates that improved training but hurt held-out performance.

        Args:
            harness_checkpoints: Past harness versions to replay
            test_window: Held-out test window identifier

        Returns:
            {
                "total_checkpoints": int,
                "improved_training": int,
                "improved_held_out": int,
                "hurt_transfer": float,  # % that improved train but hurt test
                "generalization_rate": float,  # % that improved both
            }
        """
        logger.info("Running checkpoint replay evaluation")

        return {
            "total_checkpoints": 0,
            "improved_training": 0,
            "improved_held_out": 0,
            "hurt_transfer": 0.0,
            "generalization_rate": 0.0,
            "note": "Checkpoint replay requires harness version history",
        }


def _compute_tool_accuracy(candidates: List) -> float:
    """Compute accuracy of falsifiable claims from tool calls."""
    accurate = 0
    total = 0

    for candidate in candidates:
        if "falsifiable_claim" in candidate.metadata:
            # Check if predicted impact matched realized outcome
            predicted = candidate.metadata.get("predicted_impact", "")
            actual_sharpe = candidate.metrics.get("sharpe", 0.0)

            # Simple heuristic: if claim was positive and actual is positive, mark accurate
            if predicted and actual_sharpe > 0.2:
                accurate += 1
            total += 1

    if total == 0:
        return 0.5  # Unknown

    return accurate / total


def _generate_recommendation(
    passed: bool, sharpe: float, metrics: Dict[str, Any], strategy_type: str, regime: str
) -> str:
    """Generate actionable recommendation based on eval results."""
    if passed:
        if sharpe > 0.8:
            return f"Harness performing excellently in {regime} {strategy_type}. Consider scaling or deploying."
        else:
            return f"Harness passed thresholds in {regime} {strategy_type}. Continue monitoring."
    else:
        if sharpe < 0.1:
            return f"Sharpe ratio too low ({sharpe:.2f}). Retry hypothesis generation with different parameters or web search for market context."
        if metrics.get("generalization_gap", 0) < 0.3:
            return f"High overfitting detected (gap={metrics['generalization_gap']:.2f}). Reduce grid size or simplify proposal logic."
        else:
            return f"Harness underperforming. Review tool accuracy ({metrics['tool_accuracy']:.0%}) and consider prompt edits."


# Export the eval suite tool interface
eval_suite = HarnessEvalSuite()


def run_benchmark_to_assess_quality_tool(
    strategy_type: str = "",
    regime: str = "",
) -> Dict[str, Any]:
    """
    Tool interface for agents to call quality assessment.

    This is exposed as a tool in the registry so agents can:
    1. Check if their proposals are working
    2. Get recommendations for next steps
    3. Measure generalization
    """
    return eval_suite.run_benchmark_to_assess_quality(
        strategy_type=strategy_type,
        regime=regime,
    )
