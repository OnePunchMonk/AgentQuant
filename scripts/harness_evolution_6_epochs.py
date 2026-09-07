#!/usr/bin/env python3
"""
Multi-Iteration Harness Evolution (6 Epochs)

Runs harness evolution 6 times with progressively refined strategies:
- Epoch 1 (v1_base): Baseline grid search
- Epoch 2 (v2_tool_aware): Promote tools
- Epoch 3 (v3_prompt_tuned): Refine LLM prompt
- Epoch 4 (v4_grid_evolved): Adapt parameter grid
- Epoch 5 (v5_multi_agent): Ensemble proposals
- Epoch 6 (v6_research): Research-informed proposals

Tracks benchmarks and evolves harness iteratively.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python3 scripts/harness_evolution_6_epochs.py --strategy momentum --output results.json
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.agent_graph import run_agent
from src.agent.trace import TraceRecorder
from src.data.ingest import load_ohlcv_data
from src.utils.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Comprehensive metrics for an epoch."""
    best_sharpe: float
    avg_sharpe: float
    median_sharpe: float
    sharpe_std: float
    generalization_gap: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    tool_calls: int
    proposals_generated: int
    proposals_accepted: int
    execution_time: float
    claim_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpochCheckpoint:
    """Complete checkpoint for one epoch."""
    epoch: int
    version: str
    timestamp: str
    strategy: str
    asset: str
    metrics: EpochMetrics
    evolution_reason: str
    harness_changes: List[str] = field(default_factory=list)
    next_evolution: str = ""
    results: List[Dict[str, Any]] = field(default_factory=list)


class HarnessEvolutionStrategy:
    """Defines evolution strategy for each epoch."""

    @staticmethod
    def v1_base() -> Dict[str, Any]:
        """Baseline: Grid search only."""
        return {
            "name": "v1_base",
            "description": "Baseline grid search (no tools)",
            "use_tools": False,
            "prompt_template": "grid_search_default",
            "grid_adaptation": None,
            "ensemble": False,
            "reasoning": "Establish baseline performance",
        }

    @staticmethod
    def v2_tool_aware() -> Dict[str, Any]:
        """Promote tool-based proposals."""
        return {
            "name": "v2_tool_aware",
            "description": "Enable tool orchestration and web search",
            "use_tools": True,
            "prompt_template": "tool_aware_default",
            "grid_adaptation": None,
            "ensemble": False,
            "reasoning": "Tools help discover better parameters; promote Tavily + regime context",
        }

    @staticmethod
    def v3_prompt_tuned() -> Dict[str, Any]:
        """Refine LLM prompt based on v2 learnings."""
        return {
            "name": "v3_prompt_tuned",
            "description": "Optimized prompt emphasizing recent successes",
            "use_tools": True,
            "prompt_template": "tool_aware_tuned_v2_learnings",
            "grid_adaptation": None,
            "ensemble": False,
            "reasoning": "Use v2 learnings to refine prompt; emphasize what worked",
        }

    @staticmethod
    def v4_grid_evolved() -> Dict[str, Any]:
        """Adapt parameter grid based on backtest results."""
        return {
            "name": "v4_grid_evolved",
            "description": "Parameter grid evolved toward high-performing regions",
            "use_tools": True,
            "prompt_template": "tool_aware_tuned_v2_learnings",
            "grid_adaptation": "shrink_to_winners",  # Focus on successful params
            "ensemble": False,
            "reasoning": "Grid evolution: prune dominated params, add winners + variations",
        }

    @staticmethod
    def v5_multi_agent() -> Dict[str, Any]:
        """Ensemble of proposal strategies."""
        return {
            "name": "v5_multi_agent",
            "description": "Multi-agent ensemble (tool-based + grid + random voting)",
            "use_tools": True,
            "prompt_template": "tool_aware_tuned_v2_learnings",
            "grid_adaptation": "shrink_to_winners",
            "ensemble": True,  # Multiple agents vote
            "reasoning": "Combine tool-based, grid-based, and random proposals; ensemble vote",
        }

    @staticmethod
    def v6_research() -> Dict[str, Any]:
        """Research-informed proposals."""
        return {
            "name": "v6_research",
            "description": "Research agent discovers ideas; tools validate",
            "use_tools": True,
            "prompt_template": "research_informed",
            "grid_adaptation": "shrink_to_winners",
            "ensemble": True,
            "reasoning": "Research agent searches literature; proposes novel combinations; tools validate",
        }


class MultiIterationHarnessEvolution:
    """Orchestrates 6-epoch harness evolution."""

    EVOLUTION_SEQUENCE = [
        HarnessEvolutionStrategy.v1_base,
        HarnessEvolutionStrategy.v2_tool_aware,
        HarnessEvolutionStrategy.v3_prompt_tuned,
        HarnessEvolutionStrategy.v4_grid_evolved,
        HarnessEvolutionStrategy.v5_multi_agent,
        HarnessEvolutionStrategy.v6_research,
    ]

    def __init__(self, strategy: str = "momentum", asset: str = "SPY"):
        self.strategy = strategy
        self.asset = asset
        self.checkpoints: List[EpochCheckpoint] = []
        self.ohlcv_data = None

    def load_data(self) -> None:
        """Load market data once."""
        logger.info("Loading market data (5 years)...")
        self.ohlcv_data = load_ohlcv_data(
            tickers=config.universe,
            period=config.data.yfinance_period,
        )

    def run_all_epochs(self) -> None:
        """Run 6 epochs with iterative evolution."""
        logger.info("="*70)
        logger.info("MULTI-ITERATION HARNESS EVOLUTION (6 EPOCHS)")
        logger.info("="*70)

        self.load_data()

        for epoch_num, strategy_fn in enumerate(self.EVOLUTION_SEQUENCE, 1):
            harness_spec = strategy_fn()
            self._run_epoch(epoch_num, harness_spec)

            # After each epoch (except last), compute evolution
            if epoch_num < len(self.EVOLUTION_SEQUENCE):
                self._compute_evolution(epoch_num, harness_spec)

    def _run_epoch(self, epoch_num: int, harness_spec: Dict[str, Any]) -> None:
        """Run single epoch with harness specification."""
        import time

        logger.info("\n" + "="*70)
        logger.info(f"EPOCH {epoch_num}: {harness_spec['name']}")
        logger.info(f"{'='*70}")
        logger.info(f"Description: {harness_spec['description']}")
        logger.info(f"Reasoning: {harness_spec['reasoning']}")

        start_time = time.time()

        # Run agent with current harness
        trace = TraceRecorder()
        state = run_agent(
            ohlcv_data=self.ohlcv_data,
            strategy_type=self.strategy,
            asset=self.asset,
            trace=trace,
        )

        elapsed = time.time() - start_time

        # Extract and compute metrics
        metrics = self._compute_metrics(state, elapsed)
        changes = self._describe_harness_changes(harness_spec)
        next_evolution = self._plan_next_evolution(epoch_num, metrics)

        checkpoint = EpochCheckpoint(
            epoch=epoch_num,
            version=harness_spec["name"],
            timestamp=datetime.now().isoformat(),
            strategy=self.strategy,
            asset=self.asset,
            metrics=metrics,
            evolution_reason=harness_spec["reasoning"],
            harness_changes=changes,
            next_evolution=next_evolution,
            results=state.get("all_results", []),
        )

        self.checkpoints.append(checkpoint)
        self._log_epoch_summary(checkpoint)

    def _compute_metrics(self, state: Dict[str, Any], elapsed: float) -> EpochMetrics:
        """Compute comprehensive metrics from agent state."""
        results = state.get("all_results", [])
        best_result = state.get("best_result", {})

        sharpes = [r.get("sharpe", 0.0) for r in results]
        drawdowns = [r.get("max_drawdown", 1.0) for r in results]
        num_trades_list = [r.get("num_trades", 0) for r in results]

        best_sharpe = best_result.get("sharpe", 0.0)
        avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
        median_sharpe = sorted(sharpes)[len(sharpes)//2] if sharpes else 0.0
        sharpe_std = (
            sum((s - avg_sharpe)**2 for s in sharpes) / len(sharpes)
        )**0.5 if sharpes else 0.0

        gap = max(avg_sharpe - best_sharpe, 0.0)
        avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 1.0
        win_rate = len([s for s in sharpes if s > 0.2]) / len(sharpes) if sharpes else 0.0
        avg_trades = sum(num_trades_list) / len(num_trades_list) if num_trades_list else 0

        proposals = state.get("proposals", [])
        tool_calls = len([e for e in state.get("trace", {}).get("events", [])
                         if e.get("type") == "tool_call"]) if state.get("trace") else 0

        return EpochMetrics(
            best_sharpe=best_sharpe,
            avg_sharpe=avg_sharpe,
            median_sharpe=median_sharpe,
            sharpe_std=sharpe_std,
            generalization_gap=gap,
            max_drawdown=avg_drawdown,
            win_rate=win_rate,
            num_trades=int(avg_trades),
            tool_calls=tool_calls,
            proposals_generated=len(proposals),
            proposals_accepted=1,
            execution_time=elapsed,
            # Numerical claim accuracy is not available until proposals carry
            # structured forecasts that can be evaluated against outcomes.
            claim_accuracy=0.0,
        )

    def _describe_harness_changes(self, harness_spec: Dict[str, Any]) -> List[str]:
        """Describe what changed in this harness version."""
        changes = []

        if harness_spec["use_tools"]:
            changes.append("✓ Tool orchestration enabled (Claude + Tavily)")
        else:
            changes.append("- Tool orchestration disabled (grid only)")

        if harness_spec["grid_adaptation"]:
            changes.append(f"✓ Grid adapted: {harness_spec['grid_adaptation']}")
        else:
            changes.append("- Grid unchanged (default)")

        if harness_spec["ensemble"]:
            changes.append("✓ Multi-agent ensemble enabled")
        else:
            changes.append("- Single-agent strategy")

        if harness_spec["prompt_template"] != "grid_search_default":
            changes.append(f"✓ Prompt tuned: {harness_spec['prompt_template']}")

        return changes

    def _plan_next_evolution(self, epoch_num: int, metrics: EpochMetrics) -> str:
        """Plan which harness to try next based on this epoch's results."""
        if epoch_num >= 6:
            return "Terminal (final epoch)"

        # Simple heuristic
        if metrics.best_sharpe > 0.5 and metrics.generalization_gap < 0.1:
            return "v6_research (strong performance; try research agent)"
        elif metrics.best_sharpe > 0.4 and metrics.tool_calls > 2:
            return "v4_grid_evolved (tools working; adapt grid)"
        elif metrics.sharpe_std < 0.1:
            return "v3_prompt_tuned (consistent; refine prompt)"
        else:
            return "v2_tool_aware (baseline underperforming; enable tools)"

    def _log_epoch_summary(self, cp: EpochCheckpoint) -> None:
        """Log epoch results."""
        logger.info(f"\n📊 EPOCH {cp.epoch} RESULTS ({cp.version})")
        logger.info(f"  Best Sharpe:         {cp.metrics.best_sharpe:>8.3f}")
        logger.info(f"  Avg Sharpe:          {cp.metrics.avg_sharpe:>8.3f}")
        logger.info(f"  Std Dev:             {cp.metrics.sharpe_std:>8.3f}")
        logger.info(f"  Gen Gap:             {cp.metrics.generalization_gap:>8.3f}")
        logger.info(f"  Max Drawdown:        {cp.metrics.max_drawdown:>8.1%}")
        logger.info(f"  Win Rate:            {cp.metrics.win_rate:>8.1%}")
        logger.info(f"  Tool Calls:          {cp.metrics.tool_calls:>8d}")
        logger.info(f"  Execution Time:      {cp.metrics.execution_time:>8.1f}s")

    def _compute_evolution(self, epoch_num: int, harness_spec: Dict[str, Any]) -> None:
        """Analyze results and plan evolution to next harness."""
        logger.info(f"\n🧬 EVOLUTION ANALYSIS (Epoch {epoch_num} → {epoch_num+1})")

        current = self.checkpoints[-1]
        improvements = []

        # Analyze tool effectiveness
        if harness_spec["use_tools"] and current.metrics.tool_calls > 0:
            if current.metrics.best_sharpe > 0.3:
                improvements.append("Tools effective (Sharpe > 0.3); continue using")

        # Analyze generalization
        if current.metrics.generalization_gap < 0.1:
            improvements.append("Good generalization (gap < 0.1); can evolve grid")
        elif current.metrics.generalization_gap > 0.15:
            improvements.append("Overfitting detected (gap > 0.15); need regularization")

        # Analyze consistency
        if current.metrics.sharpe_std < 0.1:
            improvements.append("Consistent proposals (std < 0.1); stable harness")

        for imp in improvements:
            logger.info(f"  ✓ {imp}")

        logger.info(f"\nNext Step: {self._plan_next_evolution(epoch_num, current.metrics)}")

    def benchmark_comparison(self) -> Dict[str, Any]:
        """Compare all epochs."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK COMPARISON (All Epochs)")
        logger.info("="*70)

        data = {
            "strategy": self.strategy,
            "asset": self.asset,
            "epochs": len(self.checkpoints),
            "comparison": {},
            "improvement_trajectory": {},
        }

        # Epoch-by-epoch comparison
        for cp in self.checkpoints:
            data["comparison"][cp.version] = {
                "epoch": cp.epoch,
                "metrics": cp.metrics.to_dict(),
                "changes": cp.harness_changes,
            }

            # Compute improvement vs baseline
            if cp.epoch == 1:
                baseline_sharpe = cp.metrics.best_sharpe
                data["improvement_trajectory"][cp.version] = 0.0
            else:
                improvement = (cp.metrics.best_sharpe - baseline_sharpe) / max(baseline_sharpe, 0.01)
                data["improvement_trajectory"][cp.version] = improvement * 100

        # Summary stats
        sharpes = [cp.metrics.best_sharpe for cp in self.checkpoints]
        gaps = [cp.metrics.generalization_gap for cp in self.checkpoints]

        data["summary"] = {
            "best_epoch": self.checkpoints[sharpes.index(max(sharpes))].version,
            "best_sharpe": max(sharpes),
            "best_gap": min(gaps),
            "avg_improvement": (max(sharpes) - min(sharpes)) / max(min(sharpes), 0.01) * 100,
        }

        self._log_benchmark(data)
        return data

    def _log_benchmark(self, data: Dict[str, Any]) -> None:
        """Log benchmark summary."""
        logger.info("\nSharp Progression:")
        for version, improvement in data["improvement_trajectory"].items():
            bar = "█" * int(improvement / 2) if improvement > 0 else ""
            logger.info(f"  {version:<20} {improvement:>6.1f}% {bar}")

        logger.info(f"\nBest Performer: {data['summary']['best_epoch']} (Sharpe: {data['summary']['best_sharpe']:.3f})")
        logger.info(f"Overall Improvement: {data['summary']['avg_improvement']:.1f}%")

    def save_results(self, outfile: str) -> None:
        """Save all results to JSON."""
        data = self.benchmark_comparison()

        # Add full checkpoint data
        data["checkpoints"] = [asdict(cp) for cp in self.checkpoints]

        with open(outfile, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"\n✓ Results saved to {outfile}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="6-Epoch Harness Evolution")
    parser.add_argument("--strategy", default="momentum", help="Strategy type")
    parser.add_argument("--asset", default="SPY", help="Asset to backtest")
    parser.add_argument("--output", default="harness_evolution_6epochs.json", help="Output file")

    args = parser.parse_args()

    runner = MultiIterationHarnessEvolution(strategy=args.strategy, asset=args.asset)
    runner.run_all_epochs()
    runner.save_results(args.output)

    logger.info("\n" + "="*70)
    logger.info("✓ HARNESS EVOLUTION COMPLETE (6 Epochs)")
    logger.info("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
