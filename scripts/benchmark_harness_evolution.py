#!/usr/bin/env python3
"""
Comprehensive Benchmarking of Harness Evolution Strategies

Compares:
1. Manual evolution (v1 → v2 → v3 → ... → v6)
2. Genetic Algorithm evolution
3. Differential Evolution
4. Random baseline

Produces detailed benchmark report with charts and metrics.

Usage:
    python3 scripts/benchmark_harness_evolution.py \
      --strategy momentum \
      --manual \
      --ga \
      --de \
      --output benchmark_report.json
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.harness_config import (
    HarnessConfig, get_harness_sequence, HarnessConfigManager
)
from src.agent.harness_evolution_algo import (
    EvolutionaryHarnessOptimizer, HarnessGenome
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvolutionStrategy:
    """Result of one evolution strategy."""
    name: str
    type: str  # "manual", "ga", "de", "random"
    harness_configs: List[HarnessConfig]
    fitness_scores: List[float]
    execution_time: float
    final_sharpe: float
    improvement_pct: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "configs": [c.to_dict() for c in self.harness_configs],
            "fitness_scores": self.fitness_scores,
            "execution_time": self.execution_time,
            "final_sharpe": self.final_sharpe,
            "improvement_pct": self.improvement_pct,
            "description": self.description,
        }


class HarnessBenchmark:
    """Comprehensive harness evolution benchmarking system."""

    def __init__(self, strategy: str = "momentum", asset: str = "SPY"):
        self.strategy = strategy
        self.asset = asset
        self.results: List[EvolutionStrategy] = []
        self.config_manager = HarnessConfigManager()

    def _mock_fitness_function(self, genome: HarnessGenome) -> float:
        """
        Mock fitness function for testing evolutionary algorithms.
        In production, this would run actual backtests.

        Simulates: Sharpe improves with tool usage and ensemble methods.
        """
        base_sharpe = 0.40

        # Tool usage helps
        if genome.use_tools:
            base_sharpe += 0.10

        # Web search helps (if tools enabled)
        if genome.use_web_search and genome.use_tools:
            base_sharpe += 0.05

        # Ensemble helps
        if genome.use_ensemble:
            base_sharpe += 0.08

        # Tool weighting: optimal around 0.6
        tool_penalty = abs(genome.tool_weight - 0.6) * 0.1
        base_sharpe -= tool_penalty

        # Temperature: lower is better but not too low
        temp_penalty = abs(genome.temperature - 0.15) * 0.1
        base_sharpe -= temp_penalty

        # Noise (simulates randomness in backtest)
        import random
        noise = random.gauss(0, 0.02)
        base_sharpe += noise

        return max(0.0, min(1.0, base_sharpe))

    def benchmark_manual_evolution(self) -> EvolutionStrategy:
        """Run manual evolution (v1 → v2 → ... → v6)."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARKING: Manual Evolution (v1 → v6)")
        logger.info("="*70)

        start_time = time.time()
        configs = get_harness_sequence()
        scores = []

        for config in configs:
            # Simulate backtest with this harness
            score = self._mock_fitness_function(
                HarnessGenome(
                    tool_weight=config.tool_weight,
                    temperature=config.temperature,
                    use_tools=config.use_tools,
                    use_web_search=config.use_web_search,
                    use_ensemble=config.use_ensemble,
                )
            )
            scores.append(score)
            logger.info(f"  {config.version}: Sharpe = {score:.3f}")
            # Save config
            self.config_manager.save_config(config)

        elapsed = time.time() - start_time
        improvement = (scores[-1] - scores[0]) / max(scores[0], 0.01) * 100

        return EvolutionStrategy(
            name="Manual Evolution (Hand-crafted)",
            type="manual",
            harness_configs=configs,
            fitness_scores=scores,
            execution_time=elapsed,
            final_sharpe=scores[-1],
            improvement_pct=improvement,
            description="v1 → v6 progression with strategic changes each epoch",
        )

    def benchmark_genetic_algorithm(self) -> EvolutionStrategy:
        """Run genetic algorithm evolution."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARKING: Genetic Algorithm")
        logger.info("="*70)

        start_time = time.time()

        optimizer = EvolutionaryHarnessOptimizer(algorithm="ga")
        best_genome = optimizer.optimize(
            fitness_fn=self._mock_fitness_function,
            population_size=20,
            generations=5,
        )

        elapsed = time.time() - start_time
        scores = optimizer.get_fitness_history()

        # Convert best genome to config
        best_config = best_genome.to_harness_config("v_ga_optimal", epoch=99)
        configs = [best_config]

        improvement = (scores[-1] - scores[0]) / max(scores[0], 0.01) * 100 if scores else 0

        return EvolutionStrategy(
            name="Genetic Algorithm",
            type="ga",
            harness_configs=configs,
            fitness_scores=scores,
            execution_time=elapsed,
            final_sharpe=best_genome.fitness,
            improvement_pct=improvement,
            description=f"GA: 20 population × 5 generations, found optimal harness",
        )

    def benchmark_differential_evolution(self) -> EvolutionStrategy:
        """Run differential evolution."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARKING: Differential Evolution")
        logger.info("="*70)

        start_time = time.time()

        optimizer = EvolutionaryHarnessOptimizer(algorithm="de")
        best_genome = optimizer.optimize(
            fitness_fn=self._mock_fitness_function,
            population_size=20,
            generations=5,
        )

        elapsed = time.time() - start_time
        scores = optimizer.get_fitness_history()

        # Convert best genome to config
        best_config = best_genome.to_harness_config("v_de_optimal", epoch=99)
        configs = [best_config]

        improvement = (scores[-1] - scores[0]) / max(scores[0], 0.01) * 100 if scores else 0

        return EvolutionStrategy(
            name="Differential Evolution",
            type="de",
            harness_configs=configs,
            fitness_scores=scores,
            execution_time=elapsed,
            final_sharpe=best_genome.fitness,
            improvement_pct=improvement,
            description=f"DE: 20 population × 5 generations, continuous optimization",
        )

    def benchmark_random_baseline(self) -> EvolutionStrategy:
        """Random harness configuration baseline."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARKING: Random Baseline (Control)")
        logger.info("="*70)

        start_time = time.time()
        scores = []

        for i in range(6):
            random_genome = HarnessGenome(
                tool_weight=__import__("random").random(),
                temperature=__import__("random").uniform(0.1, 0.5),
                use_tools=__import__("random").choice([True, False]),
                use_web_search=__import__("random").choice([True, False]),
                use_ensemble=__import__("random").choice([True, False]),
            )
            score = self._mock_fitness_function(random_genome)
            scores.append(score)
            logger.info(f"  Random {i+1}: Sharpe = {score:.3f}")

        elapsed = time.time() - start_time
        improvement = (scores[-1] - scores[0]) / max(scores[0], 0.01) * 100

        # Convert best to config
        best_idx = scores.index(max(scores))
        best_config = HarnessGenome().to_harness_config("v_random_best", epoch=99)

        return EvolutionStrategy(
            name="Random Baseline",
            type="random",
            harness_configs=[best_config],
            fitness_scores=scores,
            execution_time=elapsed,
            final_sharpe=max(scores),
            improvement_pct=improvement,
            description="6 random harness configurations (control)",
        )

    def run_all_benchmarks(self) -> None:
        """Run all benchmarking strategies."""
        logger.info("Starting comprehensive harness evolution benchmarking...")

        self.results.append(self.benchmark_manual_evolution())
        self.results.append(self.benchmark_genetic_algorithm())
        self.results.append(self.benchmark_differential_evolution())
        self.results.append(self.benchmark_random_baseline())

    def generate_report(self, output_file: str = "benchmark_report.json") -> None:
        """Generate comprehensive benchmark report."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK REPORT SUMMARY")
        logger.info("="*70)

        report = {
            "timestamp": datetime.now().isoformat(),
            "strategy": self.strategy,
            "asset": self.asset,
            "strategies": [r.to_dict() for r in self.results],
            "summary": self._compute_summary(),
        }

        # Log summary
        logger.info("\nStrategy Comparison:")
        for r in self.results:
            logger.info(f"  {r.name:<30} Sharpe: {r.final_sharpe:.3f}  Improvement: {r.improvement_pct:>7.1f}%  Time: {r.execution_time:>6.2f}s")

        # Find winner
        best = max(self.results, key=lambda r: r.final_sharpe)
        logger.info(f"\n🏆 Winner: {best.name} (Sharpe: {best.final_sharpe:.3f})")

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\n✓ Report saved to {output_file}")

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        return {
            "best_strategy": max(self.results, key=lambda r: r.final_sharpe).name,
            "best_sharpe": max(r.final_sharpe for r in self.results),
            "average_sharpe": sum(r.final_sharpe for r in self.results) / len(self.results),
            "fastest_strategy": min(self.results, key=lambda r: r.execution_time).name,
            "fastest_time": min(r.execution_time for r in self.results),
            "most_improved": max(self.results, key=lambda r: r.improvement_pct).name,
            "max_improvement": max(r.improvement_pct for r in self.results),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Harness Evolution Benchmarking")
    parser.add_argument("--strategy", default="momentum", help="Strategy type")
    parser.add_argument("--asset", default="SPY", help="Asset")
    parser.add_argument("--output", default="benchmark_report.json", help="Output file")

    args = parser.parse_args()

    benchmark = HarnessBenchmark(strategy=args.strategy, asset=args.asset)
    benchmark.run_all_benchmarks()
    benchmark.generate_report(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
