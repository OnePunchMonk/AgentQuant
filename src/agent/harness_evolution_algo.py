"""
Evolutionary Algorithm for Harness Evolution

Uses genetic algorithms and differential evolution to optimize harness configurations.
Population-based search for better evolution strategies.

Techniques:
- Genetic Algorithm: Crossover + Mutation of harness configs
- Differential Evolution: Perturbation-based search
- Particle Swarm: Population-guided exploration
"""

import logging
import random
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Tuple

from src.agent.harness_config import HarnessConfig

logger = logging.getLogger(__name__)


@dataclass
class HarnessGenome:
    """Genetic representation of a harness configuration."""

    # Continuous parameters
    tool_weight: float = 0.5  # [0.0, 1.0]
    temperature: float = 0.2  # [0.0, 1.0]
    claim_weighting: float = 0.5  # [0.0, 1.0]

    # Discrete parameters
    use_tools: bool = False
    use_web_search: bool = False
    use_ensemble: bool = False
    grid_adaptation_strategy: Optional[str] = None

    # Fitness
    fitness: float = 0.0
    generation: int = 0

    def to_harness_config(self, version: str, epoch: int) -> HarnessConfig:
        """Convert genome to HarnessConfig."""
        from datetime import datetime

        config = HarnessConfig(
            version=version,
            epoch=epoch,
            created=datetime.now().isoformat(),
            use_tools=self.use_tools,
            use_web_search=self.use_web_search and self.use_tools,
            use_ensemble=self.use_ensemble,
            tool_weight=self.tool_weight,
            temperature=self.temperature,
            claim_weighting=self.claim_weighting,
            grid_adaptation_strategy=self.grid_adaptation_strategy,
            track_falsifiable_claims=self.use_tools,
        )
        return config

    def mutate(self, mutation_rate: float = 0.1) -> "HarnessGenome":
        """Mutate genome (create offspring)."""
        mutant = asdict(self)
        mutant.pop("fitness")
        mutant.pop("generation")

        # Continuous mutation (Gaussian)
        if random.random() < mutation_rate:
            mutant["tool_weight"] = max(0.0, min(1.0, mutant["tool_weight"] + random.gauss(0, 0.1)))
        if random.random() < mutation_rate:
            mutant["temperature"] = max(0.0, min(1.0, mutant["temperature"] + random.gauss(0, 0.1)))
        if random.random() < mutation_rate:
            mutant["claim_weighting"] = max(0.0, min(1.0, mutant["claim_weighting"] + random.gauss(0, 0.1)))

        # Discrete mutation (flip)
        if random.random() < mutation_rate * 0.5:
            mutant["use_tools"] = not mutant["use_tools"]
        if random.random() < mutation_rate * 0.5:
            mutant["use_ensemble"] = not mutant["use_ensemble"]

        return HarnessGenome(**mutant)

    @staticmethod
    def crossover(parent1: "HarnessGenome", parent2: "HarnessGenome") -> Tuple["HarnessGenome", "HarnessGenome"]:
        """Crossover two genomes (uniform crossover)."""
        child1_dict = asdict(parent1)
        child2_dict = asdict(parent2)

        for key in child1_dict:
            if key not in ["fitness", "generation"] and random.random() < 0.5:
                child1_dict[key], child2_dict[key] = child2_dict[key], child1_dict[key]

        child1_dict.pop("fitness")
        child2_dict.pop("fitness")
        child1_dict.pop("generation")
        child2_dict.pop("generation")

        return HarnessGenome(**child1_dict), HarnessGenome(**child2_dict)


class GeneticAlgorithm:
    """Genetic Algorithm for harness evolution."""

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 5,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.2,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.population: List[HarnessGenome] = []
        self.best_genome: Optional[HarnessGenome] = None
        self.fitness_history: List[float] = []

    def initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            genome = HarnessGenome(
                tool_weight=random.random(),
                temperature=random.uniform(0.1, 0.5),
                claim_weighting=random.random(),
                use_tools=random.choice([True, False]),
                use_web_search=random.choice([True, False]),
                use_ensemble=random.choice([True, False]),
                grid_adaptation_strategy=random.choice([None, "shrink_to_winners", "expand_neighborhood"]),
            )
            self.population.append(genome)

    def evaluate_population(self, fitness_fn: Callable[[HarnessGenome], float]) -> None:
        """Evaluate fitness of all individuals."""
        for genome in self.population:
            genome.fitness = fitness_fn(genome)

        # Track best
        best = max(self.population, key=lambda g: g.fitness)
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best

        avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
        self.fitness_history.append(avg_fitness)

        logger.info(f"Generation fitness: best={best.fitness:.3f}, avg={avg_fitness:.3f}")

    def selection(self, tournament_size: int = 3) -> HarnessGenome:
        """Tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda g: g.fitness)

    def evolve(self, fitness_fn: Callable[[HarnessGenome], float]) -> HarnessGenome:
        """Run genetic algorithm."""
        logger.info(f"Starting GA: population={self.population_size}, generations={self.generations}")

        self.initialize_population()

        for gen in range(self.generations):
            logger.info(f"\n--- Generation {gen + 1}/{self.generations} ---")

            # Evaluate
            self.evaluate_population(fitness_fn)

            # Selection & Reproduction
            new_population = []

            # Elitism
            elite_count = int(self.population_size * self.elitism_rate)
            elite = sorted(self.population, key=lambda g: g.fitness, reverse=True)[:elite_count]
            new_population.extend(elite)

            # Crossover & Mutation
            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                child1, child2 = HarnessGenome.crossover(parent1, parent2)
                child1 = child1.mutate(self.mutation_rate)
                child2 = child2.mutate(self.mutation_rate)
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

        return self.best_genome


class DifferentialEvolution:
    """Differential Evolution for harness parameter optimization."""

    def __init__(self, population_size: int = 20, generations: int = 5, f: float = 0.8, cr: float = 0.9):
        self.population_size = population_size
        self.generations = generations
        self.f = f  # Mutation factor
        self.cr = cr  # Crossover probability
        self.population: List[HarnessGenome] = []
        self.best_genome: Optional[HarnessGenome] = None
        self.fitness_history: List[float] = []

    def initialize_population(self) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            genome = HarnessGenome(
                tool_weight=random.random(),
                temperature=random.uniform(0.1, 0.5),
                claim_weighting=random.random(),
                use_tools=random.choice([True, False]),
            )
            self.population.append(genome)

    def mutate_de(self, x_target: HarnessGenome, x_a: HarnessGenome, x_b: HarnessGenome, x_c: HarnessGenome) -> HarnessGenome:
        """DE/rand/1 mutation."""
        mutant = HarnessGenome()
        mutant.tool_weight = x_a.tool_weight + self.f * (x_b.tool_weight - x_c.tool_weight)
        mutant.temperature = x_a.temperature + self.f * (x_b.temperature - x_c.temperature)
        mutant.claim_weighting = x_a.claim_weighting + self.f * (x_b.claim_weighting - x_c.claim_weighting)

        # Clamp to valid ranges
        mutant.tool_weight = max(0.0, min(1.0, mutant.tool_weight))
        mutant.temperature = max(0.0, min(1.0, mutant.temperature))
        mutant.claim_weighting = max(0.0, min(1.0, mutant.claim_weighting))

        return mutant

    def crossover_de(self, target: HarnessGenome, mutant: HarnessGenome) -> HarnessGenome:
        """Binomial crossover."""
        trial = HarnessGenome()
        for attr in ["tool_weight", "temperature", "claim_weighting"]:
            if random.random() < self.cr:
                setattr(trial, attr, getattr(mutant, attr))
            else:
                setattr(trial, attr, getattr(target, attr))

        trial.use_tools = target.use_tools  # Keep discrete params for now
        trial.use_ensemble = target.use_ensemble

        return trial

    def evolve(self, fitness_fn: Callable[[HarnessGenome], float]) -> HarnessGenome:
        """Run Differential Evolution."""
        logger.info(f"Starting DE: population={self.population_size}, generations={self.generations}")

        self.initialize_population()

        # Initial evaluation
        for genome in self.population:
            genome.fitness = fitness_fn(genome)

        self.best_genome = max(self.population, key=lambda g: g.fitness)

        for gen in range(self.generations):
            logger.info(f"\n--- DE Generation {gen + 1}/{self.generations} ---")

            for i in range(self.population_size):
                # Select three random individuals (different from i)
                candidates = [j for j in range(self.population_size) if j != i]
                a, b, c = random.sample(candidates, 3)

                # Mutation
                mutant = self.mutate_de(self.population[i], self.population[a], self.population[b], self.population[c])

                # Crossover
                trial = self.crossover_de(self.population[i], mutant)

                # Evaluation
                trial.fitness = fitness_fn(trial)

                # Selection
                if trial.fitness > self.population[i].fitness:
                    self.population[i] = trial

                # Track best
                if trial.fitness > self.best_genome.fitness:
                    self.best_genome = trial

            avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
            self.fitness_history.append(avg_fitness)
            logger.info(f"Best fitness: {self.best_genome.fitness:.3f}, Avg: {avg_fitness:.3f}")

        return self.best_genome


class EvolutionaryHarnessOptimizer:
    """Meta-optimizer for harness evolution using evolutionary algorithms."""

    def __init__(self, algorithm: str = "ga"):
        self.algorithm = algorithm
        self.algorithm_instance = None

    def optimize(
        self,
        fitness_fn: Callable[[HarnessGenome], float],
        population_size: int = 20,
        generations: int = 5,
    ) -> HarnessGenome:
        """Optimize harness using evolutionary algorithm."""
        if self.algorithm == "ga":
            self.algorithm_instance = GeneticAlgorithm(
                population_size=population_size,
                generations=generations,
                mutation_rate=0.1,
                elitism_rate=0.2,
            )
            return self.algorithm_instance.evolve(fitness_fn)

        elif self.algorithm == "de":
            self.algorithm_instance = DifferentialEvolution(
                population_size=population_size,
                generations=generations,
                f=0.8,
                cr=0.9,
            )
            return self.algorithm_instance.evolve(fitness_fn)

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def get_fitness_history(self) -> List[float]:
        """Get fitness history from optimizer."""
        if self.algorithm_instance:
            return self.algorithm_instance.fitness_history
        return []
