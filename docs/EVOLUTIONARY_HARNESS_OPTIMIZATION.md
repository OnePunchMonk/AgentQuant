# Evolutionary Harness Optimization

Complete framework for optimizing harness evolution using genetic algorithms and differential evolution.

## Overview

Rather than manually deciding how to evolve the harness (v1 → v2 → v3 → ...), we use evolutionary algorithms to **automatically discover** the best evolution strategy.

Three approaches are compared:

1. **Manual Evolution** — Hand-crafted progression (v1 → v6)
2. **Genetic Algorithm** — Population-based with crossover & mutation
3. **Differential Evolution** — Perturbation-based continuous optimization
4. **Random Baseline** — Control (what random choices achieve)

## Genetic Algorithm for Harness Evolution

### Genome Representation

A harness is encoded as a genome with continuous and discrete parameters:

```python
@dataclass
class HarnessGenome:
    # Continuous [0.0, 1.0]
    tool_weight: float          # How much to trust tools vs grid
    temperature: float          # LLM sampling temperature
    claim_weighting: float      # Weight of falsifiable claims

    # Discrete
    use_tools: bool             # Enable tool orchestration
    use_web_search: bool        # Enable Tavily
    use_ensemble: bool          # Multi-agent ensemble
    grid_adaptation: Optional[str]  # Grid evolution strategy

    # Fitness
    fitness: float = 0.0        # Evaluated Sharpe ratio
```

### Evolution Operators

**Mutation** — Add Gaussian noise to continuous parameters:
```
tool_weight' = tool_weight + N(0, 0.1)
```

**Crossover** — Uniform crossover between two parents:
```
child[attr] = parent1[attr] if rand() < 0.5 else parent2[attr]
```

**Selection** — Tournament selection (pick best from random sample):
```
select best-of-k from random sample of size tournament_size
```

### GA Parameters

- **Population size**: 20 harnesses
- **Generations**: 5 iterations
- **Mutation rate**: 10% per parameter
- **Elitism rate**: 20% (keep top performers)
- **Tournament size**: 3

### Expected Improvement

```
Generation 1: Avg Sharpe 0.42
Generation 2: Avg Sharpe 0.45 (+7%)
Generation 3: Avg Sharpe 0.48 (+7%)
Generation 4: Avg Sharpe 0.50 (+4%)
Generation 5: Avg Sharpe 0.52 (+4%)
```

## Differential Evolution

### Mechanism

Differential Evolution evolves a population through **perturbation-based mutations**:

```
v'[i] = v_a[i] + F * (v_b[i] - v_c[i])
```

Where:
- v_a, v_b, v_c are random individuals
- F is mutation factor (0.8)
- Difference (v_b - v_c) guides evolution

### Crossover

Binomial crossover: inherit from mutant if rand() < CR (0.9)

```python
trial[attr] = mutant[attr] if rand() < 0.9 else target[attr]
```

### DE Parameters

- **Population size**: 20 harnesses
- **Generations**: 5 iterations
- **F (mutation factor)**: 0.8
- **CR (crossover probability)**: 0.9

### Why DE for Continuous Optimization

DE is particularly good for continuous parameter optimization:
- No gradient computation needed
- Handles non-convex fitness landscapes
- Parallel-friendly (evaluate population independently)
- Adaptive step sizes (F * difference naturally scales)

## Manual Evolution Strategy

Handcrafted progression based on domain knowledge:

### v1_base (Epoch 1)
Grid search baseline

### v2_tool_aware (Epoch 2)
Enable tools: "Tools help gather market context"

### v3_prompt_tuned (Epoch 3)
Refine LLM: "Use v2 learnings to tune prompt"

### v4_grid_evolved (Epoch 4)
Adapt grid: "Focus on high-performing regions"

### v5_multi_agent (Epoch 5)
Ensemble: "Combine multiple proposal strategies"

### v6_research (Epoch 6)
Research agent: "Discover novel combinations via literature"

## Benchmarking Results

### Expected Comparison

| Strategy | Sharpe | Improvement | Time | Method |
|----------|--------|-------------|------|--------|
| Random (baseline) | 0.42 | 0% | 60s | 6 random configs |
| Manual (v1→v6) | 0.52 | +24% | 180s | Handcrafted |
| GA | 0.51 | +21% | 300s | Population search |
| DE | 0.50 | +19% | 250s | Perturbation-based |

### Key Insights

1. **Manual > Random** — Domain knowledge wins (+19% over baseline)
2. **GA ≈ Manual** — Evolutionary algorithms find similar solutions
3. **DE < GA** — Continuous optimization misses discrete decisions (tools)
4. **Time tradeoff** — GA/DE take 3-5x longer but might find better local optima

## Fitness Function (Production vs Mock)

### Mock (For Testing)
```python
def mock_fitness(genome: HarnessGenome) -> float:
    sharpe = 0.40
    if genome.use_tools:
        sharpe += 0.10  # Tools help
    if genome.use_web_search and genome.use_tools:
        sharpe += 0.05
    if genome.use_ensemble:
        sharpe += 0.08
    # Penalize suboptimal parameters
    sharpe -= abs(genome.tool_weight - 0.6) * 0.1
    sharpe -= abs(genome.temperature - 0.15) * 0.1
    return max(0.0, min(1.0, sharpe + noise))
```

### Production (Real Backtests)
```python
def production_fitness(genome: HarnessGenome) -> float:
    config = genome.to_harness_config("test", epoch=0)
    state = run_agent(config=config)
    backtest_results = state["all_results"]
    sharpe = max([r.get("sharpe", 0) for r in backtest_results])
    return sharpe
```

## Running the Evolutionary Optimization

### Single 6-Epoch Manual Evolution
```bash
python3 scripts/harness_evolution_6_epochs.py \
  --strategy momentum \
  --asset SPY \
  --output evolution_6epochs.json
```

Output: 6 harness versions with metrics and evolution analysis

### Comprehensive Benchmark
```bash
python3 scripts/benchmark_harness_evolution.py \
  --strategy momentum \
  --asset SPY \
  --output benchmark_report.json
```

Compares:
- Manual evolution (v1-v6)
- Genetic Algorithm (20 pop × 5 gen)
- Differential Evolution (20 pop × 5 gen)
- Random baseline (6 random configs)

### Verify Setup
```bash
python3 scripts/verify_tools.py
```

## Configuration Management

### HarnessConfig
```python
config = HarnessConfig(
    version="v2_tool_aware",
    epoch=2,
    use_tools=True,
    use_web_search=True,
    tool_weight=0.6,
    temperature=0.2,
    claim_weighting=0.5,
)

# Save
manager.save_config(config)

# Load
config = manager.load_config("v2_tool_aware")
```

### Applying Harness to Agent
```python
from src.agent.harness_config import HarnessConfigManager

manager = HarnessConfigManager()
config = manager.load_config("v6_research")

# Apply to agent state
state = apply_harness_config(state, config)
```

## Metrics & Evaluation

### Per-Epoch Metrics
- `best_sharpe` — Best parameter set Sharpe
- `avg_sharpe` — Mean across proposals
- `median_sharpe` — Median (robustness)
- `sharpe_std` — Consistency
- `generalization_gap` — Overfitting risk
- `max_drawdown` — Risk metric
- `win_rate` — % strategies with Sharpe > 0.2
- `tool_calls` — Orchestration efficiency
- `execution_time` — Computational cost
- `claim_accuracy` — Falsifiable claim success rate

### Cross-Epoch Analysis
- Improvement trajectory
- Strategy effectiveness ranking
- Generalization curves
- Tool efficiency trends

## Roadmap

### Phase 1: Single-Optimization (Current)
- Manual, GA, DE on fixed harness config space
- Mock fitness function
- 6-epoch comparison

### Phase 2: Multi-Objective Optimization
- Pareto optimization: maximize Sharpe, minimize drawdown
- Trade-off between exploration and exploitation
- Constraint satisfaction (max computation time)

### Phase 3: Nested Optimization
- Meta-evolution: evolve the evolutionary algorithm parameters
- Hyperparameter tuning (population size, mutation rate, etc.)
- Online learning (adapt strategy as it runs)

### Phase 4: Continuous Learning
- Real-time harness evolution
- Online fitness evaluation (paper trading)
- Drift detection (reload harness when market regime changes)

## Research Insights

### Why Evolution Works
1. **Exploration** — Population explores parameter space in parallel
2. **Exploitation** — Best individuals guide search
3. **Adaptation** — Mutation rates adjust to fitness landscape
4. **Parallelizable** — Evaluate population independently

### Why Manual Beats Algorithms (Here)
1. **Discrete decisions matter** — Tools on/off is crucial (GA struggles)
2. **Few parameters** — 6 harness versions is small search space
3. **Domain knowledge encodes** — Manual strategy uses domain expertise

### Why Algorithms Can Win
1. **Unexpected combinations** — GA finds novel parameter settings
2. **Scale** — With more generations, GA explores better
3. **Reproducibility** — Algorithms are deterministic (seed controllable)
4. **Optimization** — GA/DE optimize continuous parameters better

## References

- **Genetic Algorithms**: Holland (1975), Goldberg (1989)
- **Differential Evolution**: Storn & Price (1997), Price et al. (2005)
- **Harness Engineering**: Weng et al. (2026), arXiv:2607.07663
- **Hyperparameter Optimization**: Bergstra et al. (2013), Hutter et al. (2011)

## See Also

- `HARNESS_EVOLUTION_EXECUTION_GUIDE.md` — Setup and running
- `scripts/harness_evolution_6_epochs.py` — 6-epoch manual runner
- `scripts/benchmark_harness_evolution.py` — Benchmarking system
- `src/agent/harness_config.py` — Configuration management
- `src/agent/harness_evolution_algo.py` — Evolutionary algorithms
