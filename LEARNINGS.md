# AgentQuant Learnings

## A human guide to what this repository teaches

This document is a learning companion to AgentQuant. It explains the design
decisions, experiments, reversals, and engineering lessons visible across the
project history so that a human can learn from the work directly rather than
only reading the latest architecture diagram.

The current `main` branch contains 78 commits as of 2026-09-12. The commonly
referenced “77 commits” count predates the final changelog documentation commit.

## The central lesson

An agentic quantitative system is not made reliable by adding an LLM to a
backtest. It becomes useful when the agent is placed inside a bounded research
protocol:

```text
market data
    ↓
features and regime context
    ↓
bounded proposal generation
    ↓
cost-aware, leakage-guarded evaluation
    ↓
reflection against explicit quality gates
    ↓
stored evidence, including failures
    ↓
better future proposals
```

The LLM is one proposal source inside this protocol. It is not the protocol,
the evaluator, or the evidence.

## 1. How the project evolved

### Phase 1 — From idea to working research platform

The earliest commits established the project identity, documentation, data
ingestion, strategies, visualizations, and a basic agent concept. The first
important engineering insight was that a trading research project needs a
working end-to-end path before it needs sophisticated autonomy.

Key commits include:

- `793d5a1` — initial project and major-upgrades baseline
- `f73c09a` — Gemini integration, data-ingestion fixes, ablation and walk-forward experiments
- `05740fb` — context-aware experiment results and research paper material
- `9d16c9a` — rigorous baselines and cost-aware experiments

### What to learn

- Start with a small number of executable strategies and a reproducible data
  path.
- Make the evaluation engine useful without an LLM. Otherwise it is impossible
  to tell whether the agent improved the research or merely changed the story.
- Treat transaction costs, warmup periods, and data quality as first-class
  modeling decisions, not implementation details.

## 2. Phase 2 — Turning a planner into an agent loop

The architectural refactor made the research process explicit:

```text
ANALYZE → HYPOTHESIZE → BACKTEST → REFLECT → STORE
              ↑                         │
              └──── bounded retry ─────┘
```

The key implementation sequence was:

- `1038653` — provider-agnostic LLM planning and structured context
- `1bab321` — grid-constrained proposal generation with fallback behavior
- `fe33122` — backtesting, metrics, signal generation, and warmup enforcement
- `02155df` — SQLite experiment tracking, tests, and CI
- `137ce75` — comprehensive v2 architecture overhaul
- `45f1a72` — parallel backtesting and CI/CD support

### What to learn

The most important agent design decision was not “which model?” It was creating
typed boundaries between reasoning stages:

- analysis produces context;
- hypothesis generation produces validated proposals;
- backtesting produces metrics;
- reflection makes a bounded decision;
- storage makes evidence available to the next run.

This makes the system inspectable, testable, and replaceable. Each stage can be
improved without turning the whole application into one opaque prompt.

## 3. Phase 3 — Making proposals comparable

The proposal generator evolved from multiple planners into a single entrypoint
with a controlled fallback chain:

1. LLM proposal selection
2. regime-aware grid search
3. random sampling from the same valid grid

The LLM selects from a canonical parameter space rather than inventing
arbitrary parameter values. This is a subtle but important scientific control:
LLM, grid, and random proposals can be compared under the same search space.

### What to learn

Free-form generation is attractive but makes evaluation ambiguous. Constraining
the action space gives up some apparent creativity in exchange for:

- valid proposals;
- comparable baselines;
- simpler failure analysis;
- fewer accidental out-of-range configurations;
- reproducible fallback behavior.

The general pattern applies beyond finance: define the legal action space first,
then let the model choose, rank, or explain actions inside it.

## 4. Phase 4 — Regimes and memory

The project added regime detection based on relative volatility and momentum,
then began persisting historical results:

- VIX percentile rather than fixed absolute thresholds
- trend and momentum labels
- `StrategyMemory` for run history
- `AlphaStore` for alpha candidates and validation evidence
- NLA and research workspace memory for human-visible context

### What to learn

Memory is only useful when it is conditioned on the situation in which the
memory was created. “This parameter worked” is weak knowledge. “This parameter
worked during HighVol-Bear conditions with these costs and this evaluation
window” is much more useful.

The project also exposes an important limitation: storing successes alone is
not enough. Without structured negative evidence, an agent can repeatedly
rediscover the same bad idea.

## 5. Phase 5 — Product surface and observability

The history added a research workspace, live ticker selection, dashboards,
screenshots, swarm visibility, and memory browsing:

- `330cdbc` — research workspace memory platform
- `66db79d` — live ticker selection
- `48e7f73` — visible agent memory and swarm runs
- `4576eca` — platform screenshots and documentation

### What to learn

Human visibility is part of agent quality. If a researcher cannot see:

- what the agent believed;
- which proposals it tried;
- why a proposal was rejected;
- which memory influenced the next attempt;
- where the system stopped;

then the system is difficult to debug and difficult to trust, even if its
backtests run successfully.

## 6. Phase 6 — Safety, leakage, and reversibility

The project extracted a data-leakage auditor into `Peek`, merged it, and then
reverted that extraction:

- `607fc46` — introduce Peek leakage auditor
- `9edbaa6` — merge the extraction
- `875669a` — revert Peek
- `8275a8e` — merge the revert
- `73ccd23` — fix backtest rigor and safety gaps found in review

### What to learn

The lesson is not that every extraction is bad. The lesson is that repository
boundaries, maintenance cost, and product scope must be evaluated alongside
technical quality.

It is often better to preserve a strong safety check in the main research
workflow than to split it into a new package before its ownership, API, and
release boundary are clear.

The later backtest-rigor work reinforces that safety belongs at the engine
boundary. A strategy should not be able to silently bypass warmup rules or
evaluation constraints merely because it was generated by a different planner.

## 7. Phase 7 — Tool-using agent orchestration

The next evolution made the agent capable of calling tools rather than only
generating text:

- `62344e8` — tool registry and orchestration
- `383deec` — integrate orchestration into hypothesis and reflection nodes
- `ac68642` — proposal parsing and deep-research-agent design

The tool layer provides schemas, execution, error handling, and graceful
fallbacks. It can retrieve regime context, search strategy research, search
market sentiment, inspect parameter recommendations, and run quality checks.

### What to learn

Tool use becomes engineering rather than prompting when tools have:

- explicit schemas;
- narrow responsibilities;
- deterministic local fallbacks;
- logged calls and results;
- tests independent of the external provider.

External tools should add capability, not become a single point of failure.

## 8. Phase 8 — Harness evolution

The project then turned the agent itself into an object of experimentation:

- `007e690` — harness evolution proof of concept
- `398145a` — multi-iteration evolution with genetic and differential-evolution methods
- `4228274` — complete multi-iteration harness evolution
- `33c48ab` — CI verification for harness experiments
- `4f8a2a0` — held-out-window grading

The major conceptual shift was from “find a good strategy” to “measure whether
changes to the research harness improve outcomes under a fixed evaluation
protocol.”

### What to learn

Any self-improvement claim needs:

1. a baseline;
2. a fixed or explicitly budgeted evaluation protocol;
3. a held-out quality gate;
4. cost and iteration accounting;
5. a falsification condition;
6. records of unsuccessful changes.

Without those controls, an evolving harness can optimize the benchmark rather
than improve the research process.

## 9. Phase 9 — Correcting the evidence narrative

The documentation went through several visual and narrative revisions. The
most important evidence correction was:

- `44a3bb7` — clarify experimental evolution claims
- `319752d` — merge the corrected branch into main

The project now distinguishes development benchmarks using mock fitness from
actual historical or live trading results. It also distinguishes recorded
free-form claims from calibrated Sharpe-prediction accuracy.

### What to learn

Scientific credibility is part of the implementation. A result should say what
was measured, where it was measured, and what it does not establish.

Good language is specific:

- “development benchmark using mock fitness” is not “trading performance”;
- “claim recorded for later analysis” is not “forecast accuracy validated”;
- “walk-forward utility exists” is not “the whole production agent uses
  walk-forward validation.”

This correction is one of the highest-value lessons in the repository because
it prevents a polished dashboard from making a stronger claim than the code
supports.

## 10. Current learnings added after the original evolution work

The newest implementation extends the loop in several directions:

### Structured failure memory

Failures are stored with:

```text
regime
strategy_type
params
failure_mode
metric_gap
counterfactual_hypothesis
```

The most relevant failures are injected into future proposal prompts as
constraints. This turns failure from disposable control flow into reusable
research knowledge.

### Walk-forward evaluation

`src/backtest/walk_forward.py` evaluates expanding-history / rolling-test
windows and reports median and worst-window Sharpe. This is more informative
than relying on one split or one aggregate score.

### Counterfactual stress testing

`src/backtest/stress_test.py` tests regime changes, volatility spikes, removal
of the best return days, and trend reversals. The goal is not to predict every
future market path; it is to expose strategies that are only successful under a
narrow historical path.

### Trace diagnostics

`TraceRecorder.diagnostics()` reports event counts, proposal methods,
improvement rounds, acceptances, and validator rejections. This creates the
beginning of a reusable harness-diagnostics format.

### Regime transitions

`RegimeChangeDetector` records transitions between labels. A future extension
can use those events to invalidate cached proposals, increase the iteration
budget, and retrieve what worked during similar transitions.

### Strategy code generation boundary

`src/strategies/codegen.py` validates generated source using AST checks before
registration. This is an extension point, not a claim that arbitrary generated
code is safe. Full sandboxing, resource limits, and promotion gates remain
necessary before treating code generation as production-ready.

## 11. What the repository teaches about agentic feature prioritization

The most valuable features are not necessarily the most visually impressive.
The implementation history suggests this order:

1. **Make the evaluation correct.** Guard leakage, warmup, costs, and splits.
2. **Make actions bounded.** Use schemas, registries, validators, and budgets.
3. **Make evidence persistent.** Store successes and structured failures.
4. **Make the loop observable.** Record traces, decisions, and timing.
5. **Make improvements testable.** Use baselines, holdouts, and falsification.
6. **Only then expand autonomy.** Add research tools, multi-agent behavior, and
   generated strategies behind the same gates.

This ordering reduces the risk of optimizing an impressive but untrustworthy
system.

## 12. A practical reading path for a human

Read the repository in this order:

1. `README.md` — product overview and current claims.
2. `DESIGN.md` — architectural boundaries and agent loop.
3. `src/agent/agent_graph.py` — the executable control flow.
4. `src/agent/proposal_generator.py` — bounded hypothesis generation.
5. `src/backtest/runner.py` and `src/backtest/metrics.py` — what is actually measured.
6. `src/features/regime.py` — how context is formed.
7. `src/research/alpha_store.py` and `src/agent/strategy_memory.py` — what is remembered.
8. `src/agent/tools/registry.py` — how external capabilities are bounded.
9. `src/agent/trace.py` — how behavior becomes inspectable.
10. `tests/` — the executable contract and edge cases.
11. `experiments/` and `scripts/` — how harness changes are compared.
12. `git log --reverse --oneline` — the project’s actual evolution rather than
    only its final presentation.

## 13. Questions a future contributor should ask

Before adding an agentic feature, ask:

- What decision does this feature improve?
- What is the deterministic baseline?
- What evidence would falsify the claimed benefit?
- Does it change the data, proposal space, evaluator, memory, or presentation?
- Can its cost and failure rate be measured?
- Does it introduce a new source of leakage or benchmark overfitting?
- What happens when the external model or API is unavailable?
- Is the result stored in a form that the next iteration can reuse?
- Can a human inspect why the agent made the decision?

If these questions cannot be answered, the feature is probably an interesting
prototype but not yet a reliable improvement to the research agent.

## Closing perspective

AgentQuant’s most reusable contribution is not a particular strategy or
Sharpe number. It is the pattern of putting flexible reasoning inside a strict,
observable, evidence-producing loop. The history shows both the productive
additions and the corrective reversals needed to keep that loop honest.

That is the main lesson for a human reader: build the evaluator, memory, and
feedback system with as much care as the agent itself.
