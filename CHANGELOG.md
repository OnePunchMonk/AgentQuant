# Changelog

All notable changes to AgentQuant will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Tool Registry & Orchestration** — Composable tool system for agentic decision-making
  - `src/agent/tools/registry.py` — Tool schema, registration, and execution
  - `src/agent/tools/orchestrator.py` — Claude tool-use loop orchestration
  - `src/agent/tools/evals.py` — Evaluation suite for harness quality assessment
  - Web search integration via Tavily API (sentiment, strategy research)
  - `run_benchmark_to_assess_quality` tool for agent-driven eval
- Example integration showing how to refactor hypothesize_node to use tools
- Dependencies: `anthropic>=0.30`, `tavily-python>=0.3`

### Changed
- `pyproject.toml` — Added tool-calling and web search dependencies

## [0.2.0] — 2026-04-15

### Added
- **Harness Engineering Foundation** — Components for self-improving agent architecture
  - `src/agent/strategy_memory.py` — Cross-session learning via SQLite
  - `src/agent/memory_layer.py` — Pattern extraction and historical context retrieval
  - `src/research/alpha_store.py` — Alpha candidate persistence and recall
  - Look-ahead bias guards in backtest engine
  - Regime-aware parameter grid sampling
- **Scientific Baselines** — Experimental harness for rigorous evaluation
  - `experiments/walk_forward_context.py` — Walk-forward backtesting with market regime context
  - `experiments/walk_forward_context_with_costs.py` — Realistic trading costs
  - `experiments/ablation_study.py` — Parameter sensitivity analysis
  - `experiments/rigorous_baselines.py` — Budget-matched baseline comparisons
- **Trace & Observability** — Execution tracing for agent debugging
  - `src/agent/trace.py` — TraceRecorder and hierarchical trace storage

### Changed
- **Agent Graph Refactor** — Bounded ReAct loop with 5 explicit nodes
  - `analyze` → `hypothesize` → `backtest` → `reflect` → `store`
  - Max iterations with reflect-retry loop on Sharpe threshold
- **Proposal Generator Consolidation** — Single entrypoint replacing 4 planner files
  - Fallback chain: LLM → Grid Search → Random
  - Proposals constrained to canonical ParameterGrid for valid A/B comparison
  - Regime-aware prior sampling
- **Regime Detection** — VIX percentile-based instead of absolute levels
  - `src/features/regime.py` — Statistical regime classification
  - Cross-regime performance tracking for harness learning

### Fixed
- Look-ahead bias in backtest metric computation
- Parameter grid constraints validation before backtest execution

## [0.1.0] — 2026-01-01

### Added
- **Core Quantitative Research Platform**
  - Multi-strategy support (momentum, mean reversion, volatility, trend following, breakout)
  - OHLCV data ingestion from yfinance
  - Feature engineering (technical indicators, regime signals)
  - Strategy backtesting with realistic costs (slippage, commission, market impact)
  - Metrics computation (Sharpe ratio, returns, drawdown, win rate)
- **LLM Integration**
  - Multi-planner support (Gemini, LangChain, OpenAI, fallback to grid search)
  - Temperature-controlled proposal generation
  - Configurable retry logic
- **Market Data & Features**
  - VIX, Treasury yields (FRED API integration)
  - Multi-asset universe support
  - Caching layer for data (24h TTL)
- **Streamlit Dashboard**
  - Live data sidebar with price, VIX, yields
  - Agent execution lab
  - NLA memory browser
  - Research workspace
- **Testing & CI/CD**
  - 42 unit tests covering core systems
  - GitHub Actions CI gate
  - Config validation, backtest rigor, memory consistency checks

---

## Project Milestones

### v2.0 Refactor: From Parameter-Tuning Script to Self-Improving Agent (Q2 2026)
- Introduced bounded ReAct loop with formal state machine
- Cross-session memory persistence (StrategyMemory, AlphaStore)
- Regime-aware sampling and statistical regime detection
- Walk-forward validation harness
- Foundation for harness evolution and tool-based orchestration

### v1.0 Foundation: Core Research Platform (Q1 2026)
- Multi-strategy backtesting engine
- LLM-guided parameter selection
- Dashboard and monitoring
- Comprehensive test coverage

---

## Roadmap

### Q3 2026 — Harness Evolution & Self-Improvement
- [ ] Tool-calling orchestrator (in progress)
- [ ] Weakness mining from execution traces
- [ ] Falsifiable claims tracking and scoring
- [ ] Grid evolution (learn parameter space from results)
- [ ] Prompt template variation and ranking
- [ ] Held-out test window evaluation protocol
- [ ] Generalization gap measurement

### Q4 2026+ — Agentic Research Loops
- [ ] Multi-agent swarm with specialist roles (critic, regime analyst, etc.)
- [ ] Iterative hypothesis refinement
- [ ] Strategy discovery from web research
- [ ] Production deployment harness

---

## Breaking Changes

None yet. All changes maintain backward compatibility within v0.x → v1.x → v2.x progression.

---

## Contributing

See DESIGN.md for architecture overview and implementation guidelines.

---

## References

- DESIGN.md — Technical design document (v2.0)
- EXPERIMENTAL_DETAILS.md — Original experimental methodology
- docs/PAPER_DRAFT.md — Research paper draft
