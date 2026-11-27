# AgentQuant

[![CI](https://github.com/OnePunchMonk/AgentQuant/actions/workflows/ci.yml/badge.svg)](https://github.com/OnePunchMonk/AgentQuant/actions/workflows/ci.yml)

**Give research agents a memory of what failed—and a way to test what works.**

AgentQuant is a research harness for agents that propose experiments, test them, and use recorded evidence to choose what to try next. It keeps failed attempts as well as successful ones, so a research run can be inspected and compared with other search policies.

Quantitative strategies are the first testbed. The research question is whether an agent gets better at choosing experiments, or just runs more of them. **Validated autonomous self-improvement remains an open question.**

## What you can do

- **Run bounded research loops.** Generate candidate strategies, backtest them, inspect failures, and evaluate the selected candidate on a later holdout.
- **Keep point-in-time evidence.** Store trials and notes with provenance, query memory at a market-time cutoff, and derive beliefs from visible evidence.
- **Compare search policies.** Use frozen, random, grid, memory, and mutation baselines with chronological episode splits and candidate logs.
- **Inspect and replay decisions.** Export run manifests, traces, episode bundles, and research memos. Advisory tools can retrieve evidence and suggest candidates without owning the evaluator.
- **Study promotion safeguards.** The gate package provides a trial ledger and search-aware comparison tools; their use depends on the workflow. The demo's holdout result is diagnostic and does not automatically veto a candidate.

## Quick start

Requires Python 3.10+. The demo runs offline on synthetic prices; no API key is needed.

```bash
git clone https://github.com/OnePunchMonk/AgentQuant.git
cd AgentQuant
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m src.demo
```

The demo runs up to three search iterations and writes `results/demo_run.json`. Inspect a [captured sample report and its raw run](https://github.com/OnePunchMonk/agentquant-artifacts/tree/main/onboarding) to see attempted candidates and a negative holdout result. The sample is synthetic; it is not a trading-performance benchmark.

```bash
agentquant --help             # run, memory, dream, regime-card
python -m pip install -e ".[dev]"  # development and test tools
```

Optional extras include `llm` for model providers and web research, plus `data`, `backtest`, and `regime`; see [pyproject.toml](pyproject.toml). For provider-backed features, copy `.env.example` to `.env` and set the relevant keys. Run state defaults to `.agentquant/`; set `AGENTQUANT_RUNS_DIR` to choose another location.

## How it fits together

![Research harness architecture](https://raw.githubusercontent.com/OnePunchMonk/agentquant-artifacts/main/docs/diagrams/research_harness.png)

The loop analyzes context, proposes and evaluates candidates, reflects on the results, and stores evidence for later decisions. The shared backtest code lags positions by a bar and applies transaction costs. Memory has its own temporal visibility rules; policy promotion and final-holdout safeguards are separate components and must be used by the calling workflow.

| Path | What it contains |
| --- | --- |
| [`src/agent/`](src/agent/) | Research loop, proposal generation, search baselines, sidecars, traces, and policy mutation |
| [`src/memory/`](src/memory/) | Trials, notes, point-in-time retrieval, beliefs, evidence graph, and compaction |
| [`src/gate/`](src/gate/) | Trial ledger, promotion comparisons, and prospective-evaluation utilities |
| [`src/backtest/`](src/backtest/) and [`src/strategies/`](src/strategies/) | Shared return and cost calculations, metrics, and strategy signals |
| [`src/features/`](src/features/), [`src/data/`](src/data/), and [`src/research/`](src/research/) | Regime features, data ingestion, and hypothesis evidence |

[Research loop walkthrough](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/SELF_IMPROVING_SEARCH.md) · [Architecture](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/AGENTQUANT_HARNESS_ARCHITECTURE.md) · [Memory design](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/MEMORY_LAYER_DESIGN.md)

## Evidence and limitations

The bounded loop, memory service, replay artifacts, and advisory sidecars are implemented. A matched live-model comparison has **not** established that evidence-conditioned mutation improves search over random controls; see [issue #31](https://github.com/OnePunchMonk/AgentQuant/issues/31). Offline fixture ties test wiring rather than the effect of an LLM intervention.

One measured engineering improvement is graph-memory retrieval: on 400 synthetic trials, median retrieval on an existing graph fell from **206.39 ms to 2.53 ms**. This local CPU benchmark excludes database access, model calls, and full-agent execution; it is not an end-to-end speedup. See the [method, parity checks, and raw results](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/optimizations/sidecar_graph_retrieval/README.md).

For experiment reports, diagrams, benchmark scripts, and the Streamlit dashboard, see [agentquant-artifacts](https://github.com/OnePunchMonk/agentquant-artifacts). Active research directions are tracked in [#51](https://github.com/OnePunchMonk/AgentQuant/issues/51), [#52](https://github.com/OnePunchMonk/AgentQuant/issues/52), [#53](https://github.com/OnePunchMonk/AgentQuant/issues/53), and [#54](https://github.com/OnePunchMonk/AgentQuant/issues/54).

[Contributing](CONTRIBUTING.md) · [Issues](https://github.com/OnePunchMonk/AgentQuant/issues) · [MIT license](LICENSE)
