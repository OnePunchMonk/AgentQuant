# AgentQuant

> Docs, diagrams, notebooks, the Streamlit dashboard and experiment notes live in [agentquant-artifacts](https://github.com/OnePunchMonk/agentquant-artifacts). This repo is the production package.

**Give research agents a memory of what failed—and a way to test what works.**

AgentQuant runs a research loop: propose an experiment, test it, inspect the result, and use that evidence to choose the next one. It tracks the whole search, including failed attempts, so you can compare research approaches at the same budget and replay their decisions.

Quantitative strategies are the first testbed. The broader question is simple: **does the agent get better at choosing experiments, or does it just run more of them?**

## What you get

- **Bounded experiments.** Candidates share a backtest evaluator and cost model, with explicit search budgets.
- **Memory with provenance.** Trials, notes, and temporal graph evidence retain their sources and date cutoffs.
- **Comparisons you can inspect.** Compare search policies on the same episodes; export traces, episode bundles, and research memos.
- **Tools around one research loop.** Memory retrieval, hypothesis evidence, and candidate suggestions help the agent without owning evaluation or promotion.

Use it to study research-agent behavior, test memory and proposal policies, or build on an inspectable experiment loop.

## Install and try it

Requires Python 3.10+. The offline demo needs no API keys.

```bash
git clone https://github.com/OnePunchMonk/AgentQuant.git
cd AgentQuant
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python run_app.py
```

The demo runs three search iterations on synthetic prices and writes `results/demo_run.json`. It shows the loop and its trace; it is not a trading-performance benchmark.

Optional extras, from the repository root:

```bash
python -m pip install -e ".[llm]"  # model providers and web research
python -m pip install -e ".[dev]"  # tests and development tools
```

Extras can be combined (`".[llm,dev]"`); `data`, `backtest` and `regime` also exist, see [pyproject.toml](pyproject.toml). For provider-backed features copy `.env.example` to `.env` and set the keys for your provider; never commit `.env`. Market data and provider calls need network access, the offline demo does not.

Troubleshooting: for a missing module, activate the environment and install the extra above; for a failed data download, check the network and ticker symbols and try a shorter date range. The Streamlit dashboard is in [agentquant-artifacts/app](https://github.com/OnePunchMonk/agentquant-artifacts/tree/main/app).

## Follow an experiment

![Research harness architecture](https://raw.githubusercontent.com/OnePunchMonk/agentquant-artifacts/main/docs/diagrams/research_harness.png)

[Walk through the research loop](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/SELF_IMPROVING_SEARCH.md) · [Architecture](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/AGENTQUANT_HARNESS_ARCHITECTURE.md) · [Memory](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/MEMORY_LAYER_DESIGN.md)

Run a comparison locally:

```bash
python scripts/fair_search_benchmark.py --episodes 3 --seeds 7 11 19 \
  --output results/fair_search_benchmark.json
```

This defaults to synthetic fixtures. Offline prompt-mutation arms do not call an LLM, so matching scores are not evidence of learned improvement. See [experimental details](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/EXPERIMENTAL_DETAILS.md) for interpretation.

## Where it stands

The bounded search loop, memory service, replay artifacts, and advisory tools are implemented. Validated autonomous self-improvement remains a research goal: live policy comparisons, search-aware Sharpe inference, full promotion-gate integration, and prospective evaluation are still open.

For measured runtime improvements, see [optimizations](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/optimizations/README.md), including the graph-memory retrieval [before-and-after diagram](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/optimizations/sidecar_graph_retrieval/README.md).

[Contribute](CONTRIBUTING.md) · [Roadmap](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/notes/ROADMAP.md) · [Open issues](https://github.com/OnePunchMonk/AgentQuant/issues) · [MIT license](LICENSE)
