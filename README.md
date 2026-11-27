# AgentQuant

A research agent that remembers what failed and tests what works.

It runs a bounded loop: propose an experiment, evaluate it, record the evidence, pick the next one. Every trial, including the failures, goes into a point-in-time memory, so you can compare search policies at the same budget and replay their decisions. Quant trading is the first domain; the question is whether the agent gets better at choosing experiments or just runs more of them.

```
analyze -> hypothesize -> backtest -> reflect -> holdout -> store
               ^                                          |
               +------------- memory <--------------------+
```

## Quick start

Python 3.10+. The demo is offline and needs no API keys.

```bash
git clone https://github.com/OnePunchMonk/AgentQuant.git && cd AgentQuant
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python -m src.demo            # three search iterations on synthetic prices, writes results/demo_run.json
agentquant --help             # run, memory, dream, regime-card
```

Extras: `.[llm]` (model providers, web research), `.[data]`, `.[backtest]`, `.[regime]`, `.[dev]` (tests and linting). For provider-backed runs copy `.env.example` to `.env` and set your keys; never commit it.

Run state (results database, run manifests, ledgers, holdout guard) goes to `.agentquant/`. Set `AGENTQUANT_RUNS_DIR` to move it.

## What is in the package

| Path | What it does |
|---|---|
| `src/agent/` | The loop, proposal generation and validation, search arms, sidecars (hypothesis evidence, RL shadow policy, tool runner), harness config and policy mutation |
| `src/memory/` | Point-in-time memory: trials, notes, evidence graph, beliefs deflated by the number of configs tried, compaction |
| `src/gate/` | Promotion gate: hash-chained trial ledger, max-T bootstrap, deflated-Sharpe hurdle |
| `src/backtest/`, `strategies/`, `features/`, `data/` | The quant domain: one shared evaluator and cost model, strategies, regimes, data ingest |
| `src/research/` | Alpha, hypothesis and NLA memory stores |

Design rules the code enforces: one evaluator for every arm; the final holdout can be used once; a verdict from after a trial's window is never visible to a query made before it closes.

## Status

Implemented: the bounded search loop, memory service, gate, replay artifacts and advisory sidecars. Not yet shown: that the self-improvement loop beats a random-search control with live models. That is the open research question (#31).

Planned work is tracked as charters: [#51](https://github.com/OnePunchMonk/AgentQuant/issues/51) Reflector post-training, [#52](https://github.com/OnePunchMonk/AgentQuant/issues/52) domain-agnostic package, [#53](https://github.com/OnePunchMonk/AgentQuant/issues/53) prompt optimisation, [#54](https://github.com/OnePunchMonk/AgentQuant/issues/54) harness and tool-design research.

## More

Design docs, diagrams, benchmarks, experiments, notebooks and the Streamlit dashboard live in [agentquant-artifacts](https://github.com/OnePunchMonk/agentquant-artifacts). Start with [the research loop walkthrough](https://github.com/OnePunchMonk/agentquant-artifacts/blob/main/docs/SELF_IMPROVING_SEARCH.md).

[Contributing](CONTRIBUTING.md) · [Issues](https://github.com/OnePunchMonk/AgentQuant/issues) · [MIT license](LICENSE)
