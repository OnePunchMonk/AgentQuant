# Diagrams

Every diagram is an editable `.excalidraw` file (open it at excalidraw.com) with a rendered `.png` next to it.

| Diagram | What it shows |
|---|---|
| [hld](hld.png) | Whole system: entry points, data, agent loop, tools, harness, memory, backtest, run artifacts |
| [lld-agent-loop](lld-agent-loop.png) | `run_agent()` call sequence, run_status outcomes, data passed between nodes |
| [proposal-paths](proposal-paths.png) | Tool loop vs ProposalGenerator cascade, and the search/holdout split |
| [memory-layer](memory-layer.png) | Memory write path, SQLite tables, recall with `as_of`, and the dream sidecar |
| [belief-scoring](belief-scoring.png) | How trials become beliefs: weighting, deflation, shrinkage, verdict rules, the prompt the LLM reads |
| [self-improving-search](self-improving-search.png) | Bounded prompt-mutation outer loop, episode splits, promotion gate, controls |
| [fair-search-benchmark](fair-search-benchmark.png) | The 8 benchmark arms, shared protocol, recorded offline results |
| [evolving-harness](evolving-harness.png) | The three ways the harness evolves and the config gate they all pass through |
| [harness-evolution](harness-evolution.png) | Library presets v1..v6, the gate, GA search, reported vs reproduced numbers |
| [harness-v1-base](harness-v1-base.png) ... [v6-research](harness-v6-research.png) | One per version: requested spec, gate result, what run_agent reads, live pipeline parts, prompt prefix |

Per version: [v1](harness-v1-base.png) · [v2](harness-v2-tool-aware.png) · [v3](harness-v3-prompt-tuned.png) · [v4](harness-v4-grid-evolved.png) · [v5](harness-v5-multi-agent.png) · [v6](harness-v6-research.png)
