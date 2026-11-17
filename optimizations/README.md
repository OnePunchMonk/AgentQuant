# Optimizations

Measured speed improvements to AgentQuant, with the bottleneck, code change, before-and-after diagram, and a reproducible benchmark.

| Optimization | Measured change | Scope |
|---|---|---|
| [Graph-memory retrieval](sidecar_graph_retrieval/README.md) | 206.39 → 2.53 ms retrieval; 223.41 → 22.46 ms including graph rebuild | 400 synthetic trials; local CPU wall time; median of 7 runs |

Each result describes its named workload. These timings exclude database and model latency and do not imply an equivalent speedup for the full agent.
