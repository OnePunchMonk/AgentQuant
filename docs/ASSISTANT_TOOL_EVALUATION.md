# Assistant tool routing evaluation

`scripts/benchmark_assistant_tools.py` compares fixed intent routing with the
repository's actual `AssistantToolRunner` and `DeterministicToolSelector`, plus
synthetic ablations. Each arm registers fixture-backed `AssistantTool`
handlers and executes the real runner. It is an offline mechanics benchmark,
not a model evaluation: no LLM, network, market data, backtest, or holdout
result is used. Quality means only that the route selected the fixture's
labeled relevant evidence source.

## Frozen suite

The six synthetic tasks, expected evidence source, routing intent, tool latency
and cost weights live in `results/assistant_tool_fixture.json`. The manifest
contains a SHA-256 digest over the ordered task records. The benchmark refuses
to run if the digest or suite version changes. To propose a new suite, create a
new versioned fixture and report it separately; do not silently rewrite the
held-out manifest.

## Compared variants

| Variant | Routing behavior |
| --- | --- |
| `fixed_intent` | Uses the task's frozen fixed-intent mapping. |
| `selector` | Uses the runner's deterministic relevance ranking over registered fixture tool names, descriptions, and request fields. It returns all relevant matches up to the runner's call limit; it does not use a learned scorer. |
| `no_tools` | Abstains without calling a tool. |
| `memory_only`, `hyde_only`, `rl_shadow_only` | Calls just the named sidecar for every task. |
| `all_tools` | Calls all three sidecars on every task. |

The RL shadow label represents a synthetic candidate-advice endpoint. It does
not invoke a policy or claim the policy improves candidate quality. Similarly,
HyDE and memory are represented as evidence-source labels; this benchmark does
not assess their underlying retrieval algorithms.

## Metrics and interpretation

The result reports task-level quality and abstention, total and per-task calls,
synthetic latency and cost units, and replay parity. For each arm the benchmark
runs the real runner twice per task and compares serialized `to_dict()` results,
excluding only `elapsed_seconds` because wall-clock timing varies between
replays. Latency and cost are fixture weights (memory 7 ms / 1 unit, HyDE
13 ms / 2 units, RL shadow 5 ms / 1 unit), not measured wall-clock latency or
monetary cost. Replay parity covers selections, tool results, traces, provenance,
and authority fields after removing elapsed timings.

The fixed-intent route is intentionally the oracle baseline for this fixture,
so its perfect score is a fixture sanity check rather than evidence of
generalization. The selector can return multiple relevant tools for ambiguous
wording, illustrating the call-quality tradeoff. The fixture is
small and synthetic; no model quality claim, finance-domain claim, or production
recommendation follows from its scores.

Run from the repository root:

```bash
python scripts/benchmark_assistant_tools.py --output results/assistant_tool_benchmark.json
pytest -q tests/test_benchmark_assistant_tools.py
```

This benchmark is separate from the live sidecar runner and registry. It does
not alter tool permissions, call production adapters, or train a selector.
