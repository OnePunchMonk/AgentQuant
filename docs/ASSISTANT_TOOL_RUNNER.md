# Assistant tool runner

`AssistantToolRunner` provides an optional assistant-facing path to registered sidecars. It does not change the default research loop.

- `discover()` lists adapters in the injected `SidecarToolRegistry`.
- `run()` selects tools through a fixed intent mapping, applies call limits, checks `decision_time`, and records results and errors.
- Each run gets its own counters, deadline, byte budget, and lock. Direct `registry.call()` users share that registry's budget. Configure session limits with `ToolBudget` and the per-run call limit with `max_tool_calls`.
- In-process deadlines detect overruns after a handler returns. Only opt-in process isolation forcibly stops an overrun. Register adapters at startup; shared handlers must be read-only or thread-safe.
- Requests receive only their supplied capabilities (default: `prior_results`). Holdout and promotion grants are rejected. Results are evidence for review, not strategy validation.
- Memory and HyDE calls require market cutoffs. `knowledge_as_of` can precede, but cannot follow, the decision time.

## Offline demo

```bash
python scripts/demo_assistant_tools.py
```

The demo uses a dated failure lookup without a network or model. Applications can add memory and HyDE adapters with `register_memory_search()` and `register_hyde_retrieval()`. Selection uses `AssistantToolRunner.DEFAULT_INTENTS` or an injected mapping; there is no learned selector, tool creation, or write path.
