# Assistant tool runner

`src/agent/assistant_tool_runner.py` adds an optional assistant-facing path over the registered sidecars. It does not alter or enter the default research agent loop.

`AssistantToolRunner.discover()` lists only tools registered in the injected `SidecarToolRegistry`. `run()` applies a fixed intent mapping, limits selected calls per request, stamps or verifies the request's `decision_time`, and records each `ToolResult` and error in the typed result. The underlying registry also enforces its per-instance call, wall-clock, and payload budgets. Configure the registry budget to set the hard process-side limit; `max_tool_calls` provides a smaller per-run limit.

The runner grants only the capabilities supplied by the request (default: `prior_results`) and rejects holdout/promotion grants. It has no evaluation, grading, or promotion API. Results are evidence and options for human or downstream review; they are not strategy validation. Memory and HyDE calls carry explicit market cutoffs. A separate `knowledge_as_of` may be earlier than the decision time, but cannot be later.

## Offline demo

From the repository root:

```bash
python scripts/demo_assistant_tools.py
```

The demo registers the existing idea-sidecar tools and runs a dated failure lookup without network or model access. Applications can inject memory and HyDE adapters with `register_memory_search()` and `register_hyde_retrieval()` before constructing the runner.

Intent selection is deterministic and fixed by `AssistantToolRunner.DEFAULT_INTENTS`, or by an injected mapping. There is no learned selector, tool creation, or write path.
