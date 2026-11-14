# Assistant tool contract

This note describes the typed assistant-facing boundary in `src/agent/sidecar_tools.py` and the bounded runner in `src/agent/assistant_tool_runner.py`. Tools are bounded data services: they return evidence or candidate options for review. They do not evaluate strategies, choose winners, promote beliefs, or access holdout data.

## Wire contract

Each `AssistantTool` declares a name, request schema, result schema, required capabilities, handler, and provenance source. A `ToolRequest` carries a request ID, tool name, arguments, and the caller's capability grants. A `ToolResult` carries the matching ID, success or isolated error, data, and provenance. The registry applies per-instance call, wall-time, per-call and cumulative JSON-byte limits; validates declared top-level object/array shapes and field types; and converts handler failures into tool results. Its budget starts when the registry is constructed and is shared across every request using that instance. Validation is intentionally a shallow schema subset, not full recursive JSON Schema validation. Strict JSON encoding rejects unsupported and non-finite values.

Capabilities are explicit. Existing adapters require `prior_results`. Registration rejects tools that require `holdout_data`, `promote_candidate`, or `set_holdout_grade`. These checks are boundary controls, not a sandbox for arbitrary handlers; injected handlers should remain read-only and narrowly scoped.

The current idea-sidecar adapters return dated failure records and canonical-grid candidate options. The HyDE adapter returns unevaluated candidates with evidence references. The memory adapter returns time-scoped graph evidence and states that results are not promoted beliefs. Provenance should preserve source IDs and both market-time and knowledge-time cutoffs wherever the source provides them.

## PACT-inspired action-state envelope

PACT proposes projecting agent output into a compact, action-centered public state update instead of retaining unconstrained messages in shared history. It reports improved performance-cost tradeoffs in its evaluated multi-agent settings; that result motivates testing compact tool feedback here, but does not establish a finance-domain benefit. See [PACT, “What Should Agents Say?”](https://arxiv.org/abs/2606.05304).

For this project, keep the canonical message typed and inspectable. A compact envelope can contain:

```json
{
  "schema_version": "1.0",
  "request_id": "r-123",
  "action": "retrieve_hypothesis_evidence",
  "state": "ok",
  "summary": {"candidate_count": 2},
  "evidence_refs": ["trial-17", "note-08"],
  "cutoffs": {"market_as_of": "2024-01-31", "knowledge_as_of": "2024-02-02"},
  "budget": {"calls_remaining": 2},
  "provenance": {"tool": "hyde_retrieval", "source": "memory_snapshot-5"}
}
```

The envelope is a summary for shared agent state, not a replacement for the full typed result or its evidence. Keep evidence references resolvable, preserve exact cutoffs, and include a clear error state. Schema versioning and round-trip tests should make the envelope safe to replay and inspect.

## Tool selection and tool creation are separate

LOTS studies output-aware, task-specific tool-space optimization using accumulated tool-output experience and likelihood-based contribution estimates. Its reported gains are a research result on its evaluated benchmarks, not evidence that an online selector will improve this project. See [LOTS, “Self-Evolving Agents via Likelihood-Guided Tool-Space Optimization”](https://arxiv.org/abs/2609.34151).

Treat a LOTS-inspired selector as a future policy candidate. ExpG is another candidate for experience-derived guidance over tool selection, invocation, and response quality. Both are inspiration for offline evaluation; neither learned method is implemented here. `DeterministicToolSelector` now provides capability-filtered lexical ranking, configured intent mappings, abstention and an optional injected scorer with recorded fallback. The scorer cannot register tools or grant capabilities. The synthetic routing benchmark does not establish the scorer or real tool quality. Keep the default registry fixed and explicit until a reproducible offline benchmark shows a benefit. A useful comparison should:

1. Freeze tasks, data cutoffs, tool outputs, and candidate tool sets before scoring.
2. Compare the fixed registry with the selector using identical budgets and repeated seeds.
3. Report task quality, call count, latency, input/output tokens, and failure rates separately.
4. Check temporal leakage, tool attribution, stability across task groups, and ablations against simple selection baselines.
5. Promote a selector only through normal review after the benchmark protocol and results are recorded; do not learn its tool set from live or holdout outcomes.

Tool creation is a different problem from selecting among existing tools. TOOLMAKER studies generating LLM-compatible tools from papers with code and checks generated behavior with tests. This repository currently registers injected, existing callables; it does not generate or install tools. A future tool-creation experiment would need a separate isolated workflow, source attribution, dependency controls, and generated-tool correctness/security tests before a tool could enter this registry. See [ExpG](https://arxiv.org/abs/2608.03403) and [TOOLMAKER](https://aclanthology.org/2025.acl-long.1266/).

## Compact or learned communication

An optional research lane may test compact symbolic messages with a versioned dictionary, explicit provenance escape hatches, and readable fallbacks. Require schema round-trip tests, unknown-symbol handling, a human-readable decoder, and downstream task benchmarks against the canonical typed format and plain text. The goal is measurable utility under a fixed budget, not compression alone.

Emergent-language studies show that agents can develop compositional conventions that are difficult for people to understand; that creates a monitoring and handoff cost even if messages become shorter. A study of English-forced communication measures translation losses in multilingual agent tasks; it is evidence about cross-lingual routing, not evidence that a private language helps this project. Latent-channel work reports representational compression but no task-level advantage over text in its evaluated tasks. These findings argue for keeping any private or latent channel optional and auditable, and out of production defaults until task-level gains, provenance preservation, and reliable decoding are demonstrated. See [GlossoGen](https://arxiv.org/abs/2609.01491), [Latent Communication Between Language Model Agents](https://arxiv.org/abs/2607.14103), and [Translating the Translator](https://arxiv.org/abs/2609.15079).


## Runtime authority and effective budgets

One assistant schedules tools. A tool returns evidence or options; the common evaluator and promotion gate retain their separate authority. `AssistantToolRunRequest.max_tool_calls` narrows the registry ceiling. Calls admitted to handlers consume budget even on error; rejected requests remain visible in runner attempts. Use a fresh registry for an independent budget scope. Do not reset/recreate it mid-run to evade exhausted limits. Current accounting covers JSON bytes and admitted calls, not model tokens, dollars, peak memory or CPU.

Default handlers run in process: a wall-time error is detected after return and cannot stop a hanging handler. `ToolBudget.isolate_process=True` uses a POSIX fork worker with deadline termination; unsupported environments fail explicitly. A worker is a cancellation boundary, not a security sandbox: it inherits process access to local files, network and credentials. Only trusted, read-only injected handlers belong here. Full security isolation requires external process/container policy and capability-scoped resources. Keep calls sequential unless aggregate accounting and concurrency have been verified.

The runner owns decision-time stamping for memory/HyDE and idea tools; market `as_of` must equal `decision_time` and a supplied knowledge cutoff cannot exceed it. Missing timestamps must never be filled from wall-clock time. A new custom tool must declare its temporal policy explicitly and receive scoped data; a capability name alone cannot filter data inside an arbitrary handler. Custom registration is an explicit allowlisted call, not automatic code discovery or installation.

The opt-in `AGENTQUANT_ASSISTANT_TOOLS` path currently retrieves graph-memory evidence during proposal construction and records tool payloads in agent state/trace. This provides a real runtime adapter. It does not establish complete conversion of legacy proposal/literature/RL paths or a unified run-wide budget. Disabled behavior must retain parity; enabled failures and skipped requests must be observable without changing evaluation authority.

## Replay and release acceptance

A complete assistant run artifact must bind schema/code versions, effective registry and schemas, selector/scorer version and configuration, seed, exact market/knowledge cutoffs, effective budget scope, ordered admission/attempt events, canonical request hashes, full typed results, evidence snapshot/source hashes, errors and terminal status. Preserve raw typed results beside compact summaries. Record fallbacks, bytes/calls spent and model/provider usage when applicable; never infer missing usage as zero.

Replay validates artifact hashes and consumes the recorded ordered results through the same result-validation and state-update boundary without calling handlers, scorers, models or network. It must reproduce selections, attempts, terminal status, evidence references and deterministic downstream decisions; volatile timing is reported separately. Missing/reordered events, incompatible schemas, altered tool descriptions, unresolved pinned sources and budget inconsistencies fail explicitly. Rerunning a deterministic fixture selector is a useful mechanics check, but is not this artifact replay contract. Existing episode/memo replay remains a separate shipped feature.

Release acceptance requires adversarial fixtures for forbidden/missing grants, invalid/non-JSON payloads, cumulative exhaustion, slow/failed handlers, unavailable process mode, deterministic ties and scorer fallback, timestamp boundaries, no-tool abstention and runtime feature-off parity. The routing benchmark must distinguish synthetic label quality and fixture cost weights from measured downstream task success, actual latency/tokens/dollars and real adapter failure rates. See [ASSISTANT_TOOL_EVALUATION.md](ASSISTANT_TOOL_EVALUATION.md); its present synthetic benchmark is not a model or finance evaluation.
