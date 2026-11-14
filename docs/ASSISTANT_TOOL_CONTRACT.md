# Assistant tool contract

This note describes the typed assistant-facing boundary in `src/agent/sidecar_tools.py` and a conservative direction for future tool selection. Tools are bounded data services: they return evidence or candidate options for review. They do not evaluate strategies, choose winners, promote beliefs, or access holdout data.

## Wire contract

Each `AssistantTool` declares a name, request schema, result schema, required capabilities, handler, and provenance source. A `ToolRequest` carries a request ID, tool name, arguments, and the caller's capability grants. A `ToolResult` carries the matching ID, success or isolated error, data, and provenance. The registry applies per-run call, wall-time, and payload-size limits; validates the declared object/array shapes; and converts handler failures into tool results.

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

Treat a LOTS-inspired selector as a future policy candidate. ExpG is another candidate for experience-derived guidance over tool selection, invocation, and response quality. Both are inspiration for offline evaluation; neither is implemented here. Keep the default registry fixed and explicit until a reproducible offline benchmark shows a benefit. A useful comparison should:

1. Freeze tasks, data cutoffs, tool outputs, and candidate tool sets before scoring.
2. Compare the fixed registry with the selector using identical budgets and repeated seeds.
3. Report task quality, call count, latency, input/output tokens, and failure rates separately.
4. Check temporal leakage, tool attribution, stability across task groups, and ablations against simple selection baselines.
5. Promote a selector only through normal review after the benchmark protocol and results are recorded; do not learn its tool set from live or holdout outcomes.

Tool creation is a different problem from selecting among existing tools. TOOLMAKER studies generating LLM-compatible tools from papers with code and checks generated behavior with tests. This repository currently registers injected, existing callables; it does not generate or install tools. A future tool-creation experiment would need a separate isolated workflow, source attribution, dependency controls, and generated-tool correctness/security tests before a tool could enter this registry. See [ExpG](https://arxiv.org/abs/2608.03403) and [TOOLMAKER](https://aclanthology.org/2025.acl-long.1266/).

## Compact or learned communication

An optional research lane may test compact symbolic messages with a versioned dictionary, explicit provenance escape hatches, and readable fallbacks. Require schema round-trip tests, unknown-symbol handling, a human-readable decoder, and downstream task benchmarks against the canonical typed format and plain text. The goal is measurable utility under a fixed budget, not compression alone.

Emergent-language studies show that agents can develop compositional conventions that are difficult for people to understand; that creates a monitoring and handoff cost even if messages become shorter. A study of English-forced communication measures translation losses in multilingual agent tasks; it is evidence about cross-lingual routing, not evidence that a private language helps this project. Latent-channel work reports representational compression but no task-level advantage over text in its evaluated tasks. These findings argue for keeping any private or latent channel optional and auditable, and out of production defaults until task-level gains, provenance preservation, and reliable decoding are demonstrated. See [GlossoGen](https://arxiv.org/abs/2609.01491), [Latent Communication Between Language Model Agents](https://arxiv.org/abs/2607.14103), and [Translating the Translator](https://arxiv.org/abs/2609.15079).
