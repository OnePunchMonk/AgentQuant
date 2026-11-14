# Tool communication experiment

Run the deterministic synthetic benchmark with:

```bash
python scripts/benchmark_tool_decision_protocol.py --output results/tool_communication_benchmark.json
```

It compares JSON one-shot, compact one-shot, JSON with bounded refinement, and compact with bounded refinement on the same frozen evaluation cases. The frozen candidate set contains two distinct tools with different request/result schemas. Cases deliberately include wrong-tool proposals and one request that cannot be repaired. Verified requests are sent through a real `SidecarToolRegistry` call; unverified refined requests are rejected without execution. Requests, tool responses, and the subsequent continue/defer decision are serialized and decoded; integrity checks cover identity, provenance, and exact fixture output.

The fixture keeps generic repair patterns in a training split and five cases in a separate evaluation split. Refinement uses the fixture's expected tool and result schema as a deterministic verifier oracle, then applies fixed rules for trimming strings and replacing an invalid fact limit with 2. This oracle measures bounded repair behavior; it is not a simulated model prediction. Evaluation cases do not alter those rules or the compact dictionary. The benchmark is a protocol and control-flow exercise, not evidence about model quality, user intent recognition, or production cost.

Reported metrics include tool-selection validity, argument validity, registry execution success, result and decision integrity, exact result match, serialized request/response/decision bytes, repair steps, registry calls, elapsed local latency, and provider calls. Token counts are null unless the optional `tiktoken` package is installed; when present, the benchmark uses its `cl100k_base` tokenizer. Provider calls are zero. Local compute cost is unpriced. Latency is descriptive and machine-dependent; correctness and byte counts are deterministic. The registry's timing-only provenance field is normalized to zero in the serialized fixture result so it cannot change byte counts between runs.

The compact representation is versioned and integrity checked. Any fitted value dictionary must be fit using training data only and frozen before evaluation; unseen values must remain representable without changing the dictionary. Keep the fixture and report synthetic, and do not interpret these measurements as a quality or deployment claim.

The fixture is deliberately small; compact envelopes may be larger than JSON because versioning and integrity metadata can outweigh savings from short field codes. The expected-result rate is capped below 1.0 by the unrepairable case, which is rejected before a registry call. The benchmark report records measured values and does not price local computation.
