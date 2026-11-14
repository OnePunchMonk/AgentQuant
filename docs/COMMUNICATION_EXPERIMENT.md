# Agent communication format experiment

This offline, deterministic microbenchmark compares three encodings of the same
four synthetic assistant-to-sidecar action/state trajectories:

1. **English/full result:** labeled prose with every field present.
2. **PACT-inspired envelope:** compact JSON with explicit field names and a
   version number. This is a local experimental schema, not a claim of
   conformance to an external PACT specification.
3. **Versioned symbolic dictionary:** compact JSON using the public v1 symbol
   table in `src/agent/communication.py`. It is a hand-authored, inspectable
   dictionary, not an opaque or learned language.

Each trajectory carries action, state, requested decision, numeric value, source
references, a data cutoff, and an optional error. The same deterministic
downstream decision rule is applied after decoding. The experiment reports exact
round-trip fidelity, decision fidelity, UTF-8 bytes, a token proxy, and local
serialization/replay latency. The token estimate counts regex-matched words and
punctuation; it is **not a model tokenizer**. Latency excludes network, model
inference, and sidecar work and is sensitive to machine/runtime conditions.

Malformed JSON/prose, missing fields, unknown symbolic codes, and unsupported
dictionary versions are tested. Errors and provenance are required data and are
preserved through decoding. The fixture is intentionally small and synthetic;
results describe encoding overhead for these records only. They do not show
that symbolic communication emerges naturally or improves agent capability.

Run from the repository root:

```bash
python -m scripts.benchmark_agent_communication
python -m pytest tests/test_communication.py
```

The JSON output includes experiment controls and measurements for independent
reproduction. Timings vary between runs; bytes, proxy token counts, and fidelity
are deterministic.

In the four-message fixture, all formats round-trip exactly and preserve the
deterministic decision. Labeled prose is 531 bytes/172 proxy tokens, the JSON
envelope is 618 bytes/308 proxy tokens, and symbolic JSON is 474 bytes/308
proxy tokens. This proxy shows **no token saving** from either structured format
on this fixture. A model-token and downstream task-quality comparison is still
required before claiming a communication benefit.
