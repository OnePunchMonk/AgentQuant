# Agent communication format experiment

This offline deterministic microbenchmark compares multiple wire representations
of the same four synthetic assistant-to-sidecar action/state trajectories:

- **English/full result:** labeled prose with every field present.
- **PACT-inspired envelope:** compact JSON with explicit field names and a
  version number. This local schema does not claim conformance to an external
  PACT specification.
- **Versioned symbolic dictionary:** compact JSON using the public v1 table in
  `src/agent/communication.py`. The table is hand-authored and inspectable.
- **Delta stream:** stateful deltas with periodic full checkpoints; gaps and
  reorderings fail closed until another checkpoint.
- **Content-addressed references:** an envelope containing evidence IDs and
  source references, with evidence values resolved from an external store.
- **Compressed binary:** a checksummed, versioned compressed envelope. It must
  be decoded before content is presented to a language model.
- **No communication:** baseline that transmits no payload. It cannot recover
  an artifact, so its artifact and decision fidelity are zero by construction.

## What is measured

The runner reports wire bytes, optional LLM-facing tokenizer tokens, exact
artifact recovery, deterministic decision fidelity, repeated serialization and
replay latency, and delta-stream behavior under one dropped frame and one
reordering. Artifact recovery and decision fidelity are distinct: a decoded
record may preserve bytes/fields while an independently chosen task decision
can still differ. The fixture's rule is deterministic and is not a model.

Token counts are **not estimated or silently substituted**. To count tokens,
install `tiktoken` and pass exactly one explicit encoding or model:

```bash
pip install tiktoken
python -m scripts.benchmark_agent_communication --encoding cl100k_base
# or
python -m scripts.benchmark_agent_communication --model gpt-4o
```

The count is over the LLM-facing representation after transport decoding (the
current receiver prompt view is the full English rendering). For a binary
format, bytes are transport size; they are not model tokens. Without an explicit
choice the token count is null and the output explains that it was not measured.

Latency covers local Python encoding/decoding and deterministic replay only. It
excludes a network, model inference, and sidecar computation. Results vary with
machine and runtime. The reference method's frame size excludes evidence-store
transfer/storage; it is meaningful only when receiver and sender share the
addressed evidence. The no-communication row has no decode latency.

## Interpretation and limits

The fixture is small and synthetic. Results describe representation overhead
and deterministic recovery for these records only. They do not measure
end-to-end assistant task success, usefulness, or model behavior; that outcome
is reported as unmeasured because no model/task execution occurs. Compression
and compactness can preserve sufficient fields while still changing downstream
actions under receiver-relative coordination constraints; see [Bounded
Coordination](https://arxiv.org/abs/2610.03769). This benchmark does not claim
that symbolic communication emerges naturally, that any format improves agent
capability, or that a learned private language exists. Post-training convention
formation is a possible future learned lane, not simulated here; see [Emergent
Communication in LLMs](https://arxiv.org/abs/2508.06482).

Run from the repository root:

```bash
python -m scripts.benchmark_agent_communication
python -m pytest tests/test_communication.py
```

Bytes, exact recovery, decision fidelity, and recovery outcomes are deterministic.
Token totals require the requested tokenizer. Timings are informational.
