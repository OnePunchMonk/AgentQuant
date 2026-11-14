# Development failure benchmark sidecar

`src.agent.benchmark_sidecar` turns metadata from development runs into a
reviewable proposal for the next immutable benchmark manifest. It complements
`benchmark_suite.py`: the base suite remains the versioned anchor and the
sidecar only appends proposed challenge episodes.

## Boundary and lifecycle

Inputs are `DevFailure` records. Each record must identify a `dev` source run,
its source episode, content/data/split hashes, a capability label, a difficulty
rating from 1 to 5, and provenance. The sidecar consumes these fields only; it
does not load datasets, fetch episode rows, or inspect sealed outcomes. Sealed
anchor entries are carried forward unchanged. New episodes are assigned to
`selection`, never to `sealed`.

The proposal builder applies a maximum-additions budget, minimum difficulty,
per-capability quota, deterministic ordering, exact-content deduplication, and
data/split conflict checks against the anchor and other proposed cases. Each
challenge carries its source failure IDs, provenance hash, and content hash.
Rejected records have a reason attached. The next manifest is hash-chained to
the anchor using the existing `evolve_suite` function.

The returned `SidecarProposal` starts in `pending_review`. `review()` records an
independent review decision but does not install or promote the manifest. A
separate release process must decide whether a reviewed proposal becomes the
new trusted anchor. This keeps mined examples from silently changing the
benchmark used for comparisons.

```python
from src.agent.benchmark_sidecar import DevFailure, propose_next_suite

proposal = propose_next_suite(anchor_suite, dev_failure_records,
                              max_additions=8, min_difficulty=2)
assert proposal.status == "pending_review"
assert proposal.proposed_suite.parent_hash == anchor_suite.content_hash
```

The sidecar detects hash conflicts available in the manifest and incoming
failure metadata. It cannot establish semantic independence when upstream
systems supply misleading or reused hashes; provenance generation and human
review remain part of the gate.

## Deterministic synthetic illustration

`synthetic_comparison(seed=17)` returns a fixed toy comparison between a small
fixed suite and a larger proposed suite. It is for checking the comparison
plumbing and deterministic output only. The toy score is not an agent result,
does not use market or user data, and supports no empirical performance claim.
