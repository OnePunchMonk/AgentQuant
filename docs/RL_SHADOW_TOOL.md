# RL shadow advisory tool

`src.agent.rl_shadow_tool.advise` is a bounded assistant-facing adapter for
the existing non-learning policy contracts in `src.agent.idea_sidecar`. The
name describes the intended later integration; this module does not train or
load a learned policy.

## Interface

```python
from src.agent import rl_shadow_tool

advisory = rl_shadow_tool.advise(observation, candidates)
payload = advisory.to_dict()
```

The caller supplies an existing typed `Observation`, a sequence of typed
`Candidate` values, and optionally a `Policy`. The default is the fixed
`HeuristicPolicy`. At most 64 candidates are considered by default; callers
can lower or raise the cap up to 256. The returned payload contains policy
name/version/hash, the considered candidate count and source allocation, and
the selected action including its propensity. The shared sidecar action
validator rejects an action that is outside the bounded candidate menu.

This is advisory only. It does not create observations, access market or
holdout data, evaluate a candidate, calculate reward, train a policy, or
promote a strategy. Callers are responsible for supplying observations that
respect the sidecar's dev-only visibility contract. An empty menu and invalid
cap fail with `ValueError`; invalid policy actions raise
`idea_sidecar.ActionOutOfSpace`.
