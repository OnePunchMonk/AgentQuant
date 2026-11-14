# Conservative RRSI-inspired proposal preflight

This module takes inspiration from the RRSI paper's bounded/annealed search,
novelty, critic, cost awareness, and structural pruning ideas. It is a small
local adaptation, **not an exact reproduction** and makes no claim that the
paper's results transfer to AgentQuant. See [RRSI](https://arxiv.org/abs/2609.24972).

`src/agent/evolution_regularizer.py` consumes only an edit descriptor:
component, changed paths, content metadata, estimated cost, and benchmark-token
markers. `preflight_edit` is deterministic and returns an auditable decision.
The edit budget anneals linearly from four paths to one by default. History
fingerprints discourage exact repeats. Configured benchmark tokens trigger a
critic-style leakage abstention. Cost and supplied measurement noise raise the
minimum evidence floor. Paths in data, experiments, secrets, `.env`, and `.git`
are pruned. These are conservative defaults, not a security boundary.

`select_with_paired_evidence` is a helper for a later evidence review. It
abstains without enough paired observations, on non-finite values, any paired
regression, or a mean gain at/below the cost/noise floor. A positive result
means only “worth human review”; it does not edit files, update a harness, or
promote a candidate. Neither helper is wired to automatic code editing.

## Evaluation plan before considering integration

1. Freeze candidate descriptors, incumbent, train/validation/OOD splits,
   token/cost accounting, and the stopping rule before outcomes are inspected.
2. Compare baseline proposal review against preflight on paired seeds and
   identical candidate budgets. Report held-out utility, leakage incidents,
   rejected/accepted counts, false rejection review, runtime, and tokens.
3. Use OOD episodes separated by asset and time regime, with a final sealed
   period unavailable to proposal authors. Include deliberately planted
   benchmark-specific tokens as a positive control for the leakage critic.
4. Ablate one factor at a time: no annealing, no history novelty, no leakage
   tokens, no cost/noise floor, and no structural pruning. Also compare the
   full conservative configuration with an always-abstain/null baseline.
5. Require paired evidence on held-out episodes and report uncertainty and
   regression frequency. Do not tune thresholds against the sealed OOD set.

The descriptor preflight is O(paths + history + tokens) time and O(history)
space per call because fingerprints are recomputed. Paired selection is O(n)
time and O(1) extra space. The module makes no model calls, so its direct token
cost is zero; candidate evaluation cost is supplied by the caller and remains
the dominant experimental cost. Measure wall time and model tokens in the
benchmark harness rather than inferring them from these asymptotics.
