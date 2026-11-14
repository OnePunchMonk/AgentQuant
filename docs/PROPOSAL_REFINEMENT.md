# Proposal refinement experiment

`src.agent.proposal_refinement.refine_proposals` is a small, generic search
experiment. The “diffusion-inspired” name refers only to its coarse-to-fine
shrinking proposal scale. It is not a diffusion model or sampler. Evaluation
and candidate generation are injected by the caller, so the module has no
trading, backtest, or model dependencies.

Each round requests a speculative batch from the current best candidate and
scale, then evaluates all candidates before selecting a new incumbent. The
scale decays by `scale_decay` after each round. The hard evaluation budget
includes the initial candidate. Latest-batch population standard deviation is
used as a simple uncertainty proxy. Early stopping requires both uncertainty
at or below the configured threshold and the configured number of rounds
without improvement. Scores are maximized; non-finite scores and malformed
batches fail fast.

Run the deterministic synthetic comparison with:

```bash
python scripts/benchmark_proposal_refinement.py
```

The benchmark uses a one-dimensional multimodal toy objective and 100 seeds.
For every seed, a fixed-step run is limited to exactly the number of evaluator
calls used by the refinement run; both runs use the same seeded proposal
sequence. This deliberately isolates the stopping policy at equal cost. Since
the baseline receives the same proposal prefix, quality will be equal; the
cost metric reports how much of the nominal 33-call budget the scheduler
consumes. The unit tests separately cover budget enforcement and early-stop
conditions.

This fixture validates implementation mechanics only. It offers no evidence
for actual diffusion behavior, improved strategy search, or trading benefit.
