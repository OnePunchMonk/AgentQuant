# How AgentQuant Evolves Its Search Process

Most projects described as “AI trading agents” ask whether a language model can
choose profitable parameters. That framing misses the more interesting systems
question: can an agent improve the process it uses to search?

AgentQuant uses trading as a concrete laboratory for that question. The agent
does not receive permission to trade, and its backtest results are not claims
of future returns. Instead, it runs a bounded research loop in which every
proposal becomes measurable evidence.

## The loop

```text
ANALYZE → HYPOTHESIZE → BACKTEST → REFLECT → STORE
              ↑                         │
              └──── retry when weak ────┘
```

In `ANALYZE`, the system computes features and labels the current market
context. The label combines relative volatility and trend, such as
`LowVol-Bull` or `HighVol-Bear`.

In `HYPOTHESIZE`, the agent proposes legal parameter configurations. An LLM can
rank configurations when credentials are available, but it is not required.
The fallback path uses the same canonical parameter grid and then random grid
sampling. This makes the LLM an optional search policy rather than a required
runtime dependency.

In `BACKTEST`, every candidate passes through one evaluator. The evaluator
enforces warmup history, applies commission, slippage, and market impact, and
computes Sharpe, Calmar, Sortino, drawdown, returns, and bootstrap diagnostics.

In `REFLECT`, the agent compares the outcome with a quality threshold. A weak
result can trigger another bounded iteration. Crucially, the agent now stores
negative evidence rather than only storing winners:

```text
regime, strategy, parameters, failure mode,
metric gap, counterfactual hypothesis
```

The next proposal prompt receives the most relevant failures as “do not repeat”
constraints. A failed experiment therefore changes the future search space.

Finally, `STORE` persists the result in SQLite so a later run can retrieve
evidence from a similar regime.

## A zero-configuration run

The fastest way to see the loop is:

```bash
python run_app.py
```

This creates a deterministic synthetic OHLCV fixture, runs the momentum agent,
and writes `results/demo_run.json`. On one recorded run, the output included:

```text
Iterations: 1 | Best Sharpe: 0.996
```

The same report recorded a holdout Sharpe of `-0.714`. That contrast is useful:
the agent found a strong in-sample candidate, but the held-out period did not
support the result. The demo is therefore evidence that the loop and gates are
working, not evidence that the strategy is profitable.

The command also demonstrates graceful degradation. With no API keys, the log
reports that the LLM planner is unavailable and uses grid search. With optional
credentials, Claude-compatible proposal generation and Tavily research can be
enabled without changing the evaluator.

## What “self-improving” means here

Self-improvement is deliberately narrower than autonomous model training. The
system improves several parts of its search process:

- proposal priorities are informed by prior alpha memory;
- rejected configurations become structured failure constraints;
- regime context changes which parameter regions are preferred;
- harness experiments compare tool, prompt, grid, ensemble, and research
  configurations;
- trace diagnostics reveal which proposal methods and iterations actually help.

The distinction matters. A higher score from an opaque model call is not enough
to establish improvement. AgentQuant requires a baseline, a measurable metric,
and an evaluation protocol that can expose overfitting.

## The harness evolution experiment

Earlier experiments evolved the agent’s configuration through multiple epochs:

1. grid-search baseline;
2. tool-aware proposals;
3. prompt refinement;
4. parameter-grid adaptation;
5. multi-agent or ensemble behavior;
6. research-agent context.

The repository reports these as development experiments. Some harness optimizer
benchmarks use a mock fitness function, so those scores must not be described as
historical trading performance. The reproducible scripts and result files are
kept precisely so a reader can inspect the protocol rather than accepting a
headline number.

Run the deterministic local comparison with:

```bash
python scripts/reproducible_benchmark.py
```

The output compares one bounded iteration with a three-iteration loop over
several fixed synthetic fixtures. It reports the raw case results and means in
`results/reproducible_benchmark.json`. This is a development benchmark for
search behavior, not a financial performance claim.

The current checked-in run is deliberately not presented as a success story:
the one-iteration mean Sharpe is `0.243` versus `0.193` for three iterations.
That result says these fixtures do not establish that more iterations help. It
is a useful falsification result and a signal to improve the iteration policy
and benchmark protocol before making a positive self-improvement claim.

## Why the evaluator comes first

An agent can easily optimize a noisy split. If the split changes, the data
leaks, or transaction costs are omitted, apparent self-improvement may be an
artifact. That is why the project invests in:

- warmup enforcement;
- chronological holdouts;
- anchored walk-forward evaluation;
- realistic transaction costs;
- bootstrap and worst-window diagnostics;
- counterfactual stress tests.

The stress tests perturb a candidate with a regime change, volatility spike,
removal of the best return days, and trend reversal. A strategy that collapses
under every plausible perturbation is not robust merely because its headline
Sharpe is high.

## What this project is and is not

AgentQuant is a research harness for studying self-improving search loops. It is
not a promise of profitable trading, a live execution system, or proof that an
LLM can forecast markets. The most defensible result is the engineering one:
reasoning is made useful by placing it inside a bounded, observable,
failure-aware evaluation loop.

That pattern generalizes beyond trading. Replace the strategy parameter grid
with another scientific or engineering search space, keep the evaluator and
feedback contract explicit, and the same architecture can study agents that
improve how they search for solutions.
