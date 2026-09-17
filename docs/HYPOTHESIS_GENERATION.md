# Generative Hypothesis Generation (HyDE-Inspired) — Draft

Status: **prototype**, tracking issue #23. Scoped down from the full issue
because it depends on the Literature Agent (#22), which is not implemented
yet. This draft ships the generative + retrieval pieces on their own, with
literature findings as an optional text input the Literature Agent can fill
in once it lands.

## Problem

The existing search loop (`ProposalGenerator`, see `docs/RESEARCH_AGENT_DESIGN.md`)
only ranks parameter sets from a fixed grid. It cannot propose a strategy
combination that isn't already in the grid or in published literature. HyDE
(Gao et al. 2022) suggests a fix for the analogous problem in retrieval:
generate a hypothetical answer first, then search for evidence that supports
or contradicts it, instead of only searching for what's already written.

Applied here: generate synthetic "ideal strategy" hypotheses from first
principles, then retrieve historical evidence for each one instead of
starting from historical evidence.

## Pipeline

```text
RegimeContext ──▶ HypothesisGenerator ──▶ [Hypothesis, ...]
                                                │
                                                ▼
                                       HypothesisScorer
                              (retrieves from StrategyMemory + NLAMemoryStore)
                                                │
                                                ▼
                                     [ScoredHypothesis, ...]
                                                │
                                                ▼
                                      HypothesisMemory (SQLite)
                                     proposed → backtested → confirmed/rejected
```

### 1. `src/agent/hypothesis_generator.py`

`HypothesisGenerator.generate(context, n, literature_context="")` prompts an
LLM (via the existing `BasePlanner`/`create_planner` abstraction — Gemini,
OpenAI, or Claude, whichever credential is present) to produce `n` synthetic
strategy hypotheses for the given `RegimeContext`. Each hypothesis is a
one-sentence claim plus a proposed strategy type, parameters, predicted
Sharpe, and confidence.

`literature_context` is a plain string the Literature Agent (#22) will
eventually supply. It's optional by design: this module works standalone,
so it isn't blocked on #22 landing first.

When no LLM credential is available, a deterministic template fallback
produces combinatorial hypotheses (same idea as `ProposalGenerator`'s
grid/random fallback) so the module is fully exercisable offline and in CI
without API keys.

**Multi-provider note:** `create_planner()` was extended with a `ClaudePlanner`
(Anthropic SDK) alongside the existing Gemini and OpenAI planners, so this can
be exercised against any of the three providers by setting `GOOGLE_API_KEY`,
`OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` and `llm.provider` in `config.yaml`
(`gemini` | `openai` | `claude`). All three follow the same
`generate_proposals(prompt, n) -> List[Dict]` contract, so hypothesis parsing
is identical regardless of provider.

### 2. `src/agent/hypothesis_scorer.py`

`HypothesisScorer.score(hypothesis)` embeds the hypothesis text and retrieves
the most similar prior runs from `StrategyMemory` (backtest history) and
`NLAMemoryStore` (narrative memory), then combines:

- **evidence_density** — similarity-weighted count/strength of retrieved
  matches (more corroborating evidence at higher similarity scores higher)
- **backtest_quality** — similarity-weighted average Sharpe of the retrieved
  matches

into a `composite_score`.

The embedding is a deterministic hashed bag-of-words vector (256-dim,
L2-normalized cosine similarity) — not a learned semantic model. That's a
scope cut: it needs no external API call or extra model dependency, and it's
enough to answer "did we see something like this before," which is what
evidence density needs. Swapping in a real embedding API (OpenAI/Gemini
embeddings) is a one-function change (`_embed`) if the ranking quality turns
out to matter.

### 3. `src/research/hypothesis_memory.py`

SQLite archive for generated hypotheses, following the same pattern as
`StrategyMemory`/`NLAMemoryStore`. Tracks a status lifecycle
(`proposed → backtested → confirmed/rejected`) so a future backtest pass can
record whether a generated hypothesis panned out, and future generation
passes can avoid re-proposing rejected ones via `to_prompt_context()`.

## What's NOT in this draft

Per the issue's acceptance criteria, still open:

- [ ] Wiring into the agent graph / swarm runner (this PR adds the building
      blocks, not the orchestration step)
- [ ] Actually running the backtest step and populating `backtest_sharpe`
      end-to-end (the hook exists in `HypothesisMemory.update_status`, but
      nothing calls it yet)
- [ ] Literature Agent (#22) integration — `literature_context` is currently
      always empty in practice
- [ ] Prototype run + comparison against literature-only hypotheses on 2+
      regimes, and the false-positive-rate metric from the issue

## How to test with each provider

```bash
export GOOGLE_API_KEY=...      # llm.provider: gemini (default)
export OPENAI_API_KEY=...      # llm.provider: openai
export ANTHROPIC_API_KEY=...   # llm.provider: claude
pytest tests/test_hypothesis_generation.py -v
```

The unit tests use a fake planner and don't require any credential; the env
vars above are only needed to manually smoke-test a specific provider's
`generate_proposals` output end-to-end.
