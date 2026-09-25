# AgentQuant Memory Layer: Design

Status: proposal · Written against `9b7d20f`

## 1. What exists today

AgentQuant already has memory. It is spread across five stores that all write into `experiments/results.db` but have no shared contract:

| Module | Table | Written by | Read by | Ranks by |
|---|---|---|---|---|
| `agent/strategy_memory.py` | `strategy_runs` | `store_node`, swarm `MemoryAgent` | `analyze_node`, `AgenticMemoryLayer`, CLI | timestamp |
| `research/alpha_store.py` | `alpha_candidates` | `store_node` | `analyze_node`, `ProposalGenerator._memory_generate`, Streamlit | `alpha_score` (in-sample) |
| `research/alpha_store.py` | `failure_records` | `reflect_node` (every sub-threshold result, every iteration) | `ProposalGenerator` prompt | `ABS(metric_gap)` |
| `research/hypothesis_memory.py` | `hypotheses` | nothing in `src/` | nothing in `src/` | `composite_score` |
| `research/nla_memory.py` | `nla_records` | `store_node`, JSONL ingest | `analyze_node`, Streamlit | `quality_score` = Sharpe − DD |
| `agent/memory_layer.py` | (view over `strategy_runs`) | — | swarm, CLI | avg Sharpe per (regime, strategy) |

### Problems, ordered by how much they hurt results

1. **Memory rewards overfitting.** Every ranking uses the in-sample tournament Sharpe. `holdout_sharpe` is stored in `strategy_runs` but no retrieval path uses it. `_memory_generate` then feeds the top in-sample winners back in as proposals with `confidence ≥ 0.55`. The loop finds a lucky config, stores it, re-proposes it, it wins again on the same history, and it gets stored again.
2. **No point-in-time guarantee in the stores.** `timestamp` is wall-clock time. The market dates a result was computed on are never stored. `filter_visible_memory` only works on caller-built lists that carry an `episode_id`. The real `frozen_agent_memory` arm is safe only because callers run episodes in order. A live run on 2025 data followed by a backtest episode ending in 2021 would read future-derived memory without any warning.
3. **Retrieval requires an exact regime label.** Every query is `WHERE regime = ?` over about 12 labels built from hard thresholds (`vix_pct > 65`). A `HighVol-Bull` at the 66th percentile sees nothing from `MidVol-Bull` at the 64th. Meanwhile `RegimeContext` already computes the continuous features that could be used instead.
4. **No deduplication or trial counting.** Each store call mints a random 8-character id. The same `{fast:20, slow:100}` shows up as N rows, and `failure_records` grows by `proposals × iterations` per run. Nothing counts how many configs were tried in a scope, so no multiple-testing correction is possible.
5. **Two agents see different memory.** `agent_graph` concatenates three `to_prompt_context()` strings, while the swarm reads only `AgenticMemoryLayer` patterns. Nothing limits the size of the combined prompt.
6. **Provenance gaps.** `RunManifest.memory_snapshot_id` is always `None`. Winners are the only rows that reach `strategy_runs`/`alpha_candidates`. Losing trials exist only as failure text, so the memory cannot answer "how many times did we try momentum in LowVol-Bear?"

## 2. Goals and non-goals

**Goals**
- G1: One read API and one write API, used by `agent_graph`, the swarm, `ProposalGenerator`, the CLI, and Streamlit.
- G2: No future leakage, enforced in the query layer. Every read is keyed by a market-time `as_of`.
- G3: Out-of-sample evidence outranks in-sample evidence, and in-sample evidence is deflated by the number of trials.
- G4: Retrieval by regime similarity, using the label only as a fallback.
- G5: A structured, token-budgeted `MemoryPack` with cited ids. The ids served are logged, and their hash fills `memory_snapshot_id`.
- G6: Controllable from harness config (`off | read | read_write`) so the existing benchmark arms can measure memory ablations.

**Non-goals (v1):** vector DB or embeddings service, LLM-written consolidation, multi-user concurrency. SQLite remains the store because the data is small (≈10⁴–10⁵ trials).

## 3. Architecture

```
                 ┌──────────────── MemoryService (src/memory/service.py) ────────────────┐
 writes ───────► │ record_trial()  attach_oos()  record_note()  upsert_hypothesis()      │
                 │        │                                                              │
                 │        ▼                                                              │
                 │  L1 EPISODIC  mem_trials   (append-only, every backtested proposal)   │
                 │  L3 NOTES     mem_notes    (hypotheses, NLA narratives, memos)        │
                 │  L4 AUDIT     mem_reads    (what was served to whom)                  │
                 │        │  derived at query time, filtered by as_of                    │
                 │        ▼                                                              │
                 │  L2 SEMANTIC  beliefs      (per config_key × regime neighbourhood)    │
                 │        │                                                              │
 reads ◄──────── │ recall(MemoryQuery) ──► Retriever ──► PackBuilder ──► MemoryPack      │
                 └───────────────────────────────────────────────────────────────────────┘
```

Key decision: **beliefs are computed from visible trials at query time, not stored.** A stored aggregate would already include rows that some past `as_of` must not see. Computing on the fly makes the point-in-time guarantee hold by construction. At this data size a filtered `GROUP BY` takes milliseconds. An optional cache keyed by `(as_of=None, scope)` can serve live mode.

## 4. Data model

### L1: `mem_trials` (source of truth)

One row per backtested proposal. Winners and losers are both recorded.

```sql
CREATE TABLE mem_trials (
  trial_id        TEXT PRIMARY KEY,          -- uuid4 hex (full, not [:8])
  run_id          TEXT NOT NULL,             -- RunManifest.run_id
  episode_id      TEXT,                      -- benchmark episode, NULL for live runs
  iteration       INTEGER NOT NULL,
  created_at      TEXT NOT NULL,             -- wall clock, audit only
  data_start      TEXT NOT NULL,             -- first bar the backtest saw (ISO date)
  data_end        TEXT NOT NULL,             -- last bar the backtest saw  ← visibility key
  asset           TEXT NOT NULL,
  strategy_type   TEXT NOT NULL,
  params_json     TEXT NOT NULL,             -- canonical: sorted keys, floats rounded
  config_key      TEXT NOT NULL,             -- sha1(strategy_type|asset|params_json)[:16]
  regime_label    TEXT NOT NULL,
  regime_vec_json TEXT NOT NULL,             -- see §5.2
  generation_method TEXT NOT NULL,           -- llm | grid | random | memory_seed | swarm | mutation
  hypothesis_id   TEXT,                      -- → mem_notes
  is_sharpe REAL, is_return REAL, is_max_dd REAL, is_trades INTEGER,
  is_sortino REAL, is_calmar REAL, is_boot_p5 REAL, is_n_days INTEGER,
  oos_sharpe      REAL,                      -- NULL until attach_oos()
  oos_return REAL, oos_max_dd REAL,
  oos_start TEXT, oos_end TEXT,              -- oos_end also gates visibility (§5.1)
  outcome         TEXT NOT NULL,             -- accepted | watch | rejected | error
  failure_mode    TEXT,                      -- negative_sharpe | below_threshold | drawdown | too_few_trades | oos_decay | error
  reasoning       TEXT DEFAULT '',
  source          TEXT NOT NULL              -- agent_graph | swarm | search_arm:<name> | backfill
);
CREATE INDEX ix_trials_scope ON mem_trials (strategy_type, asset, data_end);
CREATE INDEX ix_trials_config ON mem_trials (config_key, data_end);
CREATE INDEX ix_trials_run ON mem_trials (run_id);
```

This table replaces `strategy_runs`, `alpha_candidates`, and `failure_records` as the store of record. A failure is an `outcome`/`failure_mode` on the trial row, not a separate record, so repeated failures deduplicate for free through `config_key`.

### L3: `mem_notes`

```sql
CREATE TABLE mem_notes (
  note_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
  data_end TEXT,                             -- market-time visibility, NULL = timeless (e.g. literature)
  kind TEXT NOT NULL,                        -- hypothesis | nla | memo | critique
  status TEXT,                               -- hypotheses: proposed|backtested|confirmed|rejected
  strategy_type TEXT, regime_label TEXT, config_key TEXT,
  body TEXT NOT NULL, meta_json TEXT DEFAULT '{}', quality REAL DEFAULT 0
);
CREATE VIRTUAL TABLE mem_notes_fts USING fts5(body, content='mem_notes', content_rowid='rowid');
```

This absorbs `hypotheses` and `nla_records`. The hypothesis lifecycle becomes real: `confirmed`/`rejected` is set by `attach_oos()` on the linked trials, not by hand.

### L4: `mem_reads`

```sql
CREATE TABLE mem_reads (
  read_id TEXT PRIMARY KEY, run_id TEXT, iteration INTEGER, as_of TEXT,
  query_json TEXT, served_ids_json TEXT, snapshot_id TEXT, created_at TEXT
);
```

`snapshot_id = sha1(sorted(served_ids) + as_of)[:16]` is written to `RunManifest.memory_snapshot_id` and `episode_report.memory_entries_visible`. This replaces the hand-built list in `policy_eval.py`.

## 5. Read path

### 5.1 Visibility (G2)

```python
@dataclass(frozen=True)
class MemoryQuery:
    as_of: Optional[str]            # market date; None only in live mode
    strategy_types: Sequence[str]
    asset: Optional[str] = None
    regime_label: str = ""
    regime_vec: Optional[Dict[str, float]] = None
    exclude_run_id: Optional[str] = None   # don't read your own in-flight trials as "prior"
    k_beliefs: int = 6
    token_budget: int = 1200
```

Every SQL read in the service goes through one helper:

```sql
WHERE data_end < :as_of
  AND (oos_end IS NULL OR oos_end < :as_of)      -- OOS evidence counts only if its window closed before as_of
```

A trial row stays visible when its OOS window has not yet closed. In that case its OOS fields are treated as NULL: a small wrapper blanks `oos_*` whenever `oos_end >= as_of`.

- For benchmark episodes, `as_of = episode.dev_start`. This is stricter than "earlier episode", which is correct because it also blocks an earlier episode whose holdout overlaps the current dev window.
- In live mode `as_of=None` and everything is visible.
- Backfilled legacy rows have no market dates. They get `data_end = '9999-12-31'`, which makes them invisible to any dated query and visible in live mode. That is the only safe default.

`filter_visible_memory` stays as a test oracle. The property test asserts that `service.recall(as_of=X)` never returns an id that the oracle would drop.

### 5.2 Regime similarity (G4)

Use the continuous features `RegimeContext` already computes, z-scored with fixed constants so that vectors stored in different runs stay comparable:

```python
REGIME_FEATURES = {            # (center, scale)
    "vix_percentile":   (50.0, 25.0),
    "momentum_63d":     (0.0, 0.08),
    "vol_vs_avg":       (1.0, 0.35),
    "drawdown_from_peak": (-0.05, 0.08),
    "price_vs_sma200":  (0.0, 0.08),
}
sim(a, b) = exp(-||z(a) - z(b)||² / (2·τ²)),   τ = 1.0
```

Retrieval uses the vector when it is present, and falls back to exact `regime_label` match only for legacy rows without a vector. The linear scan in Python costs O(n) and is fine up to about 10⁵ rows. Past that, pre-filter by `strategy_type` and the vol-bucket prefix of the label.

### 5.3 Beliefs (L2, G3)

For each visible `config_key` (and each `strategy_type` as a family-level rollup), compute:

```
w_i       = sim(regime_vec_i, query_vec) · 0.5^(Δt_market_i / H)     # H = 504 trading days half-life
n_trials  = Σ 1 over trials in scope (all configs of this strategy_type)  → used for deflation
oos_mean  = Σ w_i·oos_sharpe_i / Σ w_i       over trials with OOS evidence
is_mean   = Σ w_i·is_sharpe_i  / Σ w_i

deflation = sqrt(2·ln(max(n_trials, 2))) · se(SR),   se(SR) ≈ sqrt((1 + SR²/2) / years)
evidence  = oos_mean                         if n_oos ≥ 1
          = is_mean − deflation − λ          otherwise  (λ = 0.25: in-sample-only penalty)
shrunk    = evidence · n_eff / (n_eff + κ)   # κ = 3, n_eff = (Σw)² / Σw²
```

The deflation term is the expected maximum Sharpe from pure noise across `n_trials` looks (Bailey & López de Prado, *The Deflated Sharpe Ratio*). It directly counters problem 1.

Verdicts replace `AgenticMemoryLayer._verdict`:

| verdict | rule |
|---|---|
| `works` | `n_oos ≥ 1` and `shrunk ≥ min_acceptable_sharpe` |
| `promising` | in-sample only and `shrunk > 0` (a candidate to *test*, not a seed) |
| `regime_sensitive` | high-sim trials with OOS > gate *and* others with OOS < 0 |
| `decays` | mean(is − oos) > 0.5 over ≥ 2 OOS trials (the overfit signature) |
| `avoid` | `n_oos ≥ 1` and `oos_mean < 0`, or ≥ 3 in-sample rejections with `n_eff ≥ 2` |
| `unproven` | anything else |

The parameter bucketing in `_bucket_params` is currently hard-coded for three strategies. It is replaced by neighbourhoods on the `ParameterGrid`: two configs are neighbours if they differ by one grid step in one parameter. Beliefs about a neighbourhood are aggregated the same way, which removes the per-strategy special cases.

### 5.4 MemoryPack (G5)

```python
@dataclass
class MemoryPack:
    beliefs: List[Belief]                 # ranked by shrunk evidence × max sim
    avoid_keys: Set[str]                  # config_keys with verdict avoid/decays
    seeds: List[Dict[str, Any]]           # params with verdict == works only, max 2
    gaps: List[Dict[str, Any]]            # grid neighbourhoods with 0 visible trials in sim>0.5 regimes
    notes: List[NoteRef]                  # top FTS/quality notes for strategy+regime
    served_ids: List[str]
    snapshot_id: str
    def to_prompt(self) -> str: ...       # budgeted rendering, below
```

Prompt rendering gives each section a fixed share of `token_budget` (estimated as chars/4) and cites ids so the LLM's reasoning can be traced back:

```
MEMORY (as of 2021-03-01, 214 prior trials, snapshot 3fa9c1e2)
Evidence is out-of-sample unless marked [IS]. In-sample numbers are deflated for 57 trials in scope.
WORKED IN SIMILAR REGIMES                                        (≤40%)
  [b:7c1e] momentum fast=20 slow=120 · OOS Sharpe 0.84 (n=3, sim 0.82)
AVOID                                                            (≤25%)
  [b:a02f] momentum fast=5 slow=30 · decays: IS 1.9 → OOS −0.2 (n=4)
UNTESTED NEARBY                                                  (≤15%)
  momentum slow∈[150,180] has 0 trials in regimes like this one
NOTES                                                            (≤20%)
  [n:91d0] "Trend signals lagged the 2020 V-shaped rebound…"
```

## 6. Write path

| Hook | Call | Replaces |
|---|---|---|
| `backtest_node`, per proposal | `record_trial(...)` with IS metrics, `data_start/data_end` from the sliced OHLCV, `regime_vec` from `state["context"]` | nothing (losers were never stored) |
| `agent_graph` holdout step (≈L421) | `attach_oos(trial_id, holdout metrics, window)` | `best["holdout_sharpe"]` only |
| `search_arms._grade_on_holdout` | `attach_oos(...)` for the winner, `source="search_arm:<arm>"` | nothing, and this is the main source of real OOS evidence |
| `reflect_node` | none. The failure mode is computed in `record_trial` | `store_failure` per result per iteration |
| `store_node` | `record_note(kind="nla", config_key=…)` for the narrative | 3 separate stores |
| swarm `MemoryAgent.store_swarm_results` | `record_trial(..., source="swarm")` per ranked item, with walk-forward windows as OOS | `PastResult` with mean Sharpe |
| `HypothesisGenerator` | `upsert_hypothesis`, linked via `hypothesis_id` on trials | unused `HypothesisMemory` |
| `ingest_nla_jsonl` | `record_note(kind="nla")` | same |

Writes are idempotent where it matters. `attach_oos` is an `UPDATE ... WHERE trial_id=?`. `record_trial` uses a caller-provided `trial_id` when it is retried.

## 7. Consumers

- **`analyze_node`**: build one `MemoryQuery(as_of=state.get("as_of"), regime_vec=context.as_vec(), …)`, store `pack` in state, and set `state["memory_context"] = pack.to_prompt()`. `RegimeContext.alpha_memory_context`/`nla_memory_context` collapse into a single `memory_context`.
- **`ProposalGenerator`**:
  - `_memory_generate` gets its seeds from `pack.seeds`, capped at 2 of `n_proposals` and tagged `generation_method="memory_seed"` so the benchmark can separate them.
  - `_rejected_param_keys` becomes `pack.avoid_keys`.
  - `failure_memory_section` renders from the pack.
- **Swarm**: `run_memory_agent` calls the same `recall`. `memory_patterns` becomes `[b.to_sentence() for b in pack.beliefs]`. Both agents now see identical memory.
- **Harness**: add `HarnessConfig.memory = {mode: off|read|read_write, k_beliefs, token_budget, allow_seeds: bool}`. `frozen_agent` = `off`, `frozen_agent_memory` = `read_write` with `as_of=episode.dev_start`. This adds a third arm, `memory_no_seeds`, to test whether seeding helps or only adds overfitting.
- **CLI / Streamlit**: `agentquant memory --beliefs --regime-like 2020-03-16` shows the verdict table, and `--trials` shows raw rows. Streamlit's Alpha/NLA panels read `beliefs`/`notes`.

## 8. Module layout

```
src/memory/
  __init__.py          # MemoryService, MemoryQuery, MemoryPack
  schema.py            # DDL + migrations (PRAGMA user_version)
  canonical.py         # canonical_params(), config_key(), regime_vec()
  service.py           # record_*/attach_oos/recall; owns the visibility helper
  beliefs.py           # weighting, deflation, verdicts, grid neighbourhoods
  pack.py              # MemoryPack + budgeted rendering
  backfill.py          # legacy tables → mem_trials/mem_notes
```

Legacy classes stay as thin deprecation shims for one release: `StrategyMemory`, `AlphaStore`, `NLAMemoryStore`, `HypothesisMemory`, and `AgenticMemoryLayer`. Their `store()` forwards to `record_trial`/`record_note`, and their `to_prompt_context()` forwards to `recall().to_prompt()`.

## 9. Migration plan (each phase is one PR)

1. **Schema + service + backfill.** Add `src/memory/` and write `backfill.py`, which maps `strategy_runs`/`alpha_candidates` rows to trials with `data_end='9999-12-31'` and `source="backfill"`, and `nla_records`/`hypotheses` to notes. Nothing reads the new tables yet. Tests cover the schema and backfill round-trip.
2. **Write path.** Add hooks in `backtest_node`, the holdout step, `_grade_on_holdout`, and the swarm, writing to both old and new tables.
3. **Read path.** `analyze_node`, the swarm, and `ProposalGenerator` switch to `recall()`. Populate `mem_reads` and `memory_snapshot_id`. Add the harness `memory` config and the `memory_no_seeds` arm.
4. **Cleanup.** Stop the dual-write, turn the legacy classes into shims, and point CLI/Streamlit at beliefs.
5. **Later.** Optional embeddings over `mem_notes` (the ROADMAP's "retrieval-augmented hypothesis generation") and LLM-written consolidation notes (`kind="critique"`) summarising `decays` beliefs.

## 10. Tests to add

- **Leakage**: write a trial with `data_end=2022-01-01` and `oos_end=2022-06-01`. `recall(as_of=2021-12-31)` does not return it. `recall(as_of=2022-03-01)` returns it with `oos_sharpe=None`. Also a Hypothesis-style property test against `filter_visible_memory`.
- **Dedup**: the same params recorded 3 times give one belief with `n=3`. Its failures do not multiply rows in the pack.
- **In-sample never "works"**: IS Sharpe 3.0 and zero OOS gives verdict `promising`, and the config is not in `seeds`.
- **Deflation monotonic**: the same IS Sharpe gets lower `evidence` as `n_trials` grows.
- **Decay detection**: IS 2.0 / OOS −0.3 twice gives `decays` and lands in `avoid_keys`.
- **Similarity**: a trial with the same label but a distant vector ranks below one with a different label but a near vector.
- **Budget**: `len(pack.to_prompt()) / 4 ≤ token_budget` for 10k synthetic trials.
- **Ablation**: `mode="off"` means zero rows written and an empty pack. `mode="read"` means zero rows written.
- **Snapshot determinism**: the same DB and query give the same `snapshot_id`.

## 11. How we will know it works

Use the existing `scripts/fair_search_benchmark.py` episodes with paired comparisons per episode and seed:

1. `frozen_agent_memory` minus `frozen_agent` holdout Sharpe, with a paired bootstrap CI. **Memory has to beat no-memory.** Today there is no such claim.
2. **Overfit gap** (IS − OOS Sharpe) of `memory_seed` proposals vs. fresh proposals. The redesign succeeds if seeds stop showing a larger gap.
3. `memory_no_seeds` vs `frozen_agent_memory` separates the value of the context from the value of the seeds.
4. **Repeat rate**: the share of proposals whose `config_key` already has verdict `avoid` in visible memory. Target ≈ 0.

## 12. Open questions

- **Per-asset or pooled:** should `config_key` include `asset`? Proposal: yes for trials, but beliefs roll up across assets at 0.5 weight so SPY evidence can inform QQQ.
- **Half-life `H`:** 2y is a guess. It should be tuned on the benchmark rather than chosen by hand.
- **Walk-forward folds:** should swarm results with walk-forward folds count as `n` OOS observations or as one? Proposal: count them as one trial with `oos_sharpe` set to the fold mean and the fold std kept in metadata, to avoid inflating `n_eff`.
