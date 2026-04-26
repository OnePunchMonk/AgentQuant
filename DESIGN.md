# AgentQuant — Technical Design Document (v2.0)

> **Last updated:** April 2026. This document reflects the post-refactor architecture.
> The original design document is preserved in `docs/EXPERIMENTAL_DETAILS.md`.

---

## Executive Summary

AgentQuant is a regime-adaptive quantitative research platform. The v2.0 refactor transforms it from a stateless parameter-tuning script into a defensible research tool with:

- A **real ReAct agent loop** (analyze → hypothesize → backtest → reflect → store)
- **Scientific comparability**: LLM proposals are constrained to a canonical `ParameterGrid`, making LLM vs. grid-search comparisons valid
- **Statistical regime detection**: VIX percentile (relative) instead of hardcoded absolute thresholds
- **Look-ahead bias guards** enforced at the backtest engine level
- **Cross-session memory** via SQLite so the agent can learn across runs
- **42 unit tests** and a GitHub Actions CI gate

---

## 1. Agent Orchestration

### 1.1 Agent Loop (`src/agent/agent_graph.py`)

The agent runs a bounded ReAct loop with 5 explicit typed nodes:

```
analyze ──► hypothesize ──► backtest ──► reflect
              ▲                              │
              └────────── (retry ≤ N) ◄──────┘
                                             │
                                           store
```

```python
class AgentState(TypedDict, total=False):
    ohlcv_data: Dict[str, pd.DataFrame]
    features_df: pd.DataFrame
    context: Optional[RegimeContext]
    proposals: List[Proposal]
    results: List[Dict]
    best_result: Optional[Dict]
    iteration: int
    max_iterations: int
    strategy_type: str
    asset: str
    should_continue: bool
    memory_context: str
    run_log: List[str]
```

**Nodes:**

| Node | Purpose |
|---|---|
| `analyze` | Build `RegimeContext` from features + regime detector |
| `hypothesize` | Generate proposals via `ProposalGenerator` |
| `backtest` | Run tournament: all proposals → rank by Sharpe |
| `reflect` | Accept if Sharpe ≥ `min_acceptable_sharpe`, else retry |
| `store` | Persist best result to `StrategyMemory` SQLite |

The loop retries at most `config.agent.max_iterations` times (default 3). On max iteration, accepts the best result found regardless of threshold.

### 1.2 LLM Abstraction (`src/agent/base_planner.py`)

```python
class BasePlanner(ABC):
    def generate_proposals(self, prompt: str, n: int) -> List[Dict]: ...
    def is_available(self) -> bool: ...
```

Concrete implementations:
- `GeminiPlanner` — `google-generativeai` SDK
- `LangChainPlanner` — `langchain-google-genai`
- `OpenAIPlanner` — `openai`
- `FallbackPlanner` — returns `[]`, triggering grid search

`create_planner()` factory checks credential availability in order and returns the first viable planner.

### 1.3 Proposal Generator (`src/agent/proposal_generator.py`)

Single entrypoint replacing 4 legacy planner files:

**Fallback chain:**
1. LLM (via `BasePlanner`) → validates JSON response
2. Grid search with regime-aware prior (shorter windows in Crisis/Bear, longer in LowVol/Bull)
3. Random sample from canonical grid

All proposals are constrained to the `ParameterGrid` — the LLM **selects from** the grid, not generates free-form integers. This makes A/B comparison against grid-search baselines statistically valid.

```python
@dataclass
class Proposal:
    params: Dict[str, Any]
    confidence: float          # LLM self-assessed
    regime_characteristic_used: str
    reasoning: str
    generation_method: str     # "llm" | "grid_search" | "random"
```

### 1.4 Strategy Memory (`src/agent/strategy_memory.py`)

SQLite-backed memory providing cross-session learning:

```sql
CREATE TABLE strategy_runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT,
    regime TEXT,
    strategy_type TEXT,
    params TEXT,      -- JSON
    sharpe REAL,
    total_return REAL,
    max_drawdown REAL,
    confidence REAL,
    generation_method TEXT,
    reasoning TEXT
)
```

`StrategyMemory.to_prompt_context(regime)` injects past results into the next LLM prompt so the agent can avoid repeating failures.

---

## 2. Regime Detection (`src/features/regime.py`)

### 2.1 VIX Percentile Classification

**Old approach (removed):**
```python
if vix > 30: return "High Volatility"
if vix > 20: return "Medium Volatility"
```
This breaks when VIX structurally shifts (e.g., post-2020 baseline is higher).

**New approach:**
```python
vix_percentile = percentileofscore(vix_history_252d, current_vix)
# Crisis: >85th | HighVol: >65th | MidVol: >35th | LowVol: <35th
```

### 2.2 Regime Label Format

`{VolRegime}-{TrendRegime}`: e.g., `LowVol-Bull`, `Crisis-Bear`, `HighVol-Neutral`

**Vol regimes:** based on VIX percentile vs. trailing 252d
**Trend regimes:** based on 63-day momentum: `Bull` (>5%), `Bear` (<-5%), `Neutral`

### 2.3 Confidence Score

```python
vol_confidence = 2 * abs(vix_percentile / 100 - 0.5)   # distance from 50th pct
mom_confidence = min(abs(momentum_63d) / 0.10, 1.0)     # normalized abs momentum
regime_confidence = (vol_confidence + mom_confidence) / 2
```

### 2.4 Optional HMM (install `hmmlearn`)

If `hmmlearn` is installed, `_try_hmm_regime()` fits a 3-state Gaussian HMM on (returns, realized_vol) features and labels states by volatility rank. Used informatively; rule-based label takes precedence.

---

## 3. Feature Engineering (`src/features/engine.py`)

Features computed (all single-level string columns after flattening):

| Feature | Details |
|---|---|
| `volatility_5d/21d/63d` | Realized vol, annualized (√252 scale) |
| `momentum_21d/63d/252d` | `close.pct_change(N)` |
| `sma_21/50/63/200` | Simple moving averages |
| `price_vs_sma63/200` | `(close/sma) - 1` |
| `rsi_14` | Wilder's smoothing RSI, bounded [0, 100] |
| `macd/macd_signal/macd_hist` | EMA(12,26,9) standard |
| `bb_upper/lower/width/pct_b` | Bollinger Bands (20, 2σ) |
| `atr_14` | Average True Range, Wilder's EWM |
| `drawdown_from_peak` | Rolling 252d drawdown |
| `vix_close` | Forward-filled from `^VIX` data |

**Stationarity check:** ADF test (via `statsmodels`) on `momentum_63d`, `volatility_21d`, `rsi_14` — logs a warning if non-stationary. Informational; does not block execution.

### 3.1 Lookback Guard (`src/features/lookback_guard.py`)

```python
@enforce_lookback(min_periods=200)
def compute_sma200(close): ...

enforcer = WarmupEnforcer(min_warmup_periods=252)
enforcer.check(df, eval_start)  # raises InsufficientWarmupError if violated
```

---

## 4. Backtest Engine (`src/backtest/`)

### 4.1 Runner (`src/backtest/runner.py`)

All strategies produce `pd.Series` of `{-1, 0, 1}` via the `Strategy` ABC. The runner applies:

```
strategy_return[t] = daily_return[t] × signal[t-1] − transaction_cost[t]
```

**Transaction costs:**
```
cost = commission + slippage + (market_impact_bps / 10000)
```
Applied proportionally to `signal.diff().abs()` (trade turnover).

**Warmup enforcement:** If `eval_start` is provided, the runner calls `WarmupEnforcer.check()` using `slow_window` (or `window`) as `min_window`. Raises `InsufficientWarmupError` rather than silently using stale indicators.

### 4.2 PerformanceMetrics (`src/backtest/metrics.py`)

Single source of truth — replaces all inline Sharpe calculations:

```python
PerformanceMetrics.sharpe(returns)              # annualized
PerformanceMetrics.max_drawdown(equity)         # positive fraction
PerformanceMetrics.calmar(equity)               # ann_ret / max_dd
PerformanceMetrics.sortino(returns)             # downside deviation
PerformanceMetrics.bootstrap_sharpe(returns, n=200, pct=5)  # p5 bootstrap
PerformanceMetrics.from_equity(equity)          # → dict of all metrics
```

---

## 5. Strategy Class Hierarchy (`src/strategies/`)

All strategies implement `Strategy(ABC)` with:
- `generate_signal(df, params) → pd.Series` — values in `{-1, 0, 1}`
- `param_schema → Dict` — JSON-serializable schema for validation

| Class | Strategy |
|---|---|
| `MomentumStrategy` | Dual MA crossover |
| `MeanReversionStrategy` | Bollinger Band reversion |
| `VolatilityStrategy` | Go long when realized vol < threshold |
| `TrendFollowingStrategy` | Triple MA (short > mid > long) |
| `BreakoutStrategy` | Price breaks rolling high/low + threshold |
| `RegimeBasedStrategy` | Dispatches to momentum/mean-reversion/vol by regime |

Registry: `STRATEGY_REGISTRY: Dict[str, Strategy]`

Legacy code using `calculate_momentum_signal()` or `run_multi_asset_strategy()` from `multi_strategy.py` continues to work via thin shims.

---

## 6. Configuration (`src/utils/config.py`)

Pydantic v2 `AppConfig` with field-level validation:

```python
cfg = load_config()              # returns AppConfig
cfg["reference_asset"]           # dict-like access (backward compat)
cfg.backtest.min_warmup_periods  # typed attribute access
cfg.get_strategy("momentum")     # → StrategyConfig with param grid
```

Invalid config (bad log level, unknown LLM provider) raises `ValidationError` at import time — fails fast instead of silently using defaults.

---

## 7. Data Layer (`src/data/ingest.py`)

**TTL-based cache invalidation:**
```python
def _is_cache_valid(file_path) -> bool:
    age_hours = (now - file_mtime).total_seconds() / 3600
    return age_hours <= config.cache.ttl_hours   # default 24h
```

All `print()` calls replaced with `logger`. FRED fetch is fully optional and gracefully skipped if `FRED_API_KEY` is absent.

---

## 8. Experiment Tracking (`experiments/results_store.py`)

```python
run = ResultsStore.make_run("walk_forward", windows=[...], notes="v2 run")
store = ResultsStore()
run_id = store.save_run(run)
```

Each record stores: `run_id`, `timestamp`, `git_hash`, `config_snapshot` (JSON), `windows` (JSON), `aggregate_metrics`.

---

## 9. Testing (`tests/`)

| File | Coverage |
|---|---|
| `test_config.py` | Pydantic validation, invalid log level, invalid provider, dict access |
| `test_metrics.py` | Sharpe (flat, known), drawdown (zero, known), from_equity keys, zero-signal flat equity |
| `test_regime.py` | LowVol-Bull, Crisis-Bear, confidence score, no-VIX fallback, VIX spike |
| `test_features.py` | Expected columns, RSI bounds, momentum accuracy, no-VIX, no-NaN |
| `test_strategies.py` | All 6 strategies registered, valid signals, backward compat, invalid strategy error |
| `test_backtest.py` | Returns result dict, invalid strategy raises, zero-signal flat equity, metrics keys |
| `test_proposal_generator.py` | Fallback without API key, valid params, correct count |

Run: `pytest tests/ -v`

---

## 10. CI/CD (`.github/workflows/ci.yml`)

- **Matrix:** Python 3.10, 3.11, 3.12
- **Steps:** `pip install -e ".[dev]"` → `ruff check` → `pytest tests/`
- **Security:** Checks that `.env` is not tracked by git

---

## 11. Removed / Replaced Components

| What was removed | Why | Replacement |
|---|---|---|
| `src/agent/planner.py` | Hardcoded Gemini, one-shot | `base_planner.py` + `proposal_generator.py` |
| `src/agent/langchain_planner.py` | Inconsistent, no fallback | `proposal_generator.py` |
| `src/agent/langchain_planner_new.py` | Dead code | Deleted |
| `src/agent/simple_planner.py` | Disconnected | Merged into fallback chain |
| `config.json` | Duplicate config | Deleted; `config.yaml` is canonical |
| `new_env.ini` | Duplicate env template | Deleted; `.env.example` is canonical |
| `quick_test.py` | Root-level test script | Moved to `tests/` |
| `requirements.txt` | Duplicate dep list | `pyproject.toml` with optional groups |

---

## 12. Future Work

- **Walk-forward with `ResultsStore`**: Update `experiments/walk_forward.py` to write each window to `ResultsStore` with git hash for full reproducibility
- **Mann-Whitney U ablation**: Compare LLM-generated proposals vs. grid-search baselines using statistical test over ≥30 windows
- **Bootstrap Sharpe tournament**: Rank proposals by `bootstrap_sharpe_p5` instead of raw Sharpe to penalize lucky parameter sets
- **Alpaca data source**: Add as optional alternative to yfinance (avoids adjusted-price inconsistencies)
- **LangGraph native StateGraph**: Replace the pure-Python loop in `agent_graph.py` with `langgraph.StateGraph` for conditional edge visualization and checkpointing

---

*For usage details, see `README.md`. For experiment results, see `docs/EXPERIMENTAL_DETAILS.md`.*
