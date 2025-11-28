# Experimental Details & Methodology

This document provides a deep dive into the configuration, parameters, and validation protocols used in the AgentQuant experiments.

## 1. LLM Configuration

The "Reasoning Engine" is powered by **Gemini 2.5 Flash**, selected for its balance of reasoning capability and latency.

*   **Model:** `gemini-2.5-flash`
*   **Temperature:** `0.2` (Low setting to encourage deterministic, analytical reasoning over creativity)
*   **Max Retries:** `0` (Fail-fast architecture; falls back to random search on API failure)
*   **Seed:** Non-deterministic (API side), but the `random` fallback uses standard Python seeding.

### Prompt Template
The agent is primed with a "Persona" and "Context" to induce Chain-of-Thought reasoning.

```text
Act as a Quantitative Researcher. Based on this context, select optimal parameters for a {strategy_type} Strategy.

Input:
Market Regime: {regime_name}
Technical Summary: {technical_summary}
Asset Name: {asset_name}

Task: Return a JSON object with the optimal parameters.
For Momentum strategy, provide 'fast_window' and 'slow_window'.
For other strategies, provide 'lookback_window', 'entry_threshold', 'stop_loss'.

{format_instructions}
```

## 2. Adaptation Mechanism

### Are parameters locked?
**Yes.** To ensure statistical rigor and prevent "overfitting by oscillation," the experiment uses a **Walk-Forward Validation** protocol with locked windows.

1.  **Observation Window (Train):** The agent analyzes the *previous* 6 months of data (e.g., Jan-Jun).
2.  **Decision Point:** At the boundary (e.g., July 1st), the agent selects parameters.
3.  **Execution Window (Test):** These parameters are **locked** and traded for the *next* 6 months (e.g., Jul-Dec).

### The "2025 Crash" Switch
The statement *"autonomously switched to defensive parameters during the 2025 crash"* refers to the decision made at the **boundary** preceding the crash.

*   **Context:** In Jan 2025, the agent analyzed the data from July 2024 - Jan 2025.
*   **Signal:** This training period likely exhibited rising volatility or momentum breakdown (the "pre-shocks").
*   **Action:** The agent selected fast/defensive parameters (`17/91`) *before* the test window started.
*   **Result:** When the crash intensified in the test window (Jan 2025 - Jul 2025), the agent was already in a defensive configuration.

## 3. Data Construction & Integrity

### Data Source
*   **Asset:** SPY (S&P 500 ETF)
*   **Source:** `yfinance` (Yahoo Finance API)
*   **Nature:** Historical Daily OHLCV data.

### The "2025 Crash"
In the context of this research environment (Date: Nov 2025), the "2025 Crash" is treated as **Historical Data**. It is not a synthetic perturbation but rather the actual market data retrieved from the provider for that period.

### Look-Ahead Prevention
We enforce strict **Temporal Separation** to prevent data leakage:
1.  **Slicing:** `train_df` and `test_df` are sliced by index.
2.  **Isolation:** The LLM *only* receives `train_regime` and `train_features`. It has zero access to `test_df`.
3.  **Warmup:** A 252-day warmup period is prepended to the *start* of the test window solely for indicator calculation (e.g., 200-day SMA), but no trading decisions are made on this warmup data.

## 4. Guardrails & Validation

To prevent the LLM from hallucinating degenerate parameters (e.g., `fast_window=0` or `slow_window=9999`), we employ a **Simulation-Based Validation Layer**.

### The "Tournament" Logic
The agent does not just "guess" parameters; it proposes and *tests* them.

1.  **Proposal:** The LLM generates $N$ candidate parameter sets (e.g., `50/200`, `20/50`, `10/30`).
2.  **Validation (In-Sample):** Each proposal is immediately backtested on the **Training Data** (the past 6 months).
3.  **Selection:** The parameter set with the highest **Sharpe Ratio** on the *Training Data* is selected for the *Test Window*.

**Safety Net:**
*   If the LLM generates invalid JSON or parameters that cause runtime errors, the system catches the exception and falls back to a **Random Search** baseline.
*   This ensures the agent always outputs a valid, executable strategy, even if the LLM fails.
