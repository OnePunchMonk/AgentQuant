# AgentQuant (Prototype)

**A modular Python framework for quantitative strategy research and backtesting.**

> **⚠️ Note:** This project is currently a **structural prototype**. The "AI Agent" logic is currently simulated using stochastic (random) generation to demonstrate the workflow. The actual LLM integration (LangChain/Gemini) requires uncommenting and API setup.

## 🎯 What This Project Is

AgentQuant is a structured codebase designed to automate the lifecycle of a trading strategy. It handles:

1.  **Data Ingestion:** Fetching market data (OHLCV).
2.  **Feature Engineering:** Calculating indicators (Momentum, Volatility, SMA).
3.  **Regime Detection:** Classifying market states (e.g., "Bear", "Bull") using heuristic rules.
4.  **Backtesting:** Running strategies against historical data.

It is designed as a **foundation** for developers who want to build an AI-driven trading bot but need the messy boilerplate (data handling, pipeline architecture) handled first.

## ⚙️ How It Works (The Honest View)

### 1. The "Brain" (`src/agent`)
* **Current State:** The strategy planner currently uses **randomized parameter search** to simulate an AI proposing strategies.
* **Future Goal:** To enable the actual AI, you must uncomment the LangChain imports in `langchain_planner.py` and provide a Google Gemini API key.
* **Why?** This allows the application to run and demo the UI without requiring expensive API credits during development.

### 2. Market Regime (`src/features/regime.py`)
* Uses hardcoded logic based on VIX levels and Momentum to classify the market into states like:
    * `Crisis-Bear` (VIX > 30, Negative Momentum)
    * `MidVol-Bull` (VIX 20-30, Positive Momentum)
    * `LowVol-MeanRevert` (VIX < 20, Flat Momentum)

### 3. Backtesting (`src/backtest`)
* Includes a fast, vectorized backtester (`simple_backtest.py`) capable of testing Momentum and Mean Reversion logic.
* Calculates Sharpe Ratio, Max Drawdown, and Total Return.

## 🚀 Quick Start

**Prerequisites:** Python 3.10+

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/OnePunchMonk/AgentQuant.git](https://github.com/OnePunchMonk/AgentQuant.git)
    cd AgentQuant
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**
    ```bash
    # Runs the Streamlit UI with the simulated agent
    python run_app.py
    ```

## 📂 Project Structure

```text
AgentQuant/
├── src/
│   ├── agent/          # Strategy planner (Currently randomized/simulated)
│   ├── data/           # Data fetching (yfinance wrapper)
│   ├── features/       # Technical indicators & Regime detection
│   ├── backtest/       # Vectorized backtesting engine
│   └── strategies/     # Strategy logic definitions
├── config.yaml         # Configuration (Tickers, Dates)
└── run_app.py          # Main entry point

This software is for educational purposes only.
