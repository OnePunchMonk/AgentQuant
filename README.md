# Alpha Optimizer: Regime-Adaptive Trading Research Agent

This project is a Python-based autonomous agent designed for investment strategy research. It continuously ingests market and macroeconomic data, detects prevailing market regimes, and evaluates a library of trading strategies to adapt to changing conditions.

This is a **research tool** and is not intended for live trading.

## Features

- **Data Ingestion**: Pulls OHLCV data from `yfinance` and macroeconomic indicators from FRED.
- **Feature Engineering**: Computes technical indicators like volatility, momentum, and correlation.
- **Regime Detection**: Classifies the market state using simple heuristics (e.g., VIX levels).
- **LLM Planner**: Uses Google's Gemini Pro to propose strategy adjustments based on the current regime.
- **Vectorized Backtesting**: Rapidly tests strategy proposals using `vectorbt`.
- **Modular Design**: Easily extensible with new strategies, data sources, and features.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OnePunchMonk/AlphaOptimizer
    cd AlphaOptimizer
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    - Copy the `.env.example` file to a new file named `.env`.
    - `cp .env.example .env`
    - Edit the `.env` file to add your API keys for Google Gemini and FRED.

## How to Run

The primary way to interact with the agent is through the demonstration notebook.

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open and run `notebooks/demo_agent_run.ipynb`**.

This notebook provides a step-by-step walkthrough of the agent's reasoning loop:
- Ingesting data
- Calculating features
- Detecting the market regime
- Running a baseline backtest
- Prompting the Gemini planner for new ideas
- Executing and evaluating the LLM's suggestions

## Project Structure

(See the project structure diagram in DESIGN.md or the main response)