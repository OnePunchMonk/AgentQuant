# Installation

AgentQuant requires Python 3.10 or later. The offline demo needs no API keys.

## Install and run

```bash
git clone https://github.com/OnePunchMonk/AgentQuant.git
cd AgentQuant
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
python run_app.py
```

The demo runs on synthetic prices and writes `results/demo_run.json`. It is a demonstration of the research loop, not a trading benchmark.

## Optional features

Install the extras you need from the repository root:

```bash
python -m pip install -e ".[ui]"   # dashboard
python -m pip install -e ".[llm]"  # model providers and web research
python -m pip install -e ".[dev]"  # tests and development tools
```

Extras can be combined, for example `python -m pip install -e ".[ui,llm,dev]"`. Other available extras are `data`, `backtest`, and `regime`; see [pyproject.toml](../pyproject.toml) for their dependencies.

Launch the dashboard with:

```bash
python run_app.py --app
```

For provider-backed features, copy `.env.example` to `.env` and set the keys needed by your chosen provider. Do not commit `.env` or share your keys. Market data and provider calls require network access; the offline demo does not.

## Troubleshooting

- **Missing module:** activate the environment and install the relevant extra above.
- **Missing API key:** check `.env` or your environment variables for the provider you are using.
- **Data download fails:** check the network connection and ticker symbols, then try a shorter date range.
- **Dashboard port in use:** stop the process using port 8501, or run `streamlit run src/app/streamlit_app.py --server.port 8502`.

See the [README](../README.md) for the research loop and [contribution guide](../CONTRIBUTING.md) for tests.
