# Contributing

Thanks for helping improve AgentQuant. For large changes, open an issue first to discuss the approach.

## Setup

Follow the [installation guide](docs/INSTALLATION.md), then install the development dependencies:

```bash
python -m pip install -e ".[dev,llm]"
```

API keys are optional for the offline tests. Never commit keys or `.env` files.

## Before opening a pull request

- Keep changes focused and explain the problem and approach.
- Add or update tests for changed behavior.
- Update the relevant documentation and `CHANGELOG.md` for user-facing changes.
- Run the checks used by CI:

```bash
ruff check src/ tests/
pytest tests/ -v --tb=short --cov=src
```

CI also runs a base install and offline demo, package checks, and harness validation. Mypy runs as a non-blocking check. See [the workflow](.github/workflows/ci.yml) for the full list.

Use type hints and concise docstrings where they help explain the code. Be respectful, help others learn, and assume good intent.
