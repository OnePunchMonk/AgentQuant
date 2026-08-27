# Contributing to AgentQuant

Thanks for interest in improving AgentQuant! Here's how to contribute.

## Getting Started

### Prerequisites
- Python 3.10+
- ~5 years of market data (auto-fetched from yfinance)

### Setup
```bash
# Clone repo
git clone https://github.com/OnePunchMonk/AgentQuant.git
cd AgentQuant

# Install with dev dependencies
pip install -e ".[dev,llm]"

# Set API keys (optional; agent degrades gracefully without them)
cp .env.example .env
export ANTHROPIC_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

## Making Changes

### Run Tests
Before submitting changes, verify all tests pass:
```bash
pytest tests/ -v --cov=src
```

### Workflow
1. **Create a branch** for your feature or fix
2. **Make changes** in `src/`
3. **Run tests locally** — ensure all pass
4. **Open a PR** with a clear description

### CI/CD Checks
Your PR will automatically run:
- ✅ **Tests** — 3 Python versions (3.10, 3.11, 3.12)
- ✅ **Linting** — ruff check
- ✅ **Type checking** — mypy
- ✅ **Harness validation** — config files present and valid
- ✅ **Security** — no secrets leaked

All must pass before merge.

## Code Standards

### Style
- Use `ruff` for formatting: `ruff check src/ --fix`
- Type hints required for all functions
- No secrets (API keys, credentials) in code or commit messages

### Testing
- Write tests for new functionality
- Keep tests focused and fast
- Use descriptive test names: `test_momentum_strategy_buys_on_signal()`

### Documentation
- Update `docs/` if changing architecture
- Add docstrings to public functions (one-line is fine unless behavior is non-obvious)
- Update `CHANGELOG.md` for user-facing changes

## Areas for Contribution

### High Priority
- [x] Core agent loop (ReAct)
- [x] Tool orchestration (Claude reasoning)
- [x] Harness evolution (GA + DE)
- [ ] **Research Agent** (Phase 1: Literature discovery)
  - Implement `fetch_and_extract_content()` tool
  - Validate against real finance papers
  - See `docs/RESEARCH_AGENT_DESIGN.md`

### Medium Priority
- [ ] Multi-objective optimization (Sharpe + Drawdown)
- [ ] Parameter space expansion (more strategies, more parameter ranges)
- [ ] Streaming data support (live market feeds vs. historical only)

### Nice to Have
- [ ] Nested optimization (evolve GA parameters themselves)
- [ ] Online learning (continuous re-adaptation)
- [ ] Multi-strategy portfolio (per-strategy harnesses)
- [ ] Dashboard UI (web interface for running agent)

## Questions?

- **Bugs** — Open an issue with reproduction steps
- **Ideas** — Discuss in issues before large PRs
- **Architecture** — Read `DESIGN.md` and `docs/TOOL_INTEGRATION_GUIDE.md`
- **History** — Check `CHANGELOG.md` for what changed and why

## Code of Conduct

Be respectful. Help others learn. Assume good intent.

---

**Thank you for contributing!** 🚀
