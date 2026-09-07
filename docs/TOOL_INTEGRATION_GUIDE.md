# Tool Integration & Harness Evolution Guide

This guide covers the tool-calling architecture integrated into the agent graph and how to enable harness evolution features.

## Architecture Overview

The agent graph now includes **tool orchestration** for:
1. **Market context gathering** — regime signals, web search
2. **Quality assessment** — eval benchmark scoring
3. **Falsifiable claim tracking** — proposal prediction accuracy

```
analyze ──► hypothesize (with tools) ──► backtest ──► reflect (with claim scoring) ──► store
                    ▲                                              │
                    └──────────────── (retry if needed) ◄──────────┘
```

## Setup

### 1. Install Dependencies

The tool system requires Claude SDK and Tavily for web search (optional):

```bash
pip install "anthropic>=0.30" "tavily-python>=0.3"
```

Or install the project with LLM extras:
```bash
pip install -e ".[llm]"
```

### 2. Configure API Keys (`.env`)

Copy `.env.example` to `.env` and set your API keys:

```bash
# Required for tool orchestrator (Claude tool-use)
ANTHROPIC_API_KEY=your_key_here

# Optional for web search in tools
TAVILY_API_KEY=your_key_here

# Existing keys (Gemini, FRED, etc.)
GOOGLE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

**⚠️ Important:** `.env` is in `.gitignore`. Never commit API keys.

### 3. Enable Tools in Agent

The `hypothesize_node` in `src/agent/agent_graph.py` now:
1. **Tries tool orchestration first** — if Claude/Tavily available
2. **Falls back to ProposalGenerator** — if tools fail or APIs missing
3. **Scores claims in reflect** — tracks proposal accuracy

No code changes needed to activate — tools are tried automatically.

## How It Works

### Tool Orchestrator Loop

When `hypothesize_node` is called:

```python
# 1. Build tool registry
registry = get_default_registry()

# 2. Create orchestrator
orchestrator = ToolOrchestrator(registry)

# 3. Run Claude with tools
result = orchestrator.run_tool_loop(
    user_prompt="Generate 5 momentum proposals...",
    regime_context=context,
    strategy_type="momentum",
    max_turns=2,
)

# 4. Claude can call tools like:
#    - get_regime_context() → retrieve stored regime signals
#    - search_market_sentiment() → web search via Tavily
#    - extract_parameter_recommendations() → access parameter grid
#    - run_benchmark_to_assess_quality() → eval harness quality

# 5. Tool results fed back to Claude
# 6. Claude reasons and generates proposals (as JSON)
# 7. Proposals parsed into Proposal objects (TODO)
```

### Available Tools

| Tool | Purpose | Requires |
|------|---------|----------|
| `get_regime_context` | Retrieve stored market context | StrategyMemory |
| `search_market_sentiment` | Web search market news, VIX | Tavily API |
| `search_strategy_research` | Web search strategy ideas | Tavily API |
| `extract_parameter_recommendations` | Access parameter grid | Config grid |
| `run_benchmark_to_assess_quality` | Quality assessment eval | Recent backtest results |

### Falsifiable Claims

When a proposal is generated with a claim (e.g., "This window length will improve Sharpe by 15%"):

1. **Proposal generation** — claim is recorded in `Proposal.reasoning`
2. **Reflect node** — `_score_falsifiable_claims()` currently records a confidence/outcome heuristic
3. **Memory storage** — proposal and backtest context can be retained for future evaluation
4. **Harness eval** — structured numerical forecasts and measured accuracy remain future work

This provides inputs for a future feedback loop; it does not yet establish prediction accuracy.

## Extending the Tool System

### Add a New Tool

1. Implement the callable function:

```python
# src/agent/tools/registry.py

def _my_new_tool(param1: str) -> Dict[str, Any]:
    """Your tool implementation."""
    return {"result": "..."}
```

2. Register it in `get_default_registry()`:

```python
registry.register(
    Tool(
        name="my_new_tool",
        description="What it does",
        input_schema={
            "properties": {
                "param1": {"type": "string"},
            },
            "required": ["param1"],
        },
        callable_fn=_my_new_tool,
        category="custom",  # For organization
    )
)
```

3. The tool is now available in Claude tool-use loops and can be called by agents.

### Modify Tool Behavior

Tools and their schemas live in `src/agent/tools/`:

- **Tool definitions** — `registry.py` (callable + schema)
- **Web search** — `_search_market_sentiment`, `_search_strategy_research`
- **Evaluation** — `evals.py` (quality assessment)
- **Orchestration** — `orchestrator.py` (Claude loop)

Change any of these and the agent's next run uses the new version.

## Evaluation & Harness Quality

### Run Quality Assessment

Agents can call `run_benchmark_to_assess_quality` to self-evaluate:

```python
result = registry.execute_tool(
    "run_benchmark_to_assess_quality",
    {"strategy_type": "momentum", "regime": "Bull"},
)

# Returns:
{
    "overall_score": 0.75,
    "passed": True,
    "metrics": {
        "oos_sharpe": 0.52,
        "stability": 0.68,
        "generalization_gap": 0.12,
        "tool_accuracy": 0.80,
    },
    "thresholds": {...},
    "recommendation": "Harness performing well. Consider scaling."
}
```

### Metrics Explained

- **OOS Sharpe** — Out-of-sample Sharpe ratio (primary metric)
- **Stability** — Consistency across regimes (1 - normalized std dev)
- **Generalization Gap** — Train Sharpe − held-out Sharpe (lower is better)
- **Tool Accuracy** — % of falsifiable claims that materialized

## Walk-Forward Validation

To prevent overfitting, partition price history into three windows:

```python
# Train window: [t0, t1)
#   - Agent evolves harness
#   - Tests proposals on held-in backtest data

# Validation window: [t1, t2)
#   - Accept/reject decisions based on OOS Sharpe
#   - Check falsifiable claims

# Held-out test window: [t2, t3)
#   - Touched once, at epoch end
#   - True generalization measurement
#   - Computes generalization gap (train improvement − test improvement)
```

Implementation in `experiments/walk_forward_context.py` and `walk_forward_context_with_costs.py`.

## Logging & Debugging

Enable DEBUG logging to see tool calls and claims scoring:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Output includes:
- Tool invocations and results
- Proposal generation methods
- Falsifiable claim scoring
- Harness evolution decisions

## Troubleshooting

### Tools Not Found

If you see `Tool <name> not found`, check:
- Tool is registered in `get_default_registry()`
- No typos in tool name

### Missing API Keys

- **Tavily** — Web search tools return empty results if `TAVILY_API_KEY` not set
- **Anthropic** — Tool orchestrator falls back to ProposalGenerator if `ANTHROPIC_API_KEY` not set

Both degrade gracefully; harness works without them.

### Falsifiable Claims Not Scoring

Check:
- `Proposal.reasoning` is populated (has the claim text)
- `reflect_node` is called after backtest
- `_score_falsifiable_claims()` is logging debug output

## Next Steps

1. **Parse Claude proposals** — Complete `_hypothesize_with_tools()` to extract proposals from Claude JSON
2. **Evaluate on held-out data** — Run walk-forward harness with three-way split
3. **Track generalization gap** — Measure train vs. held-out improvement
4. **Evolve the grid** — Learn parameter space from results
5. **Multi-agent swarm** — Add specialist agents (critic, regime analyst, etc.)

## References

- `src/agent/tools/README.md` — Tool system architecture
- `src/agent/tools/example_integration.py` — Integration examples
- `DESIGN.md` — Agent graph and harness architecture
- `docs/PAPER_DRAFT.md` — Research methodology
- Research: [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)
