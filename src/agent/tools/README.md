# Tool Registry & Orchestration

This module provides a composable tool system for agentic decision-making in AgentQuant.

## Architecture

### Registry (`registry.py`)

Defines and manages available tools:

- **Tool Schema**: JSON Schema describing inputs/outputs
- **Callable Implementation**: Python function backing each tool
- **Claude Integration**: Automatic conversion to Claude API tool format

```python
from src.agent.tools import get_default_registry

registry = get_default_registry()
tools = registry.to_claude_tools()  # For Claude API
```

### Orchestrator (`orchestrator.py`)

Runs Claude tool-use loops for agent decision-making:

```python
from src.agent.tools.orchestrator import ToolOrchestrator

orchestrator = ToolOrchestrator(registry)
result = orchestrator.run_tool_loop(
    user_prompt="Generate 3 momentum strategy proposals",
    regime_context=context,
    strategy_type="momentum",
)
```

## Available Tools

### Analysis Tools

- **get_regime_context**: Retrieve stored market regime context and prior strategy results
- **search_market_sentiment**: Search web for recent market sentiment and volatility signals (Tavily)
- **search_strategy_research**: Search for published strategy research and alpha ideas (Tavily)

### Proposal Tools

- **extract_parameter_recommendations**: Get available parameter space for a strategy

## Integration with HypothesizeNode

The hypothesize node can be extended to use the tool orchestrator:

```python
def hypothesize_node_with_tools(state: AgentState) -> AgentState:
    """Generate strategy proposals via tool-calling orchestrator."""
    from src.agent.tools import get_default_registry
    from src.agent.tools.orchestrator import ToolOrchestrator
    
    registry = get_default_registry()
    orchestrator = ToolOrchestrator(registry)
    
    result = orchestrator.run_tool_loop(
        user_prompt=f"Generate 5 {state['strategy_type']} proposals",
        regime_context=state["context"],
        strategy_type=state["strategy_type"],
    )
    
    # Parse proposals from result and add to state
    state["proposals"] = parse_proposals(result)
    return state
```

## Falsifiable Claims

Each tool call can include a `predicted_impact` field:

```json
{
  "tool_name": "search_market_sentiment",
  "input": {"query": "VIX spike volatility"},
  "predicted_impact": "High uncertainty will increase optimal window length by 20-30%"
}
```

These claims are tracked in memory and scored against actual realized outcomes for harness evaluation.

## Adding New Tools

1. Implement the callable function:
```python
def my_tool(param1: str, param2: int) -> Dict[str, Any]:
    return {"result": "..."}
```

2. Register it:
```python
registry.register(
    Tool(
        name="my_tool",
        description="What it does",
        input_schema={
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"},
            },
            "required": ["param1"],
        },
        callable_fn=my_tool,
        category="custom",
    )
)
```

## Dependencies

- `anthropic>=0.30` — Claude API for tool-use
- `tavily-python>=0.3` — Web search for sentiment and research tools (optional)

Set `TAVILY_API_KEY` environment variable to enable web search tools.
