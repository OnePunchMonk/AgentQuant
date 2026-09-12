"""
Tool Registry — Tool schemas, execution, and composition
=========================================================

Each tool is defined by:
- name: unique identifier
- description: what it does (for Claude to understand)
- input_schema: JSON Schema for parameters
- callable: the implementation
- falsifiable_claim: predicted impact (for harness evaluation)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A single tool invocation."""
    name: str
    input: Dict[str, Any]
    predicted_impact: str = ""  # Falsifiable claim


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None


class Tool:
    """A single executable tool in the registry."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        callable_fn: Callable,
        category: str = "utility",  # For organization
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.callable_fn = callable_fn
        self.category = category

    def to_claude_tool(self) -> Dict[str, Any]:
        """Convert to Claude API tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.input_schema.get("properties", {}),
                "required": self.input_schema.get("required", []),
            },
        }



class ToolRegistry:
    """Registry of available tools for the agent."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def without_tool(self, name: str) -> "ToolRegistry":
        """Return a copy of this registry with one tool removed (e.g. to
        enforce a harness config that disables web search)."""
        clone = ToolRegistry()
        clone.tools = {k: v for k, v in self.tools.items() if k != name}
        return clone

    def to_claude_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to Claude API format."""
        return [tool.to_claude_tool() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call synchronously."""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "tool_name": tool_name,
                "success": False,
                "output": None,
                "error": f"Tool {tool_name} not found",
            }
        try:
            output = tool.callable_fn(**input_dict)
            return {
                "tool_name": tool_name,
                "success": True,
                "output": output,
            }
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return {
                "tool_name": tool_name,
                "success": False,
                "output": None,
                "error": str(e),
            }


# ============================================================================
# Tool Implementations
# ============================================================================


def _get_regime_context(regime: str, strategy_type: str) -> str:
    """Retrieve market regime context from memory."""
    from src.agent.strategy_memory import StrategyMemory

    memory = StrategyMemory()
    return memory.to_prompt_context(regime, strategy_type)


def _search_market_sentiment(query: str, time_range: str = "week") -> Dict[str, Any]:
    """Search for recent market sentiment and news via Tavily."""
    try:
        from tavily import TavilyClient

        client = TavilyClient()
        results = client.search(query, include_raw_content=False, max_results=5)
        return {
            "query": query,
            "time_range": time_range,
            "results": results.get("results", []),
            "source": "tavily",
        }
    except ImportError:
        logger.warning("Tavily not installed; skipping market sentiment search")
        return {
            "query": query,
            "results": [],
            "source": "tavily",
            "error": "Tavily client not available",
        }
    except Exception as e:
        logger.error(f"Market sentiment search failed: {e}")
        return {
            "query": query,
            "results": [],
            "source": "tavily",
            "error": str(e),
        }


def _search_strategy_research(strategy_type: str, topic: str = "") -> Dict[str, Any]:
    """Search for published strategy research and alpha ideas."""
    try:
        from tavily import TavilyClient

        client = TavilyClient()
        query = f"{strategy_type} strategy {topic}".strip()
        results = client.search(query, include_raw_content=False, max_results=5)
        return {
            "strategy_type": strategy_type,
            "topic": topic,
            "results": results.get("results", []),
            "source": "tavily",
        }
    except ImportError:
        logger.warning("Tavily not installed; skipping strategy research")
        return {
            "strategy_type": strategy_type,
            "results": [],
            "source": "tavily",
            "error": "Tavily client not available",
        }
    except Exception as e:
        logger.error(f"Strategy research search failed: {e}")
        return {
            "strategy_type": strategy_type,
            "results": [],
            "source": "tavily",
            "error": str(e),
        }


def _extract_parameter_recommendations(
    regime: str, strategy_type: str, constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generate parameter recommendations from grid and prior results."""
    from src.agent.parameter_grid import ParameterGrid

    grid = ParameterGrid()
    strategy_params = grid.get_grid(strategy_type)

    return {
        "regime": regime,
        "strategy_type": strategy_type,
        "parameter_space": strategy_params,
        "recommendation_method": "grid",
    }


def _run_benchmark_tool(strategy_type: str = "", regime: str = "") -> Dict[str, Any]:
    """Run quality assessment benchmark (agent-callable eval)."""
    from src.agent.tools.evals import run_benchmark_to_assess_quality_tool

    return run_benchmark_to_assess_quality_tool(strategy_type=strategy_type, regime=regime)


def _stress_test_tool(ohlcv_data, assets, strategy_name, params):
    from src.backtest.stress_test import stress_test
    return stress_test(ohlcv_data, assets, strategy_name, params)


# ============================================================================
# Default Registry Setup
# ============================================================================


def get_default_registry() -> ToolRegistry:
    """Create and populate the default tool registry."""
    registry = ToolRegistry()

    # Market Analysis Tools
    registry.register(
        Tool(
            name="get_regime_context",
            description="Retrieve stored market regime context and prior strategy results for the current regime.",
            input_schema={
                "properties": {
                    "regime": {"type": "string", "description": "Market regime label (e.g., 'Bull', 'Bear', 'Crisis')"},
                    "strategy_type": {"type": "string", "description": "Strategy type to retrieve context for"},
                },
                "required": ["regime", "strategy_type"],
            },
            callable_fn=_get_regime_context,
            category="analysis",
        )
    )

    # Tavily Web Search Tools
    registry.register(
        Tool(
            name="search_market_sentiment",
            description="Search the web for recent market sentiment, volatility indices, macro news. Use in high-uncertainty regimes.",
            input_schema={
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g., 'VIX spike volatility')"},
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "week", "month"],
                        "description": "Time range for recency",
                    },
                },
                "required": ["query"],
            },
            callable_fn=_search_market_sentiment,
            category="web_search",
        )
    )

    registry.register(
        Tool(
            name="search_strategy_research",
            description="Search academic and industry research for strategy ideas and parameter insights.",
            input_schema={
                "properties": {
                    "strategy_type": {"type": "string", "description": "Strategy name (e.g., 'momentum', 'mean_reversion')"},
                    "topic": {"type": "string", "description": "Specific topic or constraint (e.g., 'volatility regimes')"},
                },
                "required": ["strategy_type"],
            },
            callable_fn=_search_strategy_research,
            category="web_search",
        )
    )

    # Parameter Generation Tools
    registry.register(
        Tool(
            name="extract_parameter_recommendations",
            description="Get available parameter space and recommendations for a strategy in a given regime.",
            input_schema={
                "properties": {
                    "regime": {"type": "string", "description": "Market regime"},
                    "strategy_type": {"type": "string", "description": "Strategy type"},
                },
                "required": ["regime", "strategy_type"],
            },
            callable_fn=_extract_parameter_recommendations,
            category="proposal",
        )
    )

    # Evaluation Tools
    registry.register(
        Tool(
            name="run_benchmark_to_assess_quality",
            description="Run quality assessment benchmark on recent harness performance. Measures Sharpe, stability, generalization gap, and tool accuracy. Returns score, passed flag, and recommendations.",
            input_schema={
                "properties": {
                    "strategy_type": {
                        "type": "string",
                        "description": "Strategy being evaluated (e.g., 'momentum')",
                    },
                    "regime": {
                        "type": "string",
                        "description": "Market regime to assess within",
                    },
                },
                "required": [],
            },
            callable_fn=_run_benchmark_tool,
            category="evaluation",
        )
    )

    registry.register(Tool(
        name="stress_test_strategy",
        description="Run counterfactual regime, volatility, outlier-removal, and trend-reversal stress tests.",
        input_schema={"properties": {"ohlcv_data": {"type": "object"}, "assets": {"type": "array"},
                                      "strategy_name": {"type": "string"}, "params": {"type": "object"}},
                       "required": ["ohlcv_data", "assets", "strategy_name", "params"]},
        callable_fn=_stress_test_tool, category="evaluation"))

    return registry
