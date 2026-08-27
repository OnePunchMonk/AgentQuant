"""
Example: Integrating Tool Orchestrator into HypothesizeNode

This shows how to refactor the hypothesize_node to use the tool
orchestrator instead of directly calling ProposalGenerator.

To use in production:
1. Move this logic into agent_graph.py
2. Update hypothesize_node() to call hypothesize_with_tools()
3. Add TAVILY_API_KEY to .env
"""

import logging
from typing import Dict, Any, Optional

from src.agent.agent_graph import AgentState
from src.agent.context_builder import RegimeContext
from src.agent.proposal_generator import Proposal
from src.agent.tools import get_default_registry
from src.agent.tools.orchestrator import ToolOrchestrator

logger = logging.getLogger(__name__)


def hypothesize_with_tools(state: AgentState) -> AgentState:
    """
    Generate strategy proposals via tool-calling orchestrator.

    This is an alternative to the current hypothesize_node that:
    1. Uses Claude's tool-use capability
    2. Allows tools to be edited/evolved (tool schemas become harness surface)
    3. Integrates web search (Tavily) for market context
    4. Attaches falsifiable claims to proposals
    """
    iteration = state.get("iteration", 0) + 1
    state["iteration"] = iteration
    logger.info("=== HYPOTHESIZE (iteration %d, with tools) ===", iteration)

    registry = get_default_registry()
    orchestrator = ToolOrchestrator(registry)

    context = state["context"]
    strategy_type = state.get("strategy_type", "momentum")

    # Run the tool-use loop
    result = orchestrator.run_tool_loop(
        user_prompt=f"""
Generate 5 high-confidence {strategy_type} strategy proposals.

For each proposal:
1. State which regime characteristic (volatility, trend, mean reversion) you're targeting
2. Explain the parameter choice in 1-2 sentences
3. Assign a confidence score (0.0-1.0)
4. Make a falsifiable claim about expected Sharpe improvement

Return JSON with fields: params, confidence, reasoning, falsifiable_claim
""",
        regime_context=context,
        strategy_type=strategy_type,
        max_turns=3,
    )

    # Parse proposals from Claude's response
    proposals = _parse_proposals_from_tool_result(result, strategy_type)

    state["proposals"] = proposals
    state["run_log"].append(
        f"Iteration {iteration}: Generated {len(proposals)} proposals via tools "
        f"(tools called: {len(result.get('tool_calls', []))})"
    )

    # Record tool calls in trace
    if state.get("trace"):
        for tool_call in result.get("tool_calls", []):
            from src.agent.trace import emit_trace
            emit_trace(
                state["trace"],
                "tool_call",
                f"Orchestrator called {tool_call['tool_name']}",
                tool_name=tool_call["tool_name"],
                input=tool_call.get("input"),
            )

    return state


def _parse_proposals_from_tool_result(
    result: Dict[str, Any], strategy_type: str
) -> list:
    """
    Parse Proposal objects from tool orchestrator result.

    Extracts JSON proposals from Claude's response and converts to Proposal dataclass.
    """
    proposals = []

    # For now, return empty list as a placeholder
    # In production, this would parse Claude's JSON response
    # and convert to Proposal objects
    logger.warning("Proposal parsing from tool result not yet implemented")

    return proposals


# ============================================================================
# Alternative: Hybrid Approach
# ============================================================================

def hypothesize_hybrid(state: AgentState) -> AgentState:
    """
    Hybrid approach: use tools selectively, fall back to ProposalGenerator.

    Use this if you want to gradually migrate to the tool-calling architecture
    while keeping the existing ProposalGenerator as a fallback.
    """
    iteration = state.get("iteration", 0) + 1
    state["iteration"] = iteration
    logger.info("=== HYPOTHESIZE (iteration %d, hybrid) ===", iteration)

    context = state["context"]
    strategy_type = state.get("strategy_type", "momentum")

    # Phase 1: Use tools for market context (if regime is uncertain)
    tool_context = ""
    if context.regime_confidence < 0.7:
        logger.info("Low regime confidence; gathering web context via tools")
        registry = get_default_registry()

        # Just use web search tool, no full orchestration
        search_result = registry.execute_tool(
            "search_market_sentiment",
            {"query": "market volatility VIX sentiment"},
        )
        if search_result.get("success"):
            tool_context = f"\n\nRecent market sentiment:\n{search_result.get('output', {})}"

    # Phase 2: Use existing ProposalGenerator (proven stable)
    from src.agent.proposal_generator import ProposalGenerator

    generator = ProposalGenerator()
    proposals = generator.generate(
        context=context,
        n_proposals=5,
        strategy_type=strategy_type,
        prior_results=state.get("all_results"),
    )

    state["proposals"] = proposals
    state["run_log"].append(
        f"Iteration {iteration}: Generated {len(proposals)} proposals "
        f"(tool context enrichment: {bool(tool_context)})"
    )

    return state
