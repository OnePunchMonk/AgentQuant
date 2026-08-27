"""
Tool Orchestrator — Claude tool-use loop for agent decision-making
==================================================================

Integrates Claude's native tool_use with the tool registry.
Handles tool choice, execution, and feedback loops.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """Orchestrates Claude tool-use loops."""

    def __init__(self, registry):
        self.registry = registry

    def build_system_prompt(self) -> str:
        """Build system prompt explaining available tools."""
        tools_list = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.registry.tools.values()
        )
        return f"""You are a quantitative research agent that makes decisions through tool calls.

Available tools:
{tools_list}

When generating strategy proposals:
1. First call get_regime_context to understand the current market state
2. If uncertain about regime conditions, optionally call search_market_sentiment
3. Call extract_parameter_recommendations to see the parameter space
4. Generate parameter proposals by reasoning about the regime and constraints
5. Attach a falsifiable claim: what you predict will improve Sharpe ratio

Always be explicit about your reasoning before calling tools.
"""

    def run_tool_loop(
        self,
        user_prompt: str,
        regime_context,
        strategy_type: str,
        max_turns: int = 3,
    ) -> Dict[str, Any]:
        """
        Run a Claude tool-use loop for proposal generation.

        Args:
            user_prompt: High-level task for Claude (e.g., "Generate 3 momentum strategy proposals")
            regime_context: Current RegimeContext
            strategy_type: Strategy type
            max_turns: Max number of tool calls

        Returns:
            {
                "proposals": [...],
                "tool_calls": [...],
                "reasoning": "...",
            }
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            logger.warning("Anthropic SDK not installed; falling back to default proposal generation")
            return {"proposals": [], "tool_calls": [], "error": "Anthropic SDK not available"}

        client = Anthropic()
        messages = []
        tool_calls_made = []

        # Build initial prompt
        system_prompt = self.build_system_prompt()
        initial_message = f"""
Current regime: {regime_context.regime_label} (confidence: {regime_context.regime_confidence:.0%})
Strategy type: {strategy_type}

{user_prompt}

Use available tools to gather context, then generate proposals with falsifiable claims.
"""

        messages.append({"role": "user", "content": initial_message})

        # Loop for up to max_turns
        for turn in range(max_turns):
            logger.info(f"Tool loop turn {turn + 1}/{max_turns}")

            # Call Claude with tools
            response = client.messages.create(
                model="claude-opus-5",  # Can be made configurable
                max_tokens=2000,
                system=system_prompt,
                tools=self.registry.to_claude_tools(),
                messages=messages,
            )

            # Check if Claude wants to use tools or just respond
            if response.stop_reason == "end_turn":
                # Claude is done; extract proposals from response
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text
                logger.info(f"Claude completed turn {turn + 1} without tool calls")
                return {
                    "proposals": [],
                    "tool_calls": tool_calls_made,
                    "reasoning": final_text,
                    "stop_reason": "end_turn",
                }

            if response.stop_reason != "tool_use":
                logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                break

            # Process tool calls
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            # Add Claude's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for tool_use in tool_use_blocks:
                logger.info(f"Claude called tool: {tool_use.name}")
                logger.debug(f"  Input: {tool_use.input}")

                # Execute the tool
                result = self.registry.execute_tool(tool_use.name, tool_use.input)
                tool_calls_made.append(
                    {
                        "tool_name": tool_use.name,
                        "input": tool_use.input,
                        "result": result,
                    }
                )

                # Format result for Claude
                tool_result_text = json.dumps(result, indent=2, default=str)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result_text,
                    }
                )

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})

        return {
            "proposals": [],
            "tool_calls": tool_calls_made,
            "error": "Max tool loop turns exceeded",
        }
