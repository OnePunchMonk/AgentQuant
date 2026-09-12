"""
Tool Registry — Composable tools for agentic decision-making
=============================================================

Tools are the editable surface for harness evolution.
Each tool has a JSON schema, callable implementation, and falsifiable claims.
"""

from src.agent.tools.registry import (
    ToolCall,
    ToolRegistry,
    ToolResult,
    get_default_registry,
)

__all__ = ["ToolRegistry", "ToolCall", "ToolResult", "get_default_registry"]
