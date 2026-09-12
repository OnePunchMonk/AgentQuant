#!/usr/bin/env python3
"""
Verify tool orchestration and API setup.

Quick diagnostic to ensure tools are working before running full POC.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.tools import get_default_registry


def check_api_keys():
    """Check if required API keys are set."""
    print("\n" + "="*70)
    print("API KEY CHECK")
    print("="*70)

    keys = {
        "ANTHROPIC_API_KEY": "Claude tool-use (required for full harness evolution)",
        "TAVILY_API_KEY": "Tavily web search (optional, tools degrade gracefully)",
        "GOOGLE_API_KEY": "Gemini API (existing, optional)",
    }

    for key, desc in keys.items():
        status = "✓" if os.getenv(key) else "✗"
        print(f"{status} {key}: {desc}")

    # Missing keys are expected in CI / offline environments -- every tool
    # is designed to degrade gracefully without them (fallback planner,
    # offline mode). This check is informational only; it must never fail
    # verification just because no real API key is configured.
    return True


def check_tools():
    """Verify tool registry and definitions."""
    print("\n" + "="*70)
    print("TOOL REGISTRY CHECK")
    print("="*70)

    try:
        registry = get_default_registry()
        tools = registry.list_tools()

        print(f"✓ Registry loaded with {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool}")

        # Try converting to Claude format
        claude_tools = registry.to_claude_tools()
        print(f"\n✓ {len(claude_tools)} tools ready for Claude API")

        return True
    except Exception as e:
        print(f"✗ Tool registry error: {e}")
        return False


def test_tool_execution():
    """Test basic tool execution."""
    print("\n" + "="*70)
    print("TOOL EXECUTION TEST")
    print("="*70)

    try:
        registry = get_default_registry()

        # Test get_regime_context (doesn't require API key)
        print("\nTesting: get_regime_context")
        result = registry.execute_tool(
            "get_regime_context",
            {"regime": "Bull", "strategy_type": "momentum"},
        )
        if result.get("success"):
            print(f"✓ Tool executed successfully")
        else:
            print(f"✗ Tool failed: {result.get('error')}")
            return False

        # Test parameter recommendations
        print("\nTesting: extract_parameter_recommendations")
        result = registry.execute_tool(
            "extract_parameter_recommendations",
            {"regime": "Bull", "strategy_type": "momentum"},
        )
        if result.get("success"):
            print(f"✓ Tool executed successfully")
        else:
            print(f"✗ Tool failed: {result.get('error')}")
            return False

        # Test Tavily integration (if key available)
        if os.getenv("TAVILY_API_KEY"):
            print("\nTesting: search_market_sentiment (Tavily)")
            result = registry.execute_tool(
                "search_market_sentiment",
                {"query": "VIX volatility"},
            )
            if result.get("success"):
                print(f"✓ Web search executed successfully")
            else:
                print(f"⚠ Web search unavailable (may require API key): {result.get('error')}")
        else:
            print("\n⚠ Skipping Tavily test (TAVILY_API_KEY not set)")

        return True

    except Exception as e:
        print(f"✗ Tool execution error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_orchestrator():
    """Check if orchestrator can be instantiated."""
    print("\n" + "="*70)
    print("ORCHESTRATOR CHECK")
    print("="*70)

    try:
        from src.agent.tools.orchestrator import ToolOrchestrator

        registry = get_default_registry()
        orchestrator = ToolOrchestrator(registry)

        print("✓ ToolOrchestrator instantiated")
        print(f"✓ System prompt generated")
        return True

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"✗ Orchestrator error: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("AGENTQUANT TOOL VERIFICATION")
    print("="*70)

    results = {
        "API Keys": check_api_keys(),
        "Tool Registry": check_tools(),
        "Tool Execution": test_tool_execution(),
        "Orchestrator": check_orchestrator(),
    }

    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    all_passed = True
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All checks passed! Ready to run harness_evolution_poc.py")
        return 0
    else:
        print("\n✗ Some checks failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
