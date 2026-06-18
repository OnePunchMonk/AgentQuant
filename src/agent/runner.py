"""
Agent Runner — Main Entry Point
=================================

Uses the agent graph to run the full analyze → hypothesize →
backtest → reflect → store loop.
"""

import logging

import pandas as pd
from dotenv import load_dotenv

from src.agent.agent_graph import run_agent
from src.agent.reporting import render_comparison_table, render_regime_card
from src.data.ingest import fetch_ohlcv_data
from src.utils.config import config
from src.utils.logging import setup_logging

# Setup logging
setup_logging(config.log_level)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the agent."""
    logger.info("=" * 60)
    logger.info("AgentQuant — Starting Agent Run")
    logger.info("=" * 60)

    load_dotenv()

    # Step 1: Ingest data
    logger.info("Step 1: Ingesting market data...")
    ohlcv_data = fetch_ohlcv_data()

    ref_asset = config.reference_asset
    if ref_asset not in ohlcv_data:
        logger.error("Reference asset '%s' data not found. Aborting.", ref_asset)
        return

    logger.info("Loaded data for: %s", list(ohlcv_data.keys()))

    # Step 2: Run the agent loop
    logger.info("Step 2: Running agent loop...")
    strategy_type = config.strategies[0].name if config.strategies else "momentum"

    state = run_agent(
        ohlcv_data=ohlcv_data,
        strategy_type=strategy_type,
        asset=ref_asset,
    )

    # Step 3: Report results
    logger.info("=" * 60)
    logger.info("Agent Run Summary")
    logger.info("=" * 60)

    for line in state.get("run_log", []):
        logger.info("  %s", line)

    best = state.get("best_result")
    if best:
        logger.info("")
        logger.info("BEST RESULT:")
        logger.info("  Strategy: %s", strategy_type)
        logger.info("  Params: %s", best.get("params"))
        logger.info("  Sharpe Ratio: %.4f", best.get("sharpe", 0))
        logger.info("  Total Return: %.2f%%", best.get("total_return", 0) * 100)
        logger.info("  Max Drawdown: %.2f%%", best.get("max_drawdown", 0) * 100)
        logger.info("  Num Trades: %s", best.get("num_trades", "N/A"))
        logger.info("  Generation Method: %s", best.get("generation_method", ""))
        logger.info("  Iterations Used: %d/%d", state.get("iteration", 0), state.get("max_iterations", 3))
    else:
        logger.warning("No valid result produced by the agent.")

    # Print results table if we have multiple results
    results = state.get("results", [])
    if results:
        df = pd.DataFrame(results)
        cols = [c for c in ["params", "sharpe", "total_return", "max_drawdown", "generation_method"] if c in df.columns]
        if cols:
            logger.info("\nAll Results:")
            try:
                print(df[cols].to_markdown(floatfmt=".4f", index=False))
            except Exception:
                print(df[cols].to_string(index=False))

        print()
        print(render_regime_card(state))
        print()
        print(render_comparison_table(results))

    logger.info("Agent run finished.")


if __name__ == "__main__":
    main()
