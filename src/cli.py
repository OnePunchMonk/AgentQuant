"""Command line interface for AgentQuant."""

import argparse
from typing import Any, Dict, List

import pandas as pd

from src.agent.agent_graph import run_agent
from src.agent.memory_layer import AgenticMemoryLayer
from src.agent.reporting import render_comparison_table, render_regime_card
from src.agent.swarm import run_swarm
from src.agent.trace import TraceRecorder
from src.data.ingest import fetch_ohlcv_data
from src.utils.config import config
from src.utils.logging import setup_logging


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No records found.")
        return
    df = pd.DataFrame(rows)
    try:
        print(df.to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        print(df.to_string(index=False))


def _memory_command(args: argparse.Namespace) -> int:
    layer = AgenticMemoryLayer()
    if args.patterns:
        patterns = layer.extract_patterns(
            regime=args.regime or "",
            strategy_types=[args.strategy] if args.strategy else None,
            limit=args.limit,
        )
        if not patterns:
            print("No memory patterns found.")
            return 0
        for pattern in patterns:
            print(f"- {pattern.to_sentence()}")
        return 0

    if args.export == "markdown":
        print(
            layer.render_markdown(
                regime=args.regime or "",
                strategy_type=args.strategy or "",
                limit=args.limit,
            )
        )
        return 0

    rows = []
    for row in layer.summary_rows(args.regime or "", args.strategy or "", limit=args.limit):
        rows.append({
            "Regime": row["regime"],
            "Strategy": row["strategy_type"],
            "Runs": row["runs"],
            "Avg Sharpe": row["avg_sharpe"],
            "Best Sharpe": row["best_sharpe"],
            "Avg Return": row["avg_return"],
            "Avg DD": row["avg_drawdown"],
            "Last Seen": str(row["last_seen"])[:10],
        })
    _print_table(rows)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    setup_logging(config.log_level)
    ohlcv_data = fetch_ohlcv_data(
        ticker=args.ticker if args.ticker else None,
        start_date=args.start,
        end_date=args.end,
    )
    if args.ticker and config.vix_ticker not in ohlcv_data:
        ohlcv_data.update(fetch_ohlcv_data(ticker=config.vix_ticker, start_date=args.start, end_date=args.end))

    assets = [args.ticker or config.reference_asset]
    trace = TraceRecorder(live=args.trace)

    if args.swarm:
        result = run_swarm(
            ohlcv_data=ohlcv_data,
            assets=assets,
            strategy_types=args.strategies or None,
        )
        print(result.summary())
        if result.full_ranking:
            normalized = [
                {
                    **row,
                    "sharpe": row.get("mean_sharpe", 0.0),
                    "total_return": row.get("mean_return", 0.0),
                    "max_drawdown": row.get("worst_drawdown", 0.0),
                    "generation_method": "swarm",
                }
                for row in result.full_ranking
            ]
            print()
            print(render_comparison_table(normalized))
        return 0

    strategy_type = args.strategy or (config.strategies[0].name if config.strategies else "momentum")
    state = run_agent(
        ohlcv_data=ohlcv_data,
        strategy_type=strategy_type,
        asset=assets[0],
        max_iterations=args.max_iterations,
        trace=trace if args.trace else None,
    )
    print(render_regime_card(state))
    print()
    print(render_comparison_table(state.get("results", [])))
    return 0


def _regime_card_command(args: argparse.Namespace) -> int:
    layer = AgenticMemoryLayer()
    runs = layer.recent_runs(limit=1)
    if not runs:
        print("No completed run found in strategy memory.")
        return 0
    latest = runs[0]
    card_state = {
        "regime_label": latest.regime,
        "regime_confidence": latest.confidence,
        "strategy_type": latest.strategy_type,
        "best_result": {
            "strategy_type": latest.strategy_type,
            "params": latest.params,
            "sharpe": latest.sharpe,
            "total_return": latest.total_return,
            "max_drawdown": latest.max_drawdown,
            "reasoning": latest.reasoning,
        },
    }
    print(render_regime_card(card_state))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentquant", description="AgentQuant research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the research agent")
    run_parser.add_argument("--ticker", default="", help="Single ticker to research, default SPY")
    run_parser.add_argument("--start", default=None, help="Start date, YYYY-MM-DD")
    run_parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD")
    run_parser.add_argument("--strategy", default="", help="Single-agent strategy type")
    run_parser.add_argument("--strategies", nargs="*", help="Swarm strategy specialists")
    run_parser.add_argument("--max-iterations", type=int, default=None)
    run_parser.add_argument("--trace", action="store_true", help="Show live hypothesis/backtest/reflection trace")
    run_parser.add_argument("--swarm", action="store_true", help="Run the multi-agent swarm")
    run_parser.set_defaults(func=_run_command)

    memory_parser = subparsers.add_parser("memory", help="Browse agentic strategy memory")
    memory_parser.add_argument("--regime", default="")
    memory_parser.add_argument("--strategy", default="")
    memory_parser.add_argument("--limit", type=int, default=25)
    memory_parser.add_argument("--patterns", action="store_true", help="Show learned strategy patterns")
    memory_parser.add_argument("--export", choices=["table", "markdown"], default="table")
    memory_parser.set_defaults(func=_memory_command)

    card_parser = subparsers.add_parser("regime-card", help="Render the latest stored regime card")
    card_parser.set_defaults(func=_regime_card_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
