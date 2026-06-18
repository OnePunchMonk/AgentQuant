from src.agent.reporting import render_comparison_table, render_regime_card, verdict_for_metrics
from src.cli import build_parser


def test_verdict_rejects_weak_sharpe():
    assert verdict_for_metrics({"sharpe": -0.1, "max_drawdown": 0.05}) == "rejected"


def test_regime_card_and_comparison_table_render():
    state = {
        "regime_label": "LowVol-Bull",
        "regime_confidence": 0.8,
        "strategy_type": "momentum",
        "best_result": {
            "strategy_type": "momentum",
            "params": {"fast_window": 20, "slow_window": 100},
            "sharpe": 0.9,
            "calmar": 1.2,
            "sortino": 1.4,
            "max_drawdown": 0.08,
            "bootstrap_sharpe_p5": 0.2,
            "reasoning": "Trend strength supports longer momentum windows.",
        },
    }
    card = render_regime_card(state)
    table = render_comparison_table([state["best_result"]])

    assert "AgentQuant Regime Card" in card
    assert "LowVol-Bull" in card
    assert "Boot p5" in table


def test_cli_parses_memory_patterns():
    parser = build_parser()
    args = parser.parse_args(["memory", "--patterns", "--regime", "LowVol-Bull"])
    assert args.command == "memory"
    assert args.patterns is True
