"""Console reporting helpers for agent runs."""

from typing import Any, Dict, Iterable, List, Optional

from src.utils.config import config


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def verdict_for_metrics(metrics: Dict[str, Any], min_sharpe: Optional[float] = None) -> str:
    """Return a concise pass/reject verdict for a candidate."""
    threshold = config.agent.min_acceptable_sharpe if min_sharpe is None else min_sharpe
    sharpe = float(metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0)) or 0.0)
    bootstrap_p5 = float(metrics.get("bootstrap_sharpe_p5", sharpe) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    if sharpe >= threshold and bootstrap_p5 >= 0 and max_drawdown <= config.agent.risk.max_drawdown:
        return "passed"
    if sharpe >= threshold:
        return "watch"
    return "rejected"


def result_rows(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize agent result dicts for display."""
    rows = []
    for idx, result in enumerate(results, start=1):
        metrics = {
            "sharpe": result.get("sharpe", result.get("sharpe_ratio", 0.0)),
            "calmar": result.get("calmar", 0.0),
            "sortino": result.get("sortino", 0.0),
            "max_drawdown": result.get("max_drawdown", 0.0),
            "bootstrap_sharpe_p5": result.get("bootstrap_sharpe_p5", 0.0),
        }
        rows.append({
            "Rank": idx,
            "Verdict": verdict_for_metrics(metrics),
            "Strategy": result.get("strategy_type", ""),
            "Params": result.get("params", {}),
            "Sharpe": round(float(metrics["sharpe"] or 0.0), 3),
            "Calmar": round(float(metrics["calmar"] or 0.0), 3),
            "Sortino": round(float(metrics["sortino"] or 0.0), 3),
            "Max DD": _fmt_pct(float(metrics["max_drawdown"] or 0.0)),
            "Boot p5": round(float(metrics["bootstrap_sharpe_p5"] or 0.0), 3),
            "Return": _fmt_pct(float(result.get("total_return", 0.0) or 0.0)),
            "Method": result.get("generation_method", ""),
        })
    return rows


def render_comparison_table(results: Iterable[Dict[str, Any]]) -> str:
    """Render a screenshot-friendly markdown table."""
    rows = result_rows(results)
    if not rows:
        return "No candidate backtest results."
    headers = ["Rank", "Verdict", "Strategy", "Sharpe", "Calmar", "Sortino", "Max DD", "Boot p5", "Return", "Method", "Params"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| {Rank} | {Verdict} | {Strategy} | {Sharpe:.3f} | {Calmar:.3f} | "
            "{Sortino:.3f} | {Max DD} | {Boot p5:.3f} | {Return} | {Method} | `{Params}` |".format(**row)
        )
    return "\n".join(lines)


def render_regime_card(state: Dict[str, Any]) -> str:
    """Render a one-page summary of the completed agent run."""
    context = state.get("context") or state.get("regime_context")
    best = state.get("best_result") or {}
    regime_label = getattr(context, "regime_label", state.get("regime_label", "Unknown"))
    confidence = getattr(context, "regime_confidence", state.get("regime_confidence", 0.0))
    strategy = best.get("strategy_type", state.get("strategy_type", ""))
    params = best.get("params", {})
    reasoning = best.get("reasoning", "") or best.get("thesis", "") or "No explicit reasoning captured."

    lines = [
        "# AgentQuant Regime Card",
        "",
        f"Regime: {regime_label} ({confidence:.0%} confidence)",
        f"Top strategy: {strategy} {params}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Verdict | {verdict_for_metrics(best)} |",
        f"| Sharpe | {float(best.get('sharpe', best.get('mean_sharpe', 0.0)) or 0.0):.3f} |",
        f"| Calmar | {float(best.get('calmar', 0.0) or 0.0):.3f} |",
        f"| Sortino | {float(best.get('sortino', 0.0) or 0.0):.3f} |",
        f"| Max drawdown | {_fmt_pct(float(best.get('max_drawdown', 0.0) or 0.0))} |",
        f"| Bootstrap Sharpe p5 | {float(best.get('bootstrap_sharpe_p5', 0.0) or 0.0):.3f} |",
        "",
        "Why this fits:",
        reasoning,
    ]
    return "\n".join(lines)
