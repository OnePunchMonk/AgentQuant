#!/usr/bin/env python3
"""Zero-configuration local demo for the self-improving search loop."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.agent_graph import run_agent
from src.agent.trace import TraceRecorder


def make_demo_data(n: int = 900, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    returns = 0.00025 + 0.009 * rng.standard_normal(n)
    returns[n // 2:] += 0.00015
    close = 100 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"Open": close * (1 - 0.002), "High": close * 1.005,
                          "Low": close * 0.995, "Close": close,
                          "Volume": rng.integers(1_000_000, 3_000_000, n)}, index=dates)
    return {"DEMO": frame}


def main() -> int:
    trace = TraceRecorder()
    state = run_agent(make_demo_data(), strategy_type="momentum", asset="DEMO",
                      max_iterations=3, trace=trace)
    best = state.get("best_result") or {}
    output = {"mode": "zero_config_demo", "asset": "DEMO", "iterations": state.get("iteration", 0),
              "best": best, "trace_diagnostics": trace.diagnostics(),
              "run_log": state.get("run_log", [])}
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "demo_run.json").write_text(json.dumps(output, indent=2, default=str))
    try:
        import matplotlib.pyplot as plt
        equity = best.get("equity_curve")
        if equity is not None:
            plt.figure(figsize=(10, 4)); equity.plot(); plt.title("AgentQuant zero-config demo equity curve")
            plt.tight_layout(); plt.savefig(out_dir / "demo_run.png", dpi=140); plt.close()
    except Exception as exc:
        output["chart_note"] = str(exc)
    print("AgentQuant zero-config demo complete")
    print(f"Iterations: {output['iterations']} | Best Sharpe: {best.get('sharpe', 0.0):.3f}")
    print(f"Report: {out_dir / 'demo_run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
