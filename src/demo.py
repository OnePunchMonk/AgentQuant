"""Zero-configuration local demo of the search loop: `python -m src.demo` or `agentquant-demo`."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def main(out_dir: str = "results") -> int:
    trace = TraceRecorder()
    state = run_agent(make_demo_data(), strategy_type="momentum", asset="DEMO",
                      max_iterations=3, trace=trace)
    best = state.get("best_result") or {}
    output = {"mode": "zero_config_demo", "asset": "DEMO", "iterations": state.get("iteration", 0),
              "best": best, "trace_diagnostics": trace.diagnostics(),
              "run_log": state.get("run_log", [])}
    path = Path(out_dir) / "demo_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=str))
    print("AgentQuant demo complete")
    print(f"Iterations: {output['iterations']} | Best Sharpe: {best.get('sharpe', 0.0):.3f}")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
