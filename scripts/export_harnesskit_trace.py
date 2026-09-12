#!/usr/bin/env python3
"""Run the local demo and export its trace in harnesskit format."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.demo_run import make_demo_data
from src.agent.agent_graph import run_agent
from src.agent.trace import TraceRecorder
from src.integrations.harnesskit_bridge import to_trajectory_dict


def main() -> int:
    trace = TraceRecorder()
    run_agent(make_demo_data(), strategy_type="momentum", asset="DEMO", max_iterations=3, trace=trace)
    trajectory = to_trajectory_dict(trace)
    path = Path("results/demo_run.trajectory.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"bounded-agent-loop": trajectory}, indent=2))
    print(f"Saved Harnesskit trajectory: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
