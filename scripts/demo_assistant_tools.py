"""Run a self-contained, offline demo of the assistant sidecar runner."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from src.agent.assistant_tool_runner import AssistantToolRunner, AssistantToolRunRequest
from src.agent.sidecar_tools import make_idea_sidecar_registry


def main() -> None:
    runner = AssistantToolRunner(make_idea_sidecar_registry())
    request = AssistantToolRunRequest(
        request_id="demo-2024-01-10",
        intent="failure_lookup",
        decision_time="2024-01-10",
        arguments={"failures": [
            {"failure_id": "visible", "timestamp": "2024-01-08", "reason": "drawdown"},
            {"failure_id": "future", "timestamp": "2024-01-12", "reason": "future record"},
        ]},
    )
    print(json.dumps(runner.run(request).to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
