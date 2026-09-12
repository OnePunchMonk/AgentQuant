"""Live trace events for the AgentQuant ReAct loop."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    """One visible step in the agent loop."""

    stage: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().strftime("%H:%M:%S"))


class TraceRecorder:
    """Collect trace events and optionally print them live."""

    def __init__(self, live: bool = False):
        self.live = live
        self.events: List[TraceEvent] = []

    def emit(self, stage: str, message: str, **payload: Any) -> None:
        event = TraceEvent(stage=stage, message=message, payload=payload)
        self.events.append(event)
        if self.live:
            self._print_event(event)

    def _print_event(self, event: TraceEvent) -> None:
        try:
            from rich.console import Console
        except Exception:
            print(f"[{event.timestamp}] {event.stage}: {event.message}")
            return

        console = Console()
        color = {
            "analyze": "cyan",
            "hypothesize": "magenta",
            "backtest": "yellow",
            "reflect": "blue",
            "store": "green",
            "swarm": "green",
        }.get(event.stage, "white")
        console.print(f"[dim]{event.timestamp}[/dim] [{color}]{event.stage.upper()}[/{color}] {event.message}")

    def diagnostics(self) -> Dict[str, Any]:
        """Return JSON-serialisable harness diagnostics for one run."""
        proposals = [e for e in self.events if e.stage == "hypothesize"]
        backs = [e for e in self.events if e.stage == "backtest"]
        methods = Counter(m for e in proposals for m in e.payload.get("methods", []))
        accepted = sum(bool(e.payload.get("accepted")) for e in self.events if e.stage == "reflect")
        improved = 0
        previous = None
        for e in backs:
            best = (e.payload.get("best") or {}).get("sharpe")
            if best is not None and (previous is None or best > previous):
                improved += 1
            if best is not None:
                previous = best
        return {
            "event_count": len(self.events),
            "node_counts": dict(Counter(e.stage for e in self.events)),
            "proposal_method_counts": dict(methods),
            "backtest_rounds": len(backs),
            "improving_backtest_rounds": improved,
            "accepted_reflections": accepted,
            "validator_rejections": sum(e.payload.get("validator_rejections", 0) for e in self.events),
        }

    def diagnostics_json(self) -> str:
        import json
        return json.dumps(self.diagnostics(), sort_keys=True)


def emit_trace(trace: Optional[TraceRecorder], stage: str, message: str, **payload: Any) -> None:
    if trace is not None:
        trace.emit(stage, message, **payload)
