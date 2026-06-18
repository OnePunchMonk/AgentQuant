"""Live trace events for the AgentQuant ReAct loop."""

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


def emit_trace(trace: Optional[TraceRecorder], stage: str, message: str, **payload: Any) -> None:
    if trace is not None:
        trace.emit(stage, message, **payload)
