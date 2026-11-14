"""Small typed contracts for replaceable agent-loop capabilities.

These protocols describe boundaries; they do not define orchestration or a
plugin loading system. Observers receive immutable snapshots and cannot return
decisions to the run loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, TypeVar


@dataclass(frozen=True)
class RunBudget:
    """Limits available to a capability during one run."""

    max_proposals: int = 5
    max_backtests: int = 100
    deadline_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_proposals < 0 or self.max_backtests < 0:
            raise ValueError("run budget limits must be non-negative")
        if self.deadline_seconds is not None and self.deadline_seconds < 0:
            raise ValueError("deadline_seconds must be non-negative")


@dataclass(frozen=True)
class RunVisibility:
    """Data visibility grants, expressed as explicit capabilities."""

    allow_market_data: bool = True
    allow_prior_results: bool = True
    allow_training_data: bool = True
    allow_holdout_data: bool = False
    allow_external_io: bool = False


@dataclass(frozen=True)
class RunContext:
    """Immutable run identity and limits passed to capability implementations."""

    run_id: str
    strategy: str
    budget: RunBudget = field(default_factory=RunBudget)
    visibility: RunVisibility = field(default_factory=RunVisibility)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.strategy:
            raise ValueError("run_id and strategy are required")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ProposalRequest:
    context: RunContext
    count: int
    regime: Any = None
    prior_results: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        if self.count < 0 or self.count > self.context.budget.max_proposals:
            raise ValueError("proposal count exceeds the run budget")


@dataclass(frozen=True)
class AnalysisRequest:
    context: RunContext
    evidence: Any


@dataclass(frozen=True)
class AllocationRequest:
    context: RunContext
    candidates: Sequence[Any]
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity < 0 or self.capacity > self.context.budget.max_backtests:
            raise ValueError("allocation capacity exceeds the run budget")


@dataclass(frozen=True)
class Observation:
    """Immutable event snapshot. Observer callbacks are telemetry only."""

    run_id: str
    event: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


class ProposalProvider(Protocol):
    def propose(self, request: ProposalRequest) -> Sequence[Any]: ...


class Allocator(Protocol):
    def allocate(self, request: AllocationRequest) -> Sequence[Any]: ...


class AnalysisProvider(Protocol):
    def analyze(self, request: AnalysisRequest) -> Any: ...


class Observer(Protocol):
    """Read-only sink: observe returns None and receives no mutable run state."""

    def observe(self, observation: Observation) -> None: ...


class PolicyOptimizer(Protocol):
    def optimize(self, context: RunContext, evaluations: Sequence[Any]) -> Any: ...


P = TypeVar("P", bound=ProposalProvider)
A = TypeVar("A", bound=Allocator)
N = TypeVar("N", bound=AnalysisProvider)
ObsT = TypeVar("ObsT", bound=Observer)
E = TypeVar("E", bound=PolicyOptimizer)


class CapabilityRegistry:
    """Explicit, in-memory registry with one implementation per capability name."""

    def __init__(self) -> None:
        self._proposals: dict[str, ProposalProvider] = {}
        self._allocators: dict[str, Allocator] = {}
        self._analyses: dict[str, AnalysisProvider] = {}
        self._observers: dict[str, Observer] = {}
        self._optimizers: dict[str, PolicyOptimizer] = {}

    @staticmethod
    def _put(store: dict[str, Any], name: str, value: Any) -> None:
        if not name:
            raise ValueError("capability name is required")
        if name in store:
            raise ValueError(f"capability already registered: {name}")
        store[name] = value

    @staticmethod
    def _get(store: dict[str, Any], name: str) -> Any:
        try:
            return store[name]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {name}") from exc

    def register_proposal_provider(self, name: str, value: P) -> P:
        self._put(self._proposals, name, value)
        return value

    def proposal_provider(self, name: str) -> ProposalProvider:
        return self._get(self._proposals, name)

    def register_allocator(self, name: str, value: A) -> A:
        self._put(self._allocators, name, value)
        return value

    def allocator(self, name: str) -> Allocator:
        return self._get(self._allocators, name)

    def register_analysis_provider(self, name: str, value: N) -> N:
        self._put(self._analyses, name, value)
        return value

    def analysis_provider(self, name: str) -> AnalysisProvider:
        return self._get(self._analyses, name)

    def register_observer(self, name: str, value: ObsT) -> ObsT:
        self._put(self._observers, name, value)
        return value

    def observer(self, name: str) -> Observer:
        return self._get(self._observers, name)

    def register_policy_optimizer(self, name: str, value: E) -> E:
        self._put(self._optimizers, name, value)
        return value

    def policy_optimizer(self, name: str) -> PolicyOptimizer:
        return self._get(self._optimizers, name)


__all__ = [
    "AllocationRequest", "Allocator", "AnalysisProvider", "AnalysisRequest",
    "CapabilityRegistry", "Observation", "Observer", "PolicyOptimizer",
    "ProposalProvider", "ProposalRequest", "RunBudget", "RunContext", "RunVisibility",
]
