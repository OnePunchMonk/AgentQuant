"""Promotion gate: an append-only, hash-chained trial ledger plus a search-aware decision.

`Ledger` records every trial (tamper-evident). `Gate` promotes a candidate only if it beats the
baseline after accounting for how many trials were tried (paired max-T bootstrap; optional deflated
Sharpe hurdle). `prospective` checks promoted candidates on data that arrives after promotion.
Used by `src.agent.policy_mutation`. Diagram: agentquant-artifacts/gate/architecture.svg.
"""

from .gate import Gate, GateDecision
from .ledger import Ledger, LedgerIntegrityError

__all__ = ["Ledger", "LedgerIntegrityError", "Gate", "GateDecision"]
