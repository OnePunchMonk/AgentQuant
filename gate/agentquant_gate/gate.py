"""Conservative paired bootstrap gate; Sharpe requires a future validated DSR."""
from dataclasses import dataclass, asdict
import numpy as np
from .ledger import Ledger

@dataclass(frozen=True)
class GateDecision:
    outcome: str
    raw_delta: float | None
    deflated_delta: float | None
    p_corrected: float | None
    n_trials_in_family: int
    k_seeds: int
    method: str
    reason: str

class Gate:
    def __init__(self, ledger: Ledger, *, family, metric, direction="max", n_bootstrap=9999, random_seed=0):
        if direction not in {"max", "min"} or n_bootstrap < 999:
            raise ValueError("invalid direction or bootstrap budget")
        self.ledger, self.family, self.metric, self.direction = ledger, family, metric, direction
        self.n_bootstrap, self.random_seed = n_bootstrap, random_seed

    def evaluate(self, candidate, incumbent, *, protected_candidate=None,
                 protected_incumbent=None, max_protected_regression=0.25):
        family = [e for e in self.ledger.entries() if e.get("family") == self.family and e["type"] in {"trial", "error"}]
        trials = [e for e in family if e["type"] == "trial"]
        lookup = {e["trial_id"]: e for e in trials}
        if candidate not in lookup or incumbent not in lookup:
            raise KeyError("both candidates must exist in the ledger")
        c, i = lookup[candidate], lookup[incumbent]
        if any(e.get("metric") != self.metric or e.get("direction") != self.direction for e in (c, i)):
            raise ValueError("metric/direction mismatch")
        sign = 1 if self.direction == "max" else -1
        cs, ins = np.asarray(c["scores"], float), np.asarray(i["scores"], float)
        k = min(len(cs), len(ins))
        delta = float(sign * (cs.mean() - ins.mean()))
        outcome, p, method, reason = "insufficient_evidence", None, "paired-maxT-bootstrap", "insufficient paired evidence"
        if k >= 3 and len(cs) == len(ins):
            if "sharpe" in self.metric.lower():
                method, reason = "dsr-not-implemented", "Sharpe promotion requires validated DSR and return series"
            else:
                others = [e for e in trials if e["trial_id"] != incumbent and e.get("metric") == self.metric and e.get("direction") == self.direction]
                if any(len(e["scores"]) != k for e in others):
                    reason = "incompatible family seed coverage"
                elif others:
                    diffs = np.stack([sign * (np.asarray(e["scores"], float) - ins) for e in others])
                    means = diffs.mean(axis=1)
                    se = diffs.std(axis=1, ddof=1) / np.sqrt(k)
                    ix = next((j for j, e in enumerate(others) if e["trial_id"] == candidate), None)
                    if ix is not None and np.all(np.isfinite(se)) and np.all(se > 1e-12):
                        observed = means[ix] / se[ix]
                        centered = diffs - means[:, None]
                        rng = np.random.default_rng(self.random_seed)
                        exceed = 0
                        for _ in range(self.n_bootstrap):
                            sample = centered[:, rng.integers(0, k, size=k)]
                            sample_se = sample.std(axis=1, ddof=1) / np.sqrt(k)
                            if np.any(sample_se <= 1e-12) or np.max(sample.mean(axis=1) / sample_se) >= observed:
                                exceed += 1
                        p = (exceed + 1) / (self.n_bootstrap + 1)
                        if protected_candidate is None or protected_incumbent is None:
                            reason = "protected episode scores missing"
                        elif sign * (protected_candidate - protected_incumbent) < -abs(max_protected_regression):
                            outcome, reason = "reject", "protected episode regression"
                        elif p < 0.05 and delta > 0:
                            outcome, reason = "promote", "search-corrected bootstrap passed"
                        else:
                            outcome, reason = "reject", "search-corrected significance not met"
        decision = GateDecision(outcome, delta, None, p, len(family), k, method, reason)
        self.ledger.append("decision", family=self.family, metric=self.metric,
                           candidate_id=candidate, incumbent_id=incumbent, **asdict(decision))
        return decision
