"""Conservative paired bootstrap gate. Sharpe metrics are promoted only with a recorded sample size and a deflated-Sharpe hurdle."""
from dataclasses import asdict, dataclass

import numpy as np

from .dsr import deflated_sharpe_ratio, expected_max_sharpe
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
    def __init__(self, ledger: Ledger, *, family, metric, direction="max", n_bootstrap=9999, random_seed=0, dsr_threshold=0.95):
        if direction not in {"max", "min"} or n_bootstrap < 999:
            raise ValueError("invalid direction or bootstrap budget")
        self.ledger, self.family, self.metric, self.direction = ledger, family, metric, direction
        self.n_bootstrap, self.random_seed = n_bootstrap, random_seed
        # Minimum deflated-Sharpe probability for a Sharpe-valued metric. None disables the
        # hurdle (bootstrap only); it exists so the two can be benchmarked, not as a default.
        self.dsr_threshold = dsr_threshold

    @staticmethod
    def _dsr_check(cand_scores, trials, n_obs, threshold=0.95, periods_per_year=252.0):
        """Deflated-Sharpe hurdle for a Sharpe-valued metric.

        Scores are annualised Sharpes over ``n_obs`` bars each. The candidate's mean
        Sharpe is deflated by the expected maximum of the family's trials (variance
        taken across the trials' mean Sharpes) and must clear it with probability
        ``threshold``. Returns (ok, annualised excess over the hurdle, reason).
        """
        means = np.array([np.mean(e["scores"]) for e in trials], float) / np.sqrt(periods_per_year)
        if len(means) < 2:
            return False, None, "deflated Sharpe needs at least 2 trials in the family"
        sr0 = expected_max_sharpe(len(means), float(np.var(means, ddof=1)))
        sr_hat = float(np.mean(cand_scores)) / np.sqrt(periods_per_year)
        t_obs = len(cand_scores) * int(n_obs)
        dsr = deflated_sharpe_ratio(sr_hat, sr0, t_obs)
        excess = (sr_hat - sr0) * np.sqrt(periods_per_year)
        if dsr >= threshold:
            return True, excess, ""
        return False, excess, f"deflated Sharpe {dsr:.2f} below {threshold:.2f} after {len(means)} trials"

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
        deflated_delta = None
        is_sharpe = "sharpe" in self.metric.lower()
        use_dsr = is_sharpe and self.dsr_threshold is not None
        n_obs = c.get("n_obs")
        if use_dsr and (not n_obs or n_obs != i.get("n_obs")):
            # Sharpe is not an additive score, so it needs its sample size to be deflated.
            method, reason = "dsr-requires-n_obs", "Sharpe promotion needs n_obs (bars per score) recorded identically on both trials"
        elif k >= 3 and len(cs) == len(ins):
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
                    dsr_ok = True
                    if use_dsr:
                        method = "paired-maxT-bootstrap+dsr"
                        dsr_ok, deflated_delta, dsr_reason = self._dsr_check(cs, trials, n_obs, self.dsr_threshold)
                    if protected_candidate is None or protected_incumbent is None:
                        reason = "protected episode scores missing"
                    elif sign * (protected_candidate - protected_incumbent) < -abs(max_protected_regression):
                        outcome, reason = "reject", "protected episode regression"
                    elif p < 0.05 and delta > 0 and dsr_ok:
                        outcome, reason = "promote", "search-corrected bootstrap passed" + (" and deflated Sharpe cleared" if use_dsr else "")
                    elif p < 0.05 and delta > 0 and not dsr_ok:
                        outcome, reason = "reject", dsr_reason
                    else:
                        outcome, reason = "reject", "search-corrected significance not met"
        decision = GateDecision(outcome, delta, deflated_delta, p, len(family), k, method, reason)
        self.ledger.append("decision", family=self.family, metric=self.metric,
                           candidate_id=candidate, incumbent_id=incumbent, **asdict(decision))
        return decision
