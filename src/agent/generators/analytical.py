"""Rule-based analytical proposal generator using regime descriptors."""

from src.agent.hyde_retrieval import HypothesisRequest, SyntheticHypothesis


class AnalyticalGenerator:
    """Choose interpretable parameter hypotheses from simple regime rules."""

    def generate(self, request: HypothesisRequest):
        label = request.regime.lower()
        out = []
        for strategy in request.strategy_types[:request.max_hypotheses]:
            if strategy == "momentum":
                params = {"fast_window": 5, "slow_window": 20} if any(
                    word in label for word in ("crisis", "highvol", "bear")
                ) else {"fast_window": 20, "slow_window": 100}
            elif strategy == "mean_reversion":
                params = {"window": 10, "num_std": 1.5} if "highvol" in label else {
                    "window": 20, "num_std": 2.0
                }
            else:
                params = {}
            out.append(SyntheticHypothesis(
                f"Use analytical {strategy.replace('_', ' ')} settings under {request.regime}.",
                strategy, params,
            ))
        return out
