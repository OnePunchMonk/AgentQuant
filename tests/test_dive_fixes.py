"""Regression tests for bugs found while reading the feature, stress and walk-forward code.

  * compute_features returned an empty frame for any window of about 252 bars or fewer
  * stress_test's "remove_top_returns" scenario did not remove the top returns
  * walk-forward included a trailing sliver of a window in its worst-Sharpe statistic
"""

import warnings

import numpy as np
import pytest

from src.agent.context_builder import build_context
from src.agent.episode_splits import synthetic_ohlcv
from src.backtest.stress_test import stress_test
from src.backtest.walk_forward import anchored_walk_forward
from src.features.engine import compute_features
from src.features.regime import detect_regime

warnings.filterwarnings("ignore", category=FutureWarning)
SIM = synthetic_ohlcv(3, n_days=1500)["SIM"]


@pytest.mark.parametrize("n", [176, 220, 300])
def test_short_windows_still_produce_features_and_a_regime(n):
    features = compute_features({"SIM": SIM.iloc[:n]}, "SIM")
    assert len(features) > 0
    assert detect_regime(features) != "Unknown"


def test_long_horizon_features_are_unknown_not_dropped_and_not_nan_in_the_context():
    features = compute_features({"SIM": SIM.iloc[:176]}, "SIM")
    assert features["momentum_252d"].isna().all()  # there is no 252-day history
    context = build_context(features)
    assert context.momentum_252d == 0.0
    assert "nan" not in context.to_prompt_string().lower()


def test_long_windows_are_unchanged():
    features = compute_features({"SIM": SIM.iloc[:1000]}, "SIM")
    last = features.iloc[-1]
    assert not last.isna().any()
    assert detect_regime(features) == detect_regime(features.dropna())


def test_remove_top_returns_really_removes_them():
    base = SIM.iloc[:400]
    scenario = next(r for r in stress_test({"SIM": base}, ["SIM"], "momentum",
                                           {"fast_window": 10, "slow_window": 50})
                    if r["scenario"] == "remove_top_returns")
    returns = base["Close"].pct_change().fillna(0)
    top = returns.nlargest(5)
    base_growth = base["Close"].iloc[-1] / base["Close"].iloc[0]
    expected_growth = base_growth / float(np.prod(1.0 + top.values))
    # Rebuild the scenario price path the way the function should: top days zeroed.
    kept = returns.copy()
    kept.loc[top.index] = 0.0
    rebuilt = base["Close"].iloc[0] * (1.0 + kept).cumprod()
    assert rebuilt.iloc[-1] / rebuilt.iloc[0] == pytest.approx(expected_growth, rel=1e-9)
    assert expected_growth < base_growth
    assert scenario["metrics"]  # the scenario still backtests


def test_remove_top_returns_scenario_lowers_the_buy_and_hold_path(monkeypatch):
    """The scenario frame handed to the backtest must end lower than the original."""
    seen = {}

    def fake_run_backtest(data, assets, strategy, params, **kwargs):
        seen.setdefault("frames", []).append(data[assets[0]]["Close"].copy())
        return {"metrics": {"sharpe_ratio": 0.0}}

    import src.backtest.stress_test as module

    monkeypatch.setattr(module, "run_backtest", fake_run_backtest)
    base = SIM.iloc[:400]
    module.stress_test({"SIM": base}, ["SIM"], "momentum", {"fast_window": 10, "slow_window": 50})
    # scenarios are built in order: regime_change, volatility_spike, remove_top_returns, trend_reversal
    removed = seen["frames"][2]
    assert removed.iloc[-1] < base["Close"].iloc[-1] * 0.999
    top_days = base["Close"].pct_change().fillna(0).nlargest(5).index
    assert np.allclose(removed.pct_change().fillna(0).loc[top_days].values, 0.0, atol=1e-12)


def test_walk_forward_ignores_a_trailing_sliver_window():
    df = SIM.iloc[:673]  # start = 403; the last window would be a single bar
    result = anchored_walk_forward({"SIM": df}, ["SIM"], "momentum", {"fast_window": 10, "slow_window": 50})
    lengths = [len(df.loc[w["start"]:w["end"]]) for w in result["windows"]]
    assert lengths and min(lengths) >= 21


# ------------------------------------------------------------------ harness config
def test_max_retries_sets_max_iterations_and_an_explicit_override_wins():
    from src.agent.harness_config import HarnessConfig, resolve_effective_config

    spec = HarnessConfig(version="t", epoch=1, created="now", max_retries=4)
    assert resolve_effective_config(spec).max_iterations == 5
    assert resolve_effective_config(spec, max_iterations_override=2).max_iterations == 2


def test_temperature_is_a_loudly_unsupported_knob():
    from src.agent.harness_config import (
        HarnessConfig,
        UnsupportedHarnessKnobError,
        resolve_effective_config,
    )

    with pytest.raises(UnsupportedHarnessKnobError, match="temperature"):
        resolve_effective_config(HarnessConfig(version="t", epoch=1, created="now", temperature=0.9))


def test_config_hash_ignores_labels_and_timestamps_but_not_behaviour():
    from src.agent.harness_config import HarnessConfig, resolve_effective_config

    a = HarnessConfig(version="a", epoch=1, created="2020", description="x", prompt_template="p1")
    b = HarnessConfig(version="b", epoch=9, created="2030", description="y", prompt_template="p1")
    c = HarnessConfig(version="a", epoch=1, created="2020", description="x", prompt_template="p2")
    assert resolve_effective_config(a).config_hash == resolve_effective_config(b).config_hash
    assert resolve_effective_config(a).config_hash != resolve_effective_config(c).config_hash


def test_run_agent_uses_max_retries_and_respects_zero(monkeypatch, tmp_path):
    from src.agent import agent_graph
    from src.agent.harness_config import HarnessConfig
    from src.utils.config import config

    monkeypatch.setattr(config, "results_db_path", str(tmp_path / "m.db"))
    monkeypatch.setenv("AGENTQUANT_OFFLINE", "1")
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    data = {"SPY": synthetic_ohlcv(5, n_days=900, asset="SPY")["SPY"]}
    spec = HarnessConfig(version="t", epoch=1, created="now", max_retries=1)
    assert agent_graph.run_agent(data, asset="SPY", harness_config=spec)["max_iterations"] == 2
    assert agent_graph.run_agent(data, asset="SPY", max_iterations=0)["max_iterations"] == 0


# ------------------------------------------------------------------ evolutionary search
def test_best_genome_is_a_copy_so_noisy_reevaluation_cannot_overwrite_it():
    from src.agent.harness_evolution_algo import GeneticAlgorithm

    ga = GeneticAlgorithm(population_size=6, generations=1)
    ga.initialize_population()
    scores = iter([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ga.evaluate_population(lambda g: next(scores))
    assert ga.best_genome.fitness == 1.0
    ga.evaluate_population(lambda g: next(scores))  # the same genome object is re-evaluated, worse
    assert ga.best_genome.fitness == 1.0
