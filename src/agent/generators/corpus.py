"""Evidence for the hypothesis generators: a dated, cited literature corpus and the agent's own memory.

The earlier historical benchmark used one "parameter catalog" row per strategy, so every method
drew its parameters from the same three rows and returned the same candidates. This module gives
the methods real, dated evidence to differ on:

* ``literature_corpus()``: published studies and one practitioner source. The text of each row is a
  short hand-written paraphrase of what the source reports (verify against the source before
  quoting it). No performance number is stored; ``measured_sharpe`` is None for every row. The
  ``params`` are this repo's interpretation of the study's horizon, not something the paper
  prescribes, and are labelled as such in the text.
* ``evidence_from_memory()``: the agent's own trials, as point-in-time evidence.

``market_end`` is the last date of the sample the source used. ``known_at`` is the end of its
publication year, which is conservative: a study is never visible before it appeared.
"""

from __future__ import annotations

from typing import Any, Sequence

from src.agent.hyde_retrieval import Evidence

LITERATURE_TIER = "published_literature_summary"


def _row(eid, strategy, text, source, market_end, known_at, params, tier=LITERATURE_TIER):
    return Evidence(eid, text, source, market_end, known_at, strategy, params, quality=0.5,
                    measured_sharpe=None, evidence_tier=tier)


_LITERATURE = (
    _row("lit-jegadeesh-titman-1993", "momentum",
         "Jegadeesh and Titman (1993), Journal of Finance: US stocks that were winners over the past 3 to 12 "
         "months continued to outperform losers over the next 3 to 12 months (1965 to 1989). Winners persist at "
         "intermediate horizons. Interpreted here as a 1-month versus 6-month crossover.",
         "Journal of Finance 48(1)", "1989-12-31", "1993-12-31", {"fast_window": 21, "slow_window": 126}),
    _row("lit-brock-lakonishok-lebaron-1992", "momentum",
         "Brock, Lakonishok and LeBaron (1992), Journal of Finance: moving average rules on the Dow Jones "
         "Industrial Average (1897 to 1986) compared a short average of 1 to 5 days with a long average of 50 to "
         "200 days; buy signals outperformed sell signals. Interpreted here as a 5-day versus 150-day crossover.",
         "Journal of Finance 47(5)", "1986-12-31", "1992-12-31", {"fast_window": 5, "slow_window": 150}),
    _row("lit-faber-2007", "momentum",
         "Faber (2007), Journal of Wealth Management: holding an asset class only while it trades above its "
         "10-month simple moving average, otherwise cash, reduced drawdowns and volatility in a tactical "
         "allocation test (1973 to 2005). A defensive trend rule. Interpreted here as a 200-day average.",
         "Journal of Wealth Management 9(4)", "2005-12-31", "2007-12-31", {"slow_window": 200}),
    _row("lit-moskowitz-ooi-pedersen-2012", "momentum",
         "Moskowitz, Ooi and Pedersen (2012), Journal of Financial Economics: time series momentum. An "
         "instrument's own return over the past 12 months predicts its next-month return across dozens of liquid "
         "futures and forwards (1985 to 2009); trend returns persist. Interpreted here as a 3-month versus "
         "12-month crossover.",
         "Journal of Financial Economics 104(2)", "2009-12-31", "2012-12-31", {"fast_window": 63, "slow_window": 252}),
    _row("lit-hurst-ooi-pedersen-2017", "momentum",
         "Hurst, Ooi and Pedersen (2017), Journal of Portfolio Management: trend following across many markets "
         "from 1880 to 2016 using signals over several lookbacks from one to twelve months; trend returns persist "
         "across regimes including crisis periods. Interpreted here as a 1-month versus 12-month crossover.",
         "Journal of Portfolio Management 44(1)", "2016-12-31", "2017-12-31", {"fast_window": 21, "slow_window": 252}),
    _row("lit-jegadeesh-lehmann-1990", "mean_reversion",
         "Jegadeesh (1990, Journal of Finance) and Lehmann (1990, Quarterly Journal of Economics): individual "
         "stock returns show negative serial correlation at the one-month and one-week horizons, a short-term "
         "reversal of stretched prices (US equities to 1987). Interpreted here as a 10-day window.",
         "Journal of Finance 45(3); Quarterly Journal of Economics 105(1)", "1987-12-31", "1990-12-31",
         {"window": 10, "num_std": 1.5}),
    _row("lit-bollinger-2001", "mean_reversion",
         "Bollinger (2001), Bollinger on Bollinger Bands (practitioner): bands two standard deviations around a "
         "20-day moving average flag stretched prices in range-bound markets, a reversal setup.",
         "Bollinger on Bollinger Bands (book)", "2000-12-31", "2001-12-31", {"window": 20, "num_std": 2.0},
         tier="practitioner_description"),
    _row("lit-moreira-muir-2017", "volatility",
         "Moreira and Muir (2017), Journal of Finance: scaling exposure inversely to the prior month's realized "
         "variance earned higher risk-adjusted returns than holding unscaled (1926 to 2015); managing volatility "
         "limits downside in turbulent periods. Interpreted here as a 21-day window.",
         "Journal of Finance 72(4)", "2015-12-31", "2017-12-31", {"window": 21}),
)


def literature_corpus() -> tuple:
    """The cited literature rows. Which are visible depends on the request's cutoffs."""
    return _LITERATURE


def regime_query(regime: str) -> str:
    """Words describing a regime label, for matching against evidence text."""
    r = regime.lower()
    words = []
    if any(w in r for w in ("bear", "crisis", "crash", "highvol", "shock", "volmageddon")):
        words += ["drawdowns", "volatility", "defensive", "reduced", "downside"]
    if any(w in r for w in ("bull", "trend", "expansion", "recovery", "rebound")):
        words += ["winners", "persist", "trend", "continued", "outperform"]
    if any(w in r for w in ("range", "neutral", "lowvol", "midvol")):
        words += ["reversal", "stretched", "range", "negative", "serial", "correlation"]
    return " ".join(words) or regime.replace("_", " ")


def evidence_from_memory(service: Any, query: Any) -> Sequence[Evidence]:
    """The agent's own trials visible under ``query`` as point-in-time evidence.

    For a replay the knowledge date is the end of the window the trial was measured on (its
    out-of-sample window if it has one), not the wall-clock time it was stored, so a trial is never
    visible before the data it was measured on existed.
    """
    rows = []
    for trial in service.visible_trials(query):
        end = trial.oos_end if trial.oos_sharpe is not None and trial.oos_end else trial.data_end
        if not end:
            continue
        measured = trial.oos_sharpe if trial.oos_sharpe is not None else trial.is_sharpe
        tier = "measured_out_of_sample" if trial.oos_sharpe is not None else "measured_in_sample"
        text = (f"{trial.strategy_type} {dict(trial.params)} under {trial.regime_label}: "
                f"in-sample Sharpe {trial.is_sharpe}, out-of-sample Sharpe {trial.oos_sharpe}")
        rows.append(Evidence(f"mem-{trial.trial_id}", text, "agentquant_memory", end[:10], end[:10],
                             trial.strategy_type, dict(trial.params),
                             quality=0.7 if trial.oos_sharpe is not None else 0.3,
                             measured_sharpe=measured, evidence_tier=tier))
    return rows
