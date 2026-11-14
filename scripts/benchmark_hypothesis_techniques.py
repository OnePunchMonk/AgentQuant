"""Seeded offline comparison of four hypothesis-generation techniques.

All return series and relevance labels here are synthetic fixtures. This is
method plumbing and benchmark instrumentation, not real-market evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.generators import (  # noqa: E402
    AnalyticalGenerator,
    EnsembleGenerator,
    HydeGenerator,
    LiteratureGenerator,
)
from src.agent.hyde_retrieval import Evidence, HypothesisRequest  # noqa: E402
from src.agent.hypothesis_agent import HypothesisAgent  # noqa: E402

REGIMES = {
    "bull_trend": (0.00055, 0.009),
    "bear_trend": (-0.00035, 0.014),
    "low_vol_range": (0.00008, 0.004),
    "high_vol_range": (0.00002, 0.018),
    "crisis_rebound": (0.0003, 0.025),
    "inflation_shock": (-0.00008, 0.016),
}
METHODS = ("literature", "hyde", "analytical", "ensemble")

# Calendar windows are fixed before examining results. They are descriptive
# market eras, not labels inferred from the holdout returns.
HISTORICAL_REGIMES = {
    "dotcom_bear": ("2000-03-10", "2002-10-09"),
    "housing_credit_crisis": ("2007-10-09", "2009-03-09"),
    "post_crisis_expansion": ("2010-01-04", "2015-12-31"),
    "volmageddon": ("2018-01-26", "2018-12-24"),
    "covid_crash_rebound": ("2020-02-19", "2020-12-31"),
    "inflation_rate_shock": ("2022-01-03", "2022-10-12"),
    "2023_recovery": ("2023-01-03", "2023-12-29"),
    "2024_expansion": ("2024-01-02", "2024-12-31"),
}


def load_historical_prices(csv_path=None, *, symbol="SPY", start="2000-01-01", end="2025-01-01"):
    """Load adjusted daily close from a user CSV or pinned Yahoo Finance dates.

    CSV requires Date and Adj Close (Close is accepted if adjusted prices are
    unavailable and this is recorded in provenance). Yahoo is optional and
    imported lazily. Failure raises; there is no synthetic fallback.
    """
    import pandas as pd

    if csv_path:
        frame = pd.read_csv(csv_path)
        if "Date" not in frame:
            raise ValueError("CSV must contain Date")
        price_col = "Adj Close" if "Adj Close" in frame else "Close" if "Close" in frame else None
        if not price_col:
            raise ValueError("CSV must contain Adj Close or Close")
        frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None)
        frame = frame[["Date", price_col]].rename(columns={price_col: "Close"}).dropna()
        frame = frame.drop_duplicates("Date").sort_values("Date")
        frame = frame[(frame.Date >= start) & (frame.Date < end)]
        source = {
            "provider": "user_csv",
            "path": str(Path(csv_path).resolve()),
            "price_column": price_col,
        }
    else:
        try:
            import yfinance as yf

            frame = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                actions=False,
                threads=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Yahoo Finance fetch failed: {exc}") from exc
        if frame is None or frame.empty:
            raise RuntimeError(
                "Yahoo Finance returned no prices; provide --csv with public daily prices"
            )
        close = frame["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        frame = close.rename("Close").rename_axis("Date").reset_index()
        source = {
            "provider": "Yahoo Finance via yfinance",
            "symbol": symbol,
            "start_inclusive": start,
            "end_exclusive": end,
            "auto_adjust": True,
        }
    if len(frame) < 2:
        raise RuntimeError("historical price input contains fewer than two rows")
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
    frame["Return"] = frame["Close"].pct_change()
    frame = frame.dropna(subset=["Return"]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("historical price input has no usable returns")
    source["first_date"] = frame.Date.iloc[0].isoformat()
    source["last_date"] = frame.Date.iloc[-1].isoformat()
    source["rows"] = int(len(frame))
    source["price_sha256"] = hashlib.sha256(
        frame[["Date", "Close"]].to_csv(index=False, float_format="%.12g").encode("utf-8")
    ).hexdigest()
    return frame[["Date", "Close", "Return"]], source


def _causal_strategy_returns(candidate, prices, cost_bps):
    """Close-to-close returns with yesterday-known signals and turnover costs."""
    import numpy as np
    import pandas as pd

    close = pd.Series(prices["Close"].to_numpy(dtype=float))
    ret = pd.Series(prices["Return"].to_numpy(dtype=float))
    typ, params = candidate.strategy_type, candidate.params
    if typ == "momentum":
        fast = int(params.get("fast_window", 20))
        slow = int(params.get("slow_window", 100))
        fast = max(2, min(fast, len(close) - 1))
        slow = max(fast + 1, min(slow, len(close) - 1))
        signal = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)
    elif typ == "mean_reversion":
        window = max(2, min(int(params.get("window", 20)), len(close) - 1))
        mean = close.rolling(window).mean()
        std = close.rolling(window).std().replace(0, np.nan)
        z = (close - mean) / std
        signal = pd.Series(np.where(z < -1.0, 1.0, np.where(z > 1.0, -1.0, 0.0)))
    else:
        window = max(2, min(int(params.get("window", 20)), len(close) - 1))
        vol = ret.rolling(window).std()
        cap = vol.expanding(min_periods=window).median()
        signal = (vol <= cap).astype(float)
    position = signal.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    return (position * ret - turnover * cost_bps / 10000).to_numpy(dtype=float)


def _stats(values):
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return {"observations": 0, "sharpe": 0.0, "total_return": 0.0}
    wealth = math.prod(1 + x for x in vals)
    return {
        "observations": len(vals),
        "sharpe": round(_sharpe(vals), 6),
        "total_return": round(wealth - 1, 6),
    }


def run_historical_benchmark(
    *, csv_path=None, symbol="SPY", seed=29, candidate_budget=4, cost_bps=5.0
):
    """Evaluate proposals on public historical price data with chronological holdouts."""
    import pandas as pd

    prices, provenance = load_historical_prices(csv_path, symbol=symbol)
    # Strategy catalog is static and available before every regime; evidence
    # text is not used as a performance label or scored for quality.
    evidence = [
        Evidence(
            f"catalog-{s}",
            f"{s} strategy parameter catalog",
            "benchmark_catalog",
            "1999-12-31",
            "1999-12-31",
            s,
            {"fast_window": 5, "slow_window": 20}
            if s == "momentum"
            else ({"window": 10, "num_std": 1.5} if s == "mean_reversion" else {"window": 20}),
            evidence_tier="parameter_catalog",
        )
        for s in ("momentum", "mean_reversion", "volatility")
    ]
    literature = LiteratureGenerator(evidence)
    hyde = HydeGenerator()
    analytical = AnalyticalGenerator()
    generators = {
        "literature": literature,
        "hyde": hyde,
        "analytical": analytical,
        "ensemble": EnsembleGenerator((literature, hyde, analytical)),
    }
    agent = HypothesisAgent(generators, evidence)
    records = []
    dates = pd.to_datetime(prices.Date)
    for regime, (begin, finish) in HISTORICAL_REGIMES.items():
        subset = prices[(dates >= begin) & (dates <= finish)].reset_index(drop=True)
        if len(subset) < 40:
            raise RuntimeError(f"insufficient actual data for regime {regime} ({begin}..{finish})")
        split = max(2, int(len(subset) * 0.6))
        train, holdout = (
            subset.iloc[:split].reset_index(drop=True),
            subset.iloc[split:].reset_index(drop=True),
        )
        rows = {}
        request = HypothesisRequest(
            regime,
            ("momentum", "mean_reversion", "volatility"),
            train.Date.iloc[-1].isoformat(),
            candidate_budget,
            8,
        )
        for method in METHODS:
            run = agent.run(method, request)
            evaluated = []
            for candidate in run.candidates[:candidate_budget]:
                train_stats = _stats(_causal_strategy_returns(candidate, train, cost_bps))
                hold_stats = _stats(_causal_strategy_returns(candidate, holdout, cost_bps))
                evaluated.append(
                    {
                        "strategy_type": candidate.strategy_type,
                        "params": dict(candidate.params),
                        "train": train_stats,
                        "holdout": hold_stats,
                    }
                )
            rows[method] = {
                "candidate_count": len(evaluated),
                "budget": candidate_budget,
                "cost_bps_per_unit_turnover": cost_bps,
                "holdout_mean_sharpe": round(
                    sum(x["holdout"]["sharpe"] for x in evaluated) / len(evaluated), 6
                )
                if evaluated
                else None,
                "candidates": evaluated,
                "cost_units": run.cost_units,
            }
        records.append(
            {
                "regime": regime,
                "date_start": begin,
                "date_end": finish,
                "observations": len(subset),
                "split_date": train.Date.iloc[-1].isoformat(),
                "train_observations": len(train),
                "holdout_observations": len(holdout),
                "metrics": rows,
            }
        )
    return {
        "benchmark": "hypothesis_techniques_public_historical_v1",
        "seed": seed,
        "claim_scope": "historical descriptive backtest; not a prospective or investment claim",
        "data_label": "public_historical_daily_prices",
        "symbol": symbol,
        "provenance": provenance,
        "regime_count": len(records),
        "candidate_budget": candidate_budget,
        "transaction_cost_model": "fixed basis points per unit change in position",
        "regimes": {
            name: {"start": bounds[0], "end": bounds[1]}
            for name, bounds in HISTORICAL_REGIMES.items()
        },
        "results": records,
    }


def make_fixture(regime: str, seed: int = 29, observations: int = 252):
    """Return synthetic daily market and strategy returns, deterministic by regime."""
    drift, sigma = REGIMES[regime]
    rng = random.Random(seed + sum(map(ord, regime)))
    market = [rng.gauss(drift, sigma) for _ in range(observations)]
    evidence = []
    strategy_types = ("momentum", "mean_reversion", "volatility")
    for i in range(12):
        strategy = strategy_types[i % len(strategy_types)]
        known = "2024-06-01" if i < 10 else "2025-02-01"
        evidence.append(
            Evidence(
                f"{regime}-{i:02d}",
                f"{regime} historical {strategy} signal study {i}",
                "synthetic_historical_fixture",
                "2024-05-31",
                known,
                strategy,
                {"fast_window": 5, "slow_window": 20} if strategy == "momentum" else {},
                quality=0.8 if i < 3 else 0.25,
                measured_sharpe=None,
                evidence_tier="synthetic_fixture",
            )
        )
    return market, evidence


def _sharpe(returns):
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((x - mean) ** 2 for x in returns) / (len(returns) - 1)
    return mean / math.sqrt(var) * math.sqrt(252) if var else 0.0


def _candidate_returns(candidate, market, regime):
    # Explicit fixture scoring rule: method/params receive transparent synthetic
    # signal transforms; no candidate has been backtested on historical prices.
    rng = random.Random(sum(map(ord, candidate.hypothesis + regime)))
    quality = 0.0001 if candidate.params else -0.00005
    noise = [rng.gauss(0, 0.003) for _ in market]
    return [quality + 0.12 * x + n for x, n in zip(market, noise)]


def _novelty(candidate, evidence):
    signatures = {tuple(sorted(e.params.items())) for e in evidence if e.params}
    return int(bool(candidate.params) and tuple(sorted(candidate.params.items())) not in signatures)


def evaluate_regime(regime, *, seed=29, candidate_budget=4):
    market, evidence = make_fixture(regime, seed)
    literature = LiteratureGenerator(evidence)
    hyde = HydeGenerator()
    analytical = AnalyticalGenerator()
    generators = {
        "literature": literature,
        "hyde": hyde,
        "analytical": analytical,
        "ensemble": EnsembleGenerator((literature, hyde, analytical)),
    }
    agent = HypothesisAgent(generators, evidence)
    request = HypothesisRequest(
        regime, ("momentum", "mean_reversion", "volatility"), "2024-12-31", candidate_budget, 8
    )
    rows = {}
    for method in METHODS:
        run = agent.run(method, request)
        sh = [_sharpe(_candidate_returns(c, market, regime)) for c in run.candidates]
        novelty = [_novelty(c, evidence) for c in run.candidates]
        false_positive = sum(s <= 0 for s in sh) / len(sh) if sh else 1.0
        gap = (sum(max(s, 0) for s in sh) / len(sh) - sum(sh) / len(sh)) if sh else 0.0
        rows[method] = {
            "candidate_count": len(sh),
            "fixture_sharpe_mean": sum(sh) / len(sh) if sh else 0.0,
            "fixture_sharpe_gap": gap,
            "novelty_rate": sum(novelty) / len(novelty) if novelty else 0.0,
            "false_positive_rate": false_positive,
            "cost_units": run.cost_units,
            "learning_curve_fixture_sharpe": [
                round(sum(sh[:i]) / i, 6) for i in range(1, len(sh) + 1)
            ],
            "candidate_params": [dict(c.params) for c in run.candidates],
        }
    return {
        "regime": regime,
        "data_label": "synthetic_fixture_not_real_historical_data",
        "observations": len(market),
        "metrics": rows,
    }


def run_benchmark(seed=29, candidate_budget=4):
    return {
        "benchmark": "hypothesis_techniques_offline_fixture_v1",
        "seed": seed,
        "claim_scope": "synthetic deterministic fixtures only; no real performance claim",
        "regime_count": len(REGIMES),
        "results": [
            evaluate_regime(r, seed=seed, candidate_budget=candidate_budget) for r in REGIMES
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--candidate-budget", type=int, default=4)
    parser.add_argument("--mode", choices=("fixture", "historical"), default="fixture")
    parser.add_argument("--csv", help="public daily price CSV with Date and Adj Close or Close")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.candidate_budget <= 0:
        parser.error("--candidate-budget must be positive")
    if args.mode == "historical":
        try:
            benchmark = run_historical_benchmark(
                csv_path=args.csv,
                symbol=args.symbol,
                seed=args.seed,
                candidate_budget=args.candidate_budget,
                cost_bps=args.cost_bps,
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(f"historical benchmark aborted (no fixture fallback): {exc}")
    else:
        benchmark = run_benchmark(args.seed, args.candidate_budget)
    result = json.dumps(benchmark, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
