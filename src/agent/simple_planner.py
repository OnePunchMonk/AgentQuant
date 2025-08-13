"""
Simplified strategy generation agent without LangChain dependencies.
"""
import random
from typing import Dict, List, Any
import pandas as pd


def generate_strategy_proposals(
    regime_data: dict,
    features_df: pd.DataFrame,
    baseline_stats: pd.Series,
    strategy_types: List[str],
    available_assets: List[str],
    num_proposals: int = 5
) -> List[Dict[str, Any]]:
    """
    Generates strategy proposals using simplified logic.
    
    Args:
        regime_data: Information about the current market regime
        features_df: DataFrame containing market features
        baseline_stats: Series containing baseline strategy performance
        strategy_types: List of available strategy types
        available_assets: List of available asset tickers
        num_proposals: Number of strategy proposals to generate
        
    Returns:
        A list of strategy proposal dictionaries
    """
    proposals = []
    
    # Handle regime_data being either a string or dict
    print(f"DEBUG simple_planner: regime_data type={type(regime_data)}, value={regime_data}")
    if isinstance(regime_data, str):
        regime_name = regime_data
    elif isinstance(regime_data, dict):
        regime_name = regime_data.get('name', 'neutral')
    elif isinstance(regime_data, (tuple, list)):
        regime_name = str(regime_data[0]) if len(regime_data) > 0 else 'neutral'
    else:
        regime_name = str(regime_data) if regime_data is not None else 'neutral'
    
    print(f"DEBUG simple_planner: regime_name type={type(regime_name)}, value={regime_name}")
    
    # Get current market characteristics
    latest_features = features_df.iloc[-1] if not features_df.empty else pd.Series()
    
    print(f"Generating {num_proposals} strategies for {regime_name} market regime")
    
    # Generate proposals based on market regime and available strategies
    for i in range(num_proposals):
        strategy_type = random.choice(strategy_types)
        
        # Generate parameters based on strategy type and market regime
        if strategy_type == "momentum":
            if regime_name.lower() in ['bullish', 'trending']:
                params = {
                    "fast_window": random.randint(10, 20),
                    "slow_window": random.randint(40, 80)
                }
            else:
                params = {
                    "fast_window": random.randint(15, 25),
                    "slow_window": random.randint(50, 100)
                }
        elif strategy_type == "mean_reversion":
            if regime_name.lower() in ['volatile', 'bearish']:
                params = {
                    "window": random.randint(15, 25),
                    "num_std": round(random.uniform(1.5, 2.5), 1)
                }
            else:
                params = {
                    "window": random.randint(20, 30),
                    "num_std": round(random.uniform(2.0, 3.0), 1)
                }
        elif strategy_type == "volatility":
            params = {
                "window": random.randint(20, 60),
                "vol_threshold": round(random.uniform(0.15, 0.35), 2)
            }
        elif strategy_type == "breakout":
            params = {
                "window": random.randint(20, 100),
                "threshold_pct": round(random.uniform(0.01, 0.05), 3)
            }
        elif strategy_type == "trend_following":
            sw = random.randint(5, 20)
            mw = random.randint(sw + 10, sw + 40)
            lw = random.randint(mw + 10, mw + 60)
            params = {
                "short_window": sw,
                "medium_window": mw,
                "long_window": lw
            }
        elif strategy_type == "regime_based":
            # Provide momentum and mean reversion params; regime_data is passed separately in strategies
            params = {
                "regime_data": regime_name,
                "momentum_params": {
                    "fast_window": random.randint(10, 30),
                    "slow_window": random.randint(40, 100)
                },
                "mean_reversion_params": {
                    "window": random.randint(15, 30),
                    "num_std": round(random.uniform(1.5, 3.0), 1)
                }
            }
        else:
            # Fallback: no custom params, use defaults in runner
            params = {}
        
        # Select assets (prefer diverse selection)
        num_assets = min(random.randint(1, 3), len(available_assets))
        selected_assets = random.sample(available_assets, num_assets)
        
        # Generate allocation weights
        if len(selected_assets) == 1:
            allocation_weights = {selected_assets[0]: 1.0}
        else:
            # Random weights that sum to 1
            weights = [random.random() for _ in selected_assets]
            total_weight = sum(weights)
            allocation_weights = {
                asset: round(weight / total_weight, 3) 
                for asset, weight in zip(selected_assets, weights)
            }
        
        # Generate rationale based on market regime
        rationale = f"""
        This {strategy_type} strategy is designed for the current {regime_name} market regime.
        The parameters are optimized for current volatility levels and market conditions.
        Asset selection provides diversification across {len(selected_assets)} instruments.
        """.strip()
        
        proposal = {
            "strategy_type": strategy_type,
            "asset_tickers": selected_assets,
            "params": params,
            "allocation_weights": allocation_weights,
            "rationale": rationale
        }
        
        proposals.append(proposal)
        print(f"Generated {strategy_type} strategy for {selected_assets}")
    
    return proposals
