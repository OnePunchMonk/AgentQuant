"""
AI Agent Strategy Planner using LangChain/LangGraph.

This module provides the core AI agent for generating trading strategy proposals
using LangChain's workflow orchestration capabilities.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random

# LangChain dependencies
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

def generate_random_strategies(
    regime_data: dict,
    features_df: pd.DataFrame,
    baseline_stats: pd.Series,
    strategy_types: List[str],
    available_assets: List[str],
    num_proposals: int = 5
) -> List[Dict[str, Any]]:
    """
    Generates strategy proposals using random logic (Baseline).
    """
    proposals = []
    
    # Handle regime_data being either a string or dict
    if isinstance(regime_data, str):
        regime_name = regime_data
        # Estimate volatility based on regime name
        if 'HighVol' in regime_name or 'Crisis' in regime_name:
            current_vol = 0.25
        elif 'MidVol' in regime_name:
            current_vol = 0.18
        else:  # LowVol
            current_vol = 0.12
    else:
        # Get current market volatility for parameter scaling
        current_vol = regime_data.get('current_volatility', 0.15)
        regime_name = regime_data.get('current_regime', 'normal')
    
    for i in range(num_proposals):
        # Select random strategy type
        strategy_type = random.choice(strategy_types)
        
        # Generate strategy parameters based on type and market regime
        if strategy_type == "momentum":
            # Momentum strategy expects fast_window and slow_window
            fw = random.randint(10, 40)
            sw = random.randint(fw + 10, 100)
            params = {
                "fast_window": fw,
                "slow_window": sw
            }
        elif strategy_type == "mean_reversion":
            params = {
                "lookback_period": random.randint(20, 100),
                "entry_threshold": random.uniform(1.5, 3.0),
                "exit_threshold": random.uniform(0.5, 1.0),
                "stop_loss": random.uniform(0.05, 0.15)
            }
        elif strategy_type == "volatility":
            params = {
                "window": random.randint(20, 60),
                "vol_threshold": round(random.uniform(0.1, 0.3), 3)
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
            # Default parameters for unknown strategy types
            params = {
                "lookback_period": random.randint(20, 50),
                "threshold": random.uniform(0.01, 0.05)
            }
        
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
    
    return proposals

class StrategyParams(BaseModel):
    fast_window: Optional[int] = Field(description="Fast moving average window (for momentum)")
    slow_window: Optional[int] = Field(description="Slow moving average window (for momentum)")
    lookback_window: Optional[int] = Field(description="Lookback window (for other strategies)")
    entry_threshold: Optional[float] = Field(description="Entry threshold")
    stop_loss: Optional[float] = Field(description="Stop loss percentage")
    reasoning: str = Field(description="Reasoning for the chosen parameters")

def generate_strategy_proposals(
    regime_data: dict,
    features_df: pd.DataFrame,
    baseline_stats: pd.Series,
    strategy_types: List[str],
    available_assets: List[str],
    num_proposals: int = 5
) -> List[Dict[str, Any]]:
    """
    Generates strategy proposals using Gemini LLM.
    """
    
    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not found. Falling back to random strategy generation.")
        return generate_random_strategies(regime_data, features_df, baseline_stats, strategy_types, available_assets, num_proposals)

    try:
        # Try using gemini-2.5-flash as requested
        # Disable retries to fail fast and fallback to random
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=0)
        
        proposals = []
        
        # Prepare Context
        if isinstance(regime_data, str):
            regime_name = regime_data
        else:
            regime_name = regime_data.get('current_regime', 'Unknown')
            
        # Get latest technicals
        if not features_df.empty:
            latest_features = features_df.iloc[-1].to_dict()
            technical_summary = ", ".join([f"{k}: {v:.2f}" for k, v in latest_features.items() if isinstance(v, (int, float))])
        else:
            technical_summary = "No technical data available."

        parser = JsonOutputParser(pydantic_object=StrategyParams)

        prompt = PromptTemplate(
            template="""Act as a Quantitative Researcher. Based on this context, select optimal parameters for a {strategy_type} Strategy.
            
            Input:
            Market Regime: {regime_name}
            Technical Summary: {technical_summary}
            Asset Name: {asset_name}
            
            Task: Return a JSON object with the optimal parameters.
            For Momentum strategy, provide 'fast_window' and 'slow_window'.
            For other strategies, provide 'lookback_window', 'entry_threshold', 'stop_loss'.
            
            {format_instructions}
            """,
            input_variables=["strategy_type", "regime_name", "technical_summary", "asset_name"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser

        for i in range(num_proposals):
            strategy_type = random.choice(strategy_types)
            asset = random.choice(available_assets) # Simplified asset selection for now
            
            try:
                response = chain.invoke({
                    "strategy_type": strategy_type,
                    "regime_name": regime_name,
                    "technical_summary": technical_summary,
                    "asset_name": asset
                })
                
                # Map LLM output to internal params structure (this might need adjustment based on strategy type)
                # The LLM returns generic params, we might need to map them to specific strategy params
                
                params = {
                    "lookback_period": response.get("lookback_window", 20),
                    # Map other params as needed, or just pass them through if the runner supports them
                    # For now, we'll pass the raw response as params, plus the specific ones we asked for
                    **response
                }
                
                # Clean up params for specific strategies if needed
                if strategy_type == "momentum":
                     params["fast_window"] = response.get("fast_window", 20)
                     params["slow_window"] = response.get("slow_window", 50)
                
                proposal = {
                    "strategy_type": strategy_type,
                    "asset_tickers": [asset],
                    "params": params,
                    "allocation_weights": {asset: 1.0},
                    "rationale": response.get("reasoning", "Generated by AI")
                }
                proposals.append(proposal)
                
            except Exception as e:
                logger.error(f"Error generating strategy with LLM: {e}")
                # Fallback for this iteration
                fallback = generate_random_strategies(regime_data, features_df, baseline_stats, [strategy_type], [asset], 1)[0]
                proposals.append(fallback)

        return proposals

    except Exception as e:
        logger.error(f"Failed to initialize LLM agent: {e}")
        return generate_random_strategies(regime_data, features_df, baseline_stats, strategy_types, available_assets, num_proposals)



def create_langchain_agent():
    """
    Creates a LangChain-based strategy planning agent.
    Currently disabled due to missing dependencies.
    """
    logger.warning("LangChain agent creation disabled - dependencies not available")
    return None


# State management for LangChain agent (placeholder)
class AgentState:
    """State management for the strategy planning agent."""
    pass


# LangChain implementation placeholder (requires dependencies)
# When LangChain dependencies are installed, this can be enabled

"""
Advanced AI Agent Implementation (Requires LangChain)

To use the full LangChain-powered agent, install the required dependencies:
pip install langchain langchain-google-genai langgraph

Then set up your Google AI API key:
export GOOGLE_API_KEY="your-api-key-here"

The LangChain implementation would provide:
- More sophisticated strategy reasoning
- Dynamic parameter optimization based on market analysis
- Multi-step planning with state management
- Integration with external data sources
- Advanced prompt engineering for strategy generation

Example implementation:

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

class AgentState(TypedDict):
    market_data: dict
    regime_analysis: dict
    strategy_proposals: List[dict]
    current_step: str
    
def analyze_market_regime(state: AgentState) -> AgentState:
    # Analyze current market conditions
    pass
    
def generate_strategies(state: AgentState) -> AgentState:
    # Generate strategy proposals using LLM
    pass
    
def evaluate_proposals(state: AgentState) -> AgentState:
    # Evaluate and rank strategy proposals
    pass

def create_strategy_agent() -> StateGraph:
    # Create LangGraph workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_regime", analyze_market_regime)
    workflow.add_node("generate_strategies", generate_strategies)
    workflow.add_node("evaluate_proposals", evaluate_proposals)
    
    workflow.set_entry_point("analyze_regime")
    workflow.add_edge("analyze_regime", "generate_strategies")
    workflow.add_edge("generate_strategies", "evaluate_proposals")
    workflow.add_edge("evaluate_proposals", END)
    
    return workflow.compile()
"""
