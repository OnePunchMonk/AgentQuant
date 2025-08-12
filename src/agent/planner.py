import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd

# This is a placeholder for the real tool. The LLM will learn to call this.
# The actual backtesting is done elsewhere; this just defines the interface for the LLM.
def backtest_tool(strategy_name: str, asset_ticker: str, fast_window: int, slow_window: int) -> dict:
    """
    Defines the interface for a backtest tool. 
    The LLM will be trained to call this function signature.
    The actual execution of the backtest is handled by the agent runner.

    Args:
        strategy_name (str): The name of the strategy to test (e.g., 'momentum').
        asset_ticker (str): The ticker symbol to run the backtest on (e.g., 'SPY').
        fast_window (int): The shorter lookback window for the strategy.
        slow_window (int): The longer lookback window for the strategy.

    Returns:
        A dictionary confirming the parameters to be tested.
    """
    # This function is a "tool" for the LLM. It doesn't actually run the backtest here.
    # It just provides the structure that the LLM needs to request a backtest run.
    return {
        "tool_name": "backtest",
        "args": {
            "strategy_name": strategy_name,
            "asset_ticker": asset_ticker,
            "params": {
                "fast_window": fast_window,
                "slow_window": slow_window,
            },
        },
    }

def get_llm_planner():
    """Initializes and returns the Gemini Pro model with tool configuration."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
    
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name='gemini-pro',
        tools=[backtest_tool] # Provide the tool function to the model
    )
    return model

def generate_prompt(regime, features_summary, baseline_stats):
    """Creates the detailed prompt for the Gemini planner."""
    
    prompt = f"""
    You are an expert quantitative trading research assistant. Your goal is to propose alternative strategy parameters to improve performance in the current market environment.

    **Current Market Analysis:**
    - **Detected Regime:** {regime}
    - **Key Market Features (latest data):**
    {features_summary}

    **Baseline Strategy Performance:**
    - **Strategy:** Momentum (Dual Moving Average Crossover) on SPY
    - **Baseline Parameters:** fast_window=21, slow_window=63
    - **Baseline Backtest Metrics:**
    {baseline_stats.to_string()}

    **Your Task:**
    Based on the current '{regime}' regime, propose exactly FIVE alternative sets of parameters for the 'momentum' strategy that might improve the Sharpe Ratio. 
    Consider the features: for example, in a high volatility or crisis regime, shorter lookback windows might be more responsive. In a stable, trending bull market, longer windows might be better to reduce noise.

    Call the `backtest_tool` for each of your five proposals. Do not propose the same parameters as the baseline.
    Provide a brief, one-sentence rationale for each proposal before you call the tool.
    """
    return prompt

def propose_actions(regime: str, features_df: pd.DataFrame, baseline_stats: pd.Series):
    """
    Uses the Gemini planner to propose new backtest actions.
    
    Returns:
        list: A list of dictionaries, where each dict describes a backtest to be run.
    """
    planner = get_llm_planner()
    
    # Summarize features for the prompt
    features_summary = features_df.iloc[-1][[
        'volatility_21d', 'momentum_63d', 'price_vs_sma63', 'vix_close'
    ]].round(3).to_string()

    prompt = generate_prompt(regime, features_summary, baseline_stats)
    
    print("\n----- Sending Prompt to Gemini Planner -----")
    print(prompt)
    
    response = planner.generate_content(prompt)
    
    proposals = []
    try:
        for tool_call in response.candidates[0].content.parts:
            if tool_call.function_call:
                args = tool_call.function_call.args
                proposal = {
                    "strategy_name": args['strategy_name'],
                    "asset_ticker": args['asset_ticker'],
                    "params": {
                        "fast_window": args['fast_window'],
                        "slow_window": args['slow_window']
                    }
                }
                proposals.append(proposal)
    except (AttributeError, IndexError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"LLM raw response: {response.text}")
        return []

    print("\n----- Received Proposals from Gemini -----")
    if not proposals:
        print("The LLM did not return any valid tool calls.")
    else:
        for p in proposals:
            print(p)
            
    return proposals