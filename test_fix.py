#!/usr/bin/env python3
"""
Test script to verify both strategy planners work with string regime data.
"""

import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simple_planner():
    """Test the simple planner with string regime data."""
    try:
        from src.agent.simple_planner import generate_strategy_proposals
        
        # Test data
        regime_data = 'LowVol-Bull'  # String format from detect_regime
        features_df = pd.DataFrame({'test': [1, 2, 3]})
        baseline_stats = pd.Series({'sharpe': 1.0})
        strategy_types = ['momentum', 'mean_reversion']
        available_assets = ['SPY', 'QQQ']
        
        proposals = generate_strategy_proposals(
            regime_data, features_df, baseline_stats, 
            strategy_types, available_assets, 2
        )
        
        print(f"✓ Simple planner: Generated {len(proposals)} proposals")
        print(f"  First proposal type: {proposals[0]['strategy_type']}")
        return True
        
    except Exception as e:
        print(f"✗ Simple planner failed: {e}")
        return False

def test_langchain_planner():
    """Test the langchain planner with string regime data."""
    try:
        from src.agent.langchain_planner import generate_strategy_proposals
        
        # Test data
        regime_data = 'HighVol-Uncertain'  # String format from detect_regime
        features_df = pd.DataFrame({'test': [1, 2, 3]})
        baseline_stats = pd.Series({'sharpe': 1.0})
        strategy_types = ['momentum', 'volatility']
        available_assets = ['SPY', 'QQQ', 'TLT']
        
        proposals = generate_strategy_proposals(
            regime_data, features_df, baseline_stats, 
            strategy_types, available_assets, 3
        )
        
        print(f"✓ LangChain planner: Generated {len(proposals)} proposals")
        print(f"  First proposal type: {proposals[0]['strategy_type']}")
        return True
        
    except Exception as e:
        print(f"✗ LangChain planner failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing strategy planners with string regime data...")
    print("="*50)
    
    simple_ok = test_simple_planner()
    langchain_ok = test_langchain_planner()
    
    print("="*50)
    if simple_ok and langchain_ok:
        print("✓ All tests passed! The 'str' object error should be fixed.")
    else:
        print("✗ Some tests failed. Check the error messages above.")
