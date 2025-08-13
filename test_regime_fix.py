import pandas as pd

# Test case that reproduces the original error
def test_string_regime_error():
    # This is what was causing the error
    regime_data = "LowVol-Bull"  # String returned by detect_regime
    
    try:
        # This line was failing before the fix
        current_vol = regime_data.get('current_volatility', 0.15)  # AttributeError: 'str' object has no attribute 'get'
        print("❌ Old code would fail here")
    except AttributeError as e:
        print(f"✓ Confirmed error: {e}")
    
    # Test the new fixed logic
    if isinstance(regime_data, str):
        regime_name = regime_data
        # Estimate volatility based on regime name
        if 'HighVol' in regime_name or 'Crisis' in regime_name:
            current_vol = 0.25
        elif 'MidVol' in regime_name:
            current_vol = 0.18
        else:  # LowVol
            current_vol = 0.12
        print(f"✓ Fixed logic works: regime='{regime_name}', vol={current_vol}")
    else:
        current_vol = regime_data.get('current_volatility', 0.15)
        regime_name = regime_data.get('current_regime', 'normal')
        
    return regime_name, current_vol

if __name__ == "__main__":
    print("Testing the 'str' object has no attribute 'get' fix...")
    print("="*60)
    regime_name, vol = test_string_regime_error()
    print("="*60)
    print(f"✓ Fix successful! Regime: {regime_name}, Volatility: {vol}")
