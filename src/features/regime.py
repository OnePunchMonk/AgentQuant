import pandas as pd

def detect_regime(features_df):
    """
    Detects the market regime based on simple heuristics from the latest features.
    
    Args:
        features_df (pd.DataFrame): DataFrame with computed features.
        
    Returns:
        str: The detected market regime label.
    """
    if features_df.empty:
        return "Unknown"
        
    latest = features_df.iloc[-1]
    
    vix = latest.get('vix_close', 20)  # Default to 20 if VIX is not available
    mom63d = latest.get('momentum_63d', 0)
    
    # Heuristic rules
    if vix > 30:
        if mom63d < -0.10:
            return "Crisis-Bear"
        else:
            return "HighVol-Uncertain"
    elif vix > 20 and vix <= 30:
        if mom63d > 0.05:
            return "MidVol-Bull"
        elif mom63d < -0.05:
            return "MidVol-Bear"
        else:
            return "MidVol-MeanRevert"
    else: # VIX <= 20
        if mom63d > 0.05:
            return "LowVol-Bull"
        else:
            return "LowVol-MeanRevert"
            
    return "Unknown"

if __name__ == '__main__':
    from src.data.ingest import fetch_ohlcv_data
    from src.features.engine import compute_features
    from src.utils.config import config
    
    ohlcv = fetch_ohlcv_data()
    features = compute_features(ohlcv, config['reference_asset'], config['vix_ticker'])
    
    regime = detect_regime(features)
    print(f"Latest Features:\n{features.iloc[-1]}")
    print(f"\nCurrent Detected Regime: {regime}")