import pandas as pd
import logging

logger = logging.getLogger(__name__)

def select_best_proposal(results_df: pd.DataFrame, risk_config: dict):
    """
    Selects the best performing proposal that adheres to risk constraints.

    Args:
        results_df (pd.DataFrame): DataFrame containing backtest results for all proposals.
        risk_config (dict): A dictionary with risk parameters like 'max_drawdown'.

    Returns:
        pd.Series: The row of the best performing proposal, or None if no proposal is valid.
    """
    if results_df.empty:
        logger.warning("Received an empty DataFrame of results. Cannot select a proposal.")
        return None

    max_drawdown_limit = risk_config.get('max_drawdown', 0.20)
    
    logger.info(f"Applying risk policy: Max Drawdown < {max_drawdown_limit:.2%}")

    # The 'Max Drawdown [%]' from vectorbt is positive, so we use a direct comparison.
    # We also handle the case where the column name from vectorbt might differ slightly.
    drawdown_col = next((col for col in results_df.columns if 'drawdown' in col.lower()), None)
    if not drawdown_col:
        logger.error("Could not find a drawdown column in the results DataFrame.")
        return None

    # Filter out any proposals that violate the max drawdown constraint.
    # Note: vectorbt stats report drawdown as a positive number.
    risk_compliant_proposals = results_df[results_df[drawdown_col] < max_drawdown_limit * 100].copy()

    if risk_compliant_proposals.empty:
        logger.warning("No proposals met the risk criteria. The baseline may have been too risky or all proposals were poor.")
        return None
    
    # From the compliant proposals, select the one with the highest Sharpe Ratio
    sharpe_col = next((col for col in results_df.columns if 'sharpe' in col.lower()), None)
    if not sharpe_col:
        logger.error("Could not find a Sharpe Ratio column in the results DataFrame.")
        return None
        
    best_proposal = risk_compliant_proposals.loc[risk_compliant_proposals[sharpe_col].idxmax()]
    
    logger.info(f"Selected best proposal '{best_proposal.name}' with Sharpe Ratio: {best_proposal[sharpe_col]:.2f}")
    
    return best_proposal