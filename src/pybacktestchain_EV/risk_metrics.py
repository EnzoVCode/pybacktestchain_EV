import numpy as np
import pandas as pd

def calculate_var(returns, portfolio_value, confidence_level):
    """
    Calculate Value-at-Risk (VaR) in absolute monetary value using historical simulation.

    Parameters:
        returns (pd.Series): Series of portfolio returns.
        portfolio_value (float): Current portfolio value.
        confidence_level (float): Confidence level for VaR.

    Returns:
        float: VaR value in monetary terms.
    """
    if returns.isnull().any():
        returns = returns.dropna()
    relative_var = np.percentile(returns, (1 - confidence_level) * 100)
    return portfolio_value * relative_var

def calculate_expected_shortfall(returns, portfolio_value, confidence_level):
    """
    Calculate Expected Shortfall (ES) in absolute monetary value using historical simulation.

    Parameters:
        returns (pd.Series): Series of portfolio returns.
        portfolio_value (float): Current portfolio value.
        confidence_level (float): Confidence level for ES.

    Returns:
        float: Expected Shortfall value in monetary terms.
    """
    if returns.isnull().any():
        returns = returns.dropna()
    var = np.percentile(returns, (1 - confidence_level) * 100)
    relative_es = returns[returns <= var].mean()
    return portfolio_value * relative_es
