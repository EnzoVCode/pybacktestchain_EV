import numpy as np
import pandas as pd

def equal_weight_strategy(prices):
    """Allocates equal weights to all assets."""
    n = len(prices)
    weights = {ticker: 1/n for ticker in prices.index}
    return weights

def min_variance_strategy(prices, covariance_matrix):
    """Finds the minimum variance portfolio weights."""
    n = len(prices)
    ones = np.ones(n)
    try:
        inv_cov = np.linalg.inv(covariance_matrix)
        weights = inv_cov.dot(ones) / (ones.T.dot(inv_cov).dot(ones))
        weights = np.nan_to_num(weights, nan=0.0)
        return dict(zip(prices.index, weights))
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular or not invertible.")

def max_sharpe_ratio_strategy(expected_return, covariance_matrix, risk_free_rate=0.01):
    """Calculates the maximum Sharpe ratio portfolio weights."""
    n = len(expected_return)
    ones = np.ones(n)
    try:
        excess_returns = expected_return - risk_free_rate
        inv_cov = np.linalg.inv(covariance_matrix)
        weights = inv_cov.dot(excess_returns) / (ones.T.dot(inv_cov).dot(excess_returns))
        weights = np.nan_to_num(weights, nan=0.0)
        return dict(enumerate(weights))  # Return weights indexed by asset numbers
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular or not invertible.")


def risk_parity_strategy(prices, covariance_matrix):
    """Allocates weights to achieve risk parity."""
    n = len(prices)
    inv_volatility = 1 / np.sqrt(np.diag(covariance_matrix))
    weights = inv_volatility / np.sum(inv_volatility)
    weights = np.nan_to_num(weights, nan=0.0)
    return dict(zip(prices.index, weights))


def apply_strategy(strategy, prices, covariance_matrix, expected_return):
    if strategy == "Equal Weight":
        return equal_weight_strategy(prices)
    elif strategy == "Minimum Variance":
        return min_variance_strategy(prices, covariance_matrix)
    elif strategy == "Maximum Sharpe Ratio":
        return max_sharpe_ratio_strategy(expected_return, covariance_matrix)
    elif strategy == "Risk-Parity":
        return risk_parity_strategy(prices, covariance_matrix)
