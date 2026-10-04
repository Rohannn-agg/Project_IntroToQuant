"""Conditional Value at Risk (CVaR) and Tail Risk Metrics.

Implements empirical Value at Risk (VaR) and Expected Shortfall / CVaR
for portfolio return distributions.

Reference:
    Rockafellar and Uryasev (2000), "Optimization of Conditional Value-at-Risk",
    Journal of Risk.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


def compute_portfolio_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Computes time series of portfolio log/simple returns: R_p = X * w."""
    X = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return np.dot(X, w)


def portfolio_var_cvar(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95
) -> Tuple[float, float]:
    """Computes empirical VaR and CVaR (Expected Shortfall) as positive loss numbers.

    Args:
        returns: (T, N) matrix of asset returns.
        weights: (N,) vector of portfolio weights.
        alpha: Confidence level (e.g. 0.95 or 0.99).
        
    Returns:
        (VaR_alpha, CVaR_alpha) expressed as positive loss percentages.
    """
    port_ret = compute_portfolio_returns(returns, weights)
    losses = -port_ret  # Loss is negative of return
    
    # Empirical quantile
    var = float(np.percentile(losses, alpha * 100.0))
    tail_losses = losses[losses >= var]
    
    cvar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var
    return max(0.0, var), max(0.0, cvar)


def cvar_subgradient(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95
) -> np.ndarray:
    """Computes the subgradient of CVaR with respect to weights w.

    grad_w CVaR = - (1 / (T * (1 - alpha))) sum_{t: L_t >= VaR} X_t
    """
    X = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    T, N = X.shape
    
    port_ret = np.dot(X, w)
    losses = -port_ret
    var = np.percentile(losses, alpha * 100.0)
    
    mask = (losses >= var)
    if not np.any(mask):
        return np.zeros(N)
    
    # Subgradient is the negative average of asset returns on tail loss days
    grad = -np.mean(X[mask, :], axis=0)
    return grad
