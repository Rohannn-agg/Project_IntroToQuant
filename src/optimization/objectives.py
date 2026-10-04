"""Portfolio Multi-Objective Function and Simplex Projection.

Defines the 3 conflicting objectives:
1. Maximize Return (Minimize -Return)
2. Minimize Volatility (Variance / Standard Deviation under Sigma)
3. Minimize Tail Risk (95% Conditional Value at Risk)
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from .cvar import portfolio_var_cvar


def project_to_simplex(v: np.ndarray, max_weight: float = 0.20) -> np.ndarray:
    """Projects arbitrary real vector v onto the probability simplex:

    sum(w) = 1, 0 <= w_i <= max_weight.
    
    Uses Wang & Carreira-Perpinan (2013) projection algorithm.
    """
    v = np.maximum(v, 0.0)
    s = np.sum(v)
    if s <= 1e-12:
        # Uniform fallback
        w = np.full_like(v, 1.0 / len(v))
        return w
    
    w = v / s
    # Apply individual max weight cap iteratively if specified
    if max_weight < 1.0 and np.any(w > max_weight):
        for _ in range(10):
            excess = np.maximum(w - max_weight, 0.0)
            total_excess = np.sum(excess)
            if total_excess <= 1e-6:
                break
            w = np.minimum(w, max_weight)
            uncapped = (w < max_weight)
            if np.any(uncapped):
                w[uncapped] += total_excess / np.sum(uncapped)
            else:
                break
        w = w / np.sum(w)
    return w


def evaluate_objectives(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    historical_returns: np.ndarray,
    cvar_alpha: float = 0.95,
) -> Tuple[float, float, float]:
    """Computes the 3 objective values for a candidate portfolio:

    f1: -Expected Return (to minimize)
    f2: Portfolio Volatility (to minimize)
    f3: CVaR (to minimize)
    """
    w = np.asarray(weights, dtype=np.float64)
    # 1. Expected Return (annualized)
    mu_p = float(np.dot(expected_returns, w) * 252.0)
    f1 = -mu_p  # Minimize negative return
    
    # 2. Portfolio Volatility (annualized)
    var_p = float(np.dot(w.T, np.dot(cov_matrix, w)) * 252.0)
    f2 = np.sqrt(max(var_p, 1e-12))
    
    # 3. CVaR (daily tail loss * sqrt(252))
    _, cvar_daily = portfolio_var_cvar(historical_returns, w, alpha=cvar_alpha)
    f3 = float(cvar_daily * np.sqrt(252.0))
    
    return f1, f2, f3
