"""Drift-Based Portfolio Turnover and Transaction Cost Accounting.

Computes exact asset price drift between rebalance cycles and applies
proportional transaction costs (slippage + commissions) on turnover.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


def compute_drifted_weights(
    initial_weights: np.ndarray,
    period_returns: np.ndarray,
) -> np.ndarray:
    """Computes drifted portfolio weights at the end of a holding period.

    w_i(t^-) = w_i(0) * (1 + R_i) / sum_j [ w_j(0) * (1 + R_j) ]
    """
    w0 = np.asarray(initial_weights, dtype=np.float64)
    R = np.asarray(period_returns, dtype=np.float64)
    
    growth = w0 * (1.0 + R)
    total_wealth = np.sum(growth)
    if total_wealth <= 1e-12:
        return np.full_like(w0, 1.0 / len(w0))
    return growth / total_wealth


def calculate_rebalance_turnover(
    target_weights: np.ndarray,
    current_or_drifted_weights: np.ndarray,
    cost_bps: float = 10.0,
) -> Tuple[float, float]:
    """Calculates half-turnover: 0.5 * sum |w_target - w_drifted|

    and total transaction cost deduction.
    
    Args:
        target_weights: Desired portfolio weights after rebalance.
        current_or_drifted_weights: Pre-rebalance drifted portfolio weights.
        cost_bps: One-way transaction cost in basis points (10 bps = 0.0010 = 0.10%).
        
    Returns:
        (turnover_percentage, transaction_cost_deduction)
    """
    w_target = np.asarray(target_weights, dtype=np.float64)
    w_drifted = np.asarray(current_or_drifted_weights, dtype=np.float64)

    turnover = float(0.5 * np.sum(np.abs(w_target - w_drifted)))
    cost_rate = cost_bps * 1e-4  # Convert bps to decimal
    # Two-way cost on turnover
    cost_deduction = float(turnover * cost_rate * 2.0)
    return turnover, cost_deduction
