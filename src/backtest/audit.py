"""Lookahead-Bias and Portfolio Constraint Audit Module.

Conducts automated regulatory-style integrity checks:
1. Strict t-1 temporal boundary enforcement (zero future data leakage).
2. Weight simplex constraints: sum(w) == 1.0.
3. No shorting: w_i >= 0.0.
4. Diversification cap: w_i <= max_weight.

Logs all check results into the SQLite audit_log table.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np
try:
    from database.db import DatabaseManager
except ImportError:
    from ..database.db import DatabaseManager


class PortfolioAuditor:
    """Verifies backtest and optimization pipeline integrity."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def audit_lookahead_bias(
        self,
        rebalance_date: str,
        lookback_dates: List[str] | pd.Index,
    ) -> bool:
        """Asserts that all lookback training dates are strictly before the rebalance date."""
        str_dates = [str(d)[:10] for d in lookback_dates]
        reb_date = str(rebalance_date)[:10]
        
        violations = [d for d in str_dates if d >= reb_date]
        passed = (len(violations) == 0)
        
        status = "PASS" if passed else "FAIL"
        msg = (
            f"Verified {len(str_dates)} lookback dates strictly precede rebalance date {reb_date}."
            if passed
            else f"LOOKAHEAD BIAS DETECTED! {len(violations)} dates on or after rebalance date: {violations[:3]}"
        )
        
        self.db.log_audit(
            check_name="LOOKAHEAD_BIAS_TEMPORAL_CHECK",
            status=status,
            records_checked=len(str_dates),
            violation_count=len(violations),
            message=msg,
        )
        
        if not passed:
            raise AssertionError(msg)
        return passed

    def audit_weights(
        self,
        weights: np.ndarray,
        cycle_id: int,
        max_weight: float = 0.20,
    ) -> bool:
        """Verifies mathematical validity of portfolio weights."""
        w = np.asarray(weights, dtype=np.float64)
        sum_w = float(np.sum(w))
        min_w = float(np.min(w))
        max_w = float(np.max(w))

        violations = 0
        reasons = []

        if abs(sum_w - 1.0) > 1e-4:
            violations += 1
            reasons.append(f"Sum of weights = {sum_w:.6f} != 1.0")

        if min_w < -1e-5:
            violations += 1
            reasons.append(f"Negative weight detected: min = {min_w:.6f}")

        if max_w > max_weight + 1e-4:
            violations += 1
            reasons.append(f"Max weight cap exceeded: max = {max_w:.6f} > {max_weight}")

        passed = (violations == 0)
        status = "PASS" if passed else "FAIL"
        msg = (
            f"Cycle {cycle_id}: Weights verified (Sum=1.000, Max={max_w*100:.1f}%)."
            if passed
            else f"Cycle {cycle_id} Weight violation: {'; '.join(reasons)}"
        )

        self.db.log_audit(
            check_name="PORTFOLIO_CONSTRAINTS_CHECK",
            status=status,
            records_checked=len(w),
            violation_count=violations,
            message=msg,
        )
        return passed
