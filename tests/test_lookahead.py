import numpy as np
import pytest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.db import DatabaseManager
from backtest.audit import PortfolioAuditor


def test_lookahead_auditor_catches_future_leakage(tmp_path):
    test_db_path = tmp_path / "test_audit.db"
    db = DatabaseManager(db_path=test_db_path)
    auditor = PortfolioAuditor(db)

    # Valid strictly historical dates
    valid_dates = ["2020-01-02", "2020-01-03", "2020-01-04"]
    rebalance_date = "2020-01-05"
    assert auditor.audit_lookahead_bias(rebalance_date, valid_dates) is True

    # Invalid dates containing future or same-day data
    invalid_dates = ["2020-01-02", "2020-01-03", "2020-01-05"]  # 2020-01-05 is same day
    with pytest.raises(AssertionError, match="LOOKAHEAD BIAS DETECTED"):
        auditor.audit_lookahead_bias(rebalance_date, invalid_dates)


def test_portfolio_auditor_weight_constraints(tmp_path):
    test_db_path = tmp_path / "test_audit.db"
    db = DatabaseManager(db_path=test_db_path)
    auditor = PortfolioAuditor(db)

    # Valid simplex weights
    valid_w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assert auditor.audit_weights(valid_w, cycle_id=1, max_weight=0.25) is True

    # Invalid: does not sum to 1
    invalid_sum = np.array([0.2, 0.2, 0.2])
    assert auditor.audit_weights(invalid_sum, cycle_id=2) is False

    # Invalid: negative weight
    invalid_neg = np.array([1.2, -0.2])
    assert auditor.audit_weights(invalid_neg, cycle_id=3) is False
