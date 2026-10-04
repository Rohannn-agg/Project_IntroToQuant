import numpy as np
import pytest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optimization.cvar import portfolio_var_cvar, compute_portfolio_returns


@pytest.fixture
def returns_sample():
    rng = np.random.default_rng(123)
    return rng.normal(0.0005, 0.015, size=(1000, 10))


def test_cvar_greater_than_var(returns_sample):
    weights = np.full(10, 0.10)
    for alpha in [0.90, 0.95, 0.99]:
        var, cvar = portfolio_var_cvar(returns_sample, weights, alpha=alpha)
        assert cvar >= var - 1e-8, f"Expected Shortfall (CVaR) must be >= VaR at alpha={alpha}."


def test_cvar_monotonicity_in_alpha(returns_sample):
    weights = np.full(10, 0.10)
    var90, cvar90 = portfolio_var_cvar(returns_sample, weights, alpha=0.90)
    var95, cvar95 = portfolio_var_cvar(returns_sample, weights, alpha=0.95)
    var99, cvar99 = portfolio_var_cvar(returns_sample, weights, alpha=0.99)
    
    assert var99 >= var95 >= var90, "VaR must be monotonically increasing with confidence level."
    assert cvar99 >= cvar95 >= cvar90, "CVaR must be monotonically increasing with confidence level."
