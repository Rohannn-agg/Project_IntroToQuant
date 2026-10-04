import numpy as np
import pytest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.regime import GaussianHMM2State
from models.volatility import GJRGARCH


def test_hmm_2state_fit_and_separation():
    rng = np.random.default_rng(999)
    # Generate mixture: 300 calm days (low vol), 100 turbulent days (high vol)
    calm = rng.normal(0.001, 0.008, 300)
    turb = rng.normal(-0.002, 0.025, 100)
    series = np.concatenate([calm, turb, calm])

    hmm = GaussianHMM2State(max_iter=50).fit(series)
    assert hmm.is_fitted is True
    # Verify identifying constraint: stds[0] < stds[1]
    assert hmm.stds[0] < hmm.stds[1], "State 0 must have lower standard deviation than State 1."

    probs = hmm.predict_probabilities(series)
    assert probs.shape == (len(series), 2)
    # Probabilities must sum to 1 across rows
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_gjr_garch_fit_and_forecast():
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0004, 0.015, 300)

    gjr = GJRGARCH().fit(returns)
    assert gjr.is_fitted is True
    assert gjr.params["omega"] > 0
    assert gjr.params["alpha"] >= 0
    assert gjr.params["beta"] >= 0
    assert gjr.params["gamma"] >= 0

    vol_forecast = gjr.forecast_next_vol(annualize=True)
    assert vol_forecast > 0, "Forecasted annualized volatility must be strictly positive."
