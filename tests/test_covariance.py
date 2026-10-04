import numpy as np
import pytest
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.covariance import sample_covariance, ledoit_wolf_shrinkage, rmt_denoise_covariance


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(42)
    # 500 observations of 20 correlated assets
    raw = rng.normal(0, 0.015, size=(500, 20))
    # Add common market factor
    market = rng.normal(0, 0.02, size=(500, 1))
    return raw + 0.5 * market


def test_sample_covariance_psd(synthetic_returns):
    cov = sample_covariance(synthetic_returns)
    assert cov.shape == (20, 20)
    evals = np.linalg.eigvalsh(cov)
    assert np.all(evals >= -1e-10), "Sample covariance must be positive semi-definite."


def test_ledoit_wolf_shrinkage_bounds_and_conditioning(synthetic_returns):
    s_cov = sample_covariance(synthetic_returns)
    lw_cov, intensity = ledoit_wolf_shrinkage(synthetic_returns)
    
    assert 0.0 <= intensity <= 1.0, "Shrinkage intensity must lie in [0, 1]."
    assert lw_cov.shape == (20, 20)
    
    # Check condition number: Ledoit-Wolf should improve condition number (cond_shrunk <= cond_sample)
    cond_sample = np.linalg.cond(s_cov)
    cond_lw = np.linalg.cond(lw_cov)
    assert cond_lw <= cond_sample + 1e-5, "Ledoit-Wolf shrinkage must improve or match matrix conditioning."


def test_rmt_denoising_reconstruction(synthetic_returns):
    denoised_cov, stats = rmt_denoise_covariance(synthetic_returns)
    assert denoised_cov.shape == (20, 20)
    assert stats["num_signal_eigenvalues"] >= 1
    assert stats["noise_eigenvalues_count"] > 0
    
    # Check PSD
    evals = np.linalg.eigvalsh(denoised_cov)
    assert np.all(evals > 0), "Denoised covariance must be strictly positive definite."
