"""Covariance Estimation and Denoising Models.

Implements:
1. Sample Covariance Matrix
2. Ledoit-Wolf Analytical Shrinkage (2004) to Constant Correlation Target
3. Random Matrix Theory (RMT) Marchenko-Pastur Eigenvalue Denoising (López de Prado)
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def sample_covariance(returns: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Computes the standard sample covariance matrix: S = (1/(T-1)) * (X - mu)^T (X - mu)."""
    X = np.asarray(returns)
    T, N = X.shape
    if T < 2:
        raise ValueError("At least 2 observations required for covariance.")
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    cov = np.dot(X_centered.T, X_centered) / (T - 1)
    return cov


def ledoit_wolf_shrinkage(returns: np.ndarray | pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Computes the Ledoit-Wolf (2004) optimal shrinkage estimator towards

    the constant-correlation target.
    
    Reference:
    Olivier Ledoit and Michael Wolf, "Honey, I Shrunk the Sample Covariance Matrix",
    Journal of Portfolio Management (2004).
    
    Returns:
        (shrunk_covariance, optimal_shrinkage_intensity)
    """
    X = np.asarray(returns, dtype=np.float64)
    T, N = X.shape
    if T < 3 or N < 2:
        raise ValueError("Insufficient sample size for Ledoit-Wolf shrinkage.")

    # 1. De-mean returns
    mean = np.mean(X, axis=0, keepdims=True)
    Y = X - mean

    # 2. Sample covariance
    S = np.dot(Y.T, Y) / T  # ML estimator for asymptotic derivation
    variances = np.diag(S)
    std_devs = np.sqrt(np.maximum(variances, 1e-12))

    # 3. Sample correlation matrix
    inv_std = 1.0 / std_devs
    R = S * np.outer(inv_std, inv_std)

    # 4. Target matrix F: Constant correlation model
    # Average off-diagonal correlation
    r_bar = (np.sum(R) - N) / (N * (N - 1))
    r_bar = np.clip(r_bar, -1.0 / (N - 1), 1.0)  # Ensure positive definiteness

    F = r_bar * np.outer(std_devs, std_devs)
    np.fill_diagonal(F, variances)

    # 5. Estimate pi-hat (sum of asymptotic variances of sample covariance elements)
    # y_{t,i} * y_{t,j}
    # pi_ij = (1/T) sum_t (y_ti * y_tj - s_ij)^2
    Z = np.zeros((N, N), dtype=np.float64)
    for t in range(T):
        y_t = Y[t, :]
        outer_yt = np.outer(y_t, y_t)
        Z += (outer_yt - S) ** 2
    pi_mat = Z / T
    pi_hat = np.sum(pi_mat)

    # 6. Estimate rho-hat (cross-product of variances and target deviations)
    gamma_mat = (S - F) ** 2
    gamma_hat = np.sum(gamma_mat)

    # Term for diagonal vs off-diagonal
    rho_diag = np.sum(np.diag(pi_mat))
    rho_off = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                # theta_iij = (1/T) sum_t (y_ti^2 - s_ii)(y_ti y_tj - s_ij)
                # Approximation formula per Ledoit-Wolf (2004)
                theta_iij = np.mean((Y[:, i] ** 2 - variances[i]) * (Y[:, i] * Y[:, j] - S[i, j]))
                theta_jjj = np.mean((Y[:, j] ** 2 - variances[j]) * (Y[:, i] * Y[:, j] - S[i, j]))
                term = 0.5 * (std_devs[j] / std_devs[i] * theta_iij + std_devs[i] / std_devs[j] * theta_jjj)
                rho_off += r_bar * term

    rho_hat = rho_diag + rho_off

    # 7. Optimal shrinkage intensity kappa / T
    kappa = (pi_hat - rho_hat) / max(gamma_hat, 1e-12)
    shrinkage = max(0.0, min(1.0, kappa / T))

    # 8. Shrunk covariance matrix
    # Scale back to unbiased (T / (T-1))
    sigma_star = (shrinkage * F + (1.0 - shrinkage) * S) * (T / (T - 1))
    return sigma_star, float(shrinkage)


def rmt_denoise_covariance(
    returns: np.ndarray | pd.DataFrame,
    alpha: float = 0.0,
    detone: bool = False
) -> Tuple[np.ndarray, dict]:
    """Denoises the empirical covariance matrix using Random Matrix Theory (RMT).

    Applies the Marchenko-Pastur distribution to filter noisy eigenvalues.
    
    References:
    - Laloux, Cizeau, Bouchaud, Potters (1999), "Noise Dressing of Financial Correlation Matrices".
    - Marcos López de Prado (2020), "Machine Learning for Asset Managers", Cambridge University Press.
    
    Args:
        returns: (T, N) matrix of log returns.
        alpha: Shrinkage parameter for noise eigenvalues (0.0 = replace with average).
        detone: If True, removes the first eigenvector (market mode) for residual analysis.
        
    Returns:
        (denoised_covariance, stats_dict)
    """
    X = np.asarray(returns, dtype=np.float64)
    T, N = X.shape
    q = T / N

    S = sample_covariance(X)
    variances = np.diag(S)
    std_devs = np.sqrt(np.maximum(variances, 1e-12))
    inv_std = 1.0 / std_devs
    corr = S * np.outer(inv_std, inv_std)

    # Eigen-decomposition of correlation matrix (symmetric Hermitian)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Marchenko-Pastur theoretical upper bound for noise
    # sigma_sq is estimated as average of eigenvalues belonging to noise
    # Standard iterative threshold search
    sigma_sq = 1.0  # Normalized correlation variance
    lambda_max = sigma_sq * (1.0 + np.sqrt(1.0 / q)) ** 2
    lambda_min = sigma_sq * (1.0 - np.sqrt(1.0 / q)) ** 2

    # Identify signal vs noise eigenvalues
    n_signals = int(np.sum(eigenvalues > lambda_max))
    if n_signals == 0:
        n_signals = 1  # Retain at least market mode

    clean_evals = eigenvalues.copy()
    # Denoise: Replace noise eigenvalues with their mean
    noise_evals = clean_evals[n_signals:]
    noise_mean = np.mean(noise_evals) if len(noise_evals) > 0 else 0.0
    clean_evals[n_signals:] = alpha * clean_evals[n_signals:] + (1.0 - alpha) * noise_mean

    # Detoning: Remove market-wide component if requested
    if detone:
        clean_evals[0] = 0.0

    # Reconstruct correlation matrix
    denoised_corr = np.dot(eigenvectors * clean_evals, eigenvectors.T)
    # Re-normalize diagonal to exactly 1.0
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(denoised_corr), 1e-12))
    denoised_corr = denoised_corr * np.outer(d_inv, d_inv)
    np.fill_diagonal(denoised_corr, 1.0)

    # Reconstruct covariance matrix
    denoised_cov = denoised_corr * np.outer(std_devs, std_devs)

    stats = {
        "n_assets": N,
        "n_observations": T,
        "q_ratio": float(q),
        "lambda_max_theoretical": float(lambda_max),
        "lambda_min_theoretical": float(lambda_min),
        "num_signal_eigenvalues": n_signals,
        "top_eigenvalue": float(eigenvalues[0]),
        "variance_explained_top1": float(eigenvalues[0] / N),
        "noise_eigenvalues_count": N - n_signals,
    }
    return denoised_cov, stats
