"""2-State Gaussian Hidden Markov Model (HMM) for Financial Regime Detection.

Implements Expectation-Maximization (Baum-Welch algorithm) and Forward-Backward
filtering to detect market regimes:
    State 0: Low-volatility / Calm regime (bull market dynamics)
    State 1: High-volatility / Turbulent regime (crisis / bear market dynamics)

Enforces the identifying constraint sigma_0 < sigma_1 to eliminate label switching.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


class GaussianHMM2State:
    """Two-state Gaussian Hidden Markov Model for univariate financial returns."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
        
        # Model parameters
        self.pi = np.array([0.7, 0.3])                     # Initial state probabilities
        self.A = np.array([[0.95, 0.05], [0.10, 0.90]])    # Transition probability matrix
        self.means = np.array([0.0005, -0.001])            # Emission means
        self.stds = np.array([0.008, 0.022])               # Emission standard deviations
        self.is_fitted = False

    def _gaussian_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Univariate Gaussian probability density function with numerical guard."""
        std = max(std, 1e-8)
        var = std ** 2
        denom = np.sqrt(2.0 * np.pi * var)
        numer = np.exp(-0.5 * ((x - mean) ** 2) / var)
        return np.maximum(numer / denom, 1e-300)

    def fit(self, observations: np.ndarray) -> "GaussianHMM2State":
        """Fits HMM parameters via the Baum-Welch (EM) algorithm."""
        y = np.asarray(observations, dtype=np.float64)
        T = len(y)
        if T < 50:
            raise ValueError("At least 50 observations required to fit 2-state HMM.")

        # Data-driven parameter initialization
        overall_mean = np.mean(y)
        overall_std = np.std(y, ddof=1)
        self.means = np.array([overall_mean + 0.2 * overall_std, overall_mean - 0.5 * overall_std])
        self.stds = np.array([0.7 * overall_std, 1.6 * overall_std])

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # 1. Emission probabilities B: shape (T, 2)
            B = np.zeros((T, 2), dtype=np.float64)
            B[:, 0] = self._gaussian_pdf(y, self.means[0], self.stds[0])
            B[:, 1] = self._gaussian_pdf(y, self.means[1], self.stds[1])

            # 2. Forward pass with scaling
            alpha = np.zeros((T, 2), dtype=np.float64)
            c = np.zeros(T, dtype=np.float64)

            alpha[0] = self.pi * B[0]
            c[0] = np.sum(alpha[0])
            alpha[0] /= max(c[0], 1e-300)

            for t in range(1, T):
                alpha[t] = np.dot(alpha[t - 1], self.A) * B[t]
                c[t] = np.sum(alpha[t])
                alpha[t] /= max(c[t], 1e-300)

            log_likelihood = np.sum(np.log(np.maximum(c, 1e-300)))

            # Check convergence
            if np.abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood

            # 3. Backward pass with same scaling
            beta = np.zeros((T, 2), dtype=np.float64)
            beta[-1] = 1.0

            for t in range(T - 2, -1, -1):
                beta[t] = np.dot(self.A, (B[t + 1] * beta[t + 1]))
                beta[t] /= max(c[t + 1], 1e-300)

            # 4. Posterior state probabilities (gamma) and transitions (xi)
            gamma = alpha * beta
            gamma /= np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300)

            xi = np.zeros((T - 1, 2, 2), dtype=np.float64)
            for t in range(T - 1):
                numerator = self.A * np.outer(alpha[t], B[t + 1] * beta[t + 1])
                denom = max(c[t + 1], 1e-300)
                xi[t] = numerator / denom

            # 5. M-step: Update parameters
            self.pi = gamma[0] / np.sum(gamma[0])

            sum_xi = np.sum(xi, axis=0)
            sum_gamma_prev = np.sum(gamma[:-1], axis=0, keepdims=True).T
            self.A = sum_xi / np.maximum(sum_gamma_prev, 1e-300)
            self.A /= np.sum(self.A, axis=1, keepdims=True)

            sum_gamma = np.sum(gamma, axis=0)
            for k in range(2):
                w_k = gamma[:, k]
                self.means[k] = np.sum(w_k * y) / max(sum_gamma[k], 1e-300)
                self.stds[k] = np.sqrt(np.sum(w_k * ((y - self.means[k]) ** 2)) / max(sum_gamma[k], 1e-300))

            # 6. Identifying constraint: Ensure State 0 is low volatility (Calm)
            if self.stds[0] > self.stds[1]:
                # Swap states 0 and 1
                self.stds = self.stds[::-1]
                self.means = self.means[::-1]
                self.pi = self.pi[::-1]
                self.A = self.A[::-1, ::-1]

        self.is_fitted = True
        return self

    def predict_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Returns posterior state probabilities: shape (T, 2) [P(Calm), P(Turbulent)]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        y = np.asarray(observations, dtype=np.float64)
        T = len(y)

        B = np.zeros((T, 2), dtype=np.float64)
        B[:, 0] = self._gaussian_pdf(y, self.means[0], self.stds[0])
        B[:, 1] = self._gaussian_pdf(y, self.means[1], self.stds[1])

        alpha = np.zeros((T, 2), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)
        alpha[0] = self.pi * B[0]
        c[0] = np.sum(alpha[0])
        alpha[0] /= max(c[0], 1e-300)

        for t in range(1, T):
            alpha[t] = np.dot(alpha[t - 1], self.A) * B[t]
            c[t] = np.sum(alpha[t])
            alpha[t] /= max(c[t], 1e-300)

        beta = np.zeros((T, 2), dtype=np.float64)
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = np.dot(self.A, (B[t + 1] * beta[t + 1]))
            beta[t] /= max(c[t + 1], 1e-300)

        gamma = alpha * beta
        gamma /= np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300)
        return gamma

    def get_regime_summary(self) -> Dict[str, float]:
        """Returns summary of estimated regime parameters."""
        return {
            "calm_mean_annualized": float(self.means[0] * 252.0),
            "calm_vol_annualized": float(self.stds[0] * np.sqrt(252.0)),
            "turbulent_mean_annualized": float(self.means[1] * 252.0),
            "turbulent_vol_annualized": float(self.stds[1] * np.sqrt(252.0)),
            "transition_calm_to_calm": float(self.A[0, 0]),
            "transition_turbulent_to_turbulent": float(self.A[1, 1]),
            "expected_duration_calm_days": float(1.0 / max(1.0 - self.A[0, 0], 1e-4)),
            "expected_duration_turbulent_days": float(1.0 / max(1.0 - self.A[1, 1], 1e-4)),
        }
