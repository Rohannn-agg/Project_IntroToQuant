"""GJR-GARCH(1,1) Asymmetric Volatility Modeling.

Implements Maximum Likelihood Estimation (MLE) of the GJR-GARCH(1,1) process:
    r_t = mu + eps_t
    sigma_t^2 = omega + (alpha + gamma * I_{eps_{t-1} < 0}) * eps_{t-1}^2 + beta * sigma_{t-1}^2

Captures the leverage effect where negative returns cause higher volatility spikes
than positive returns of the same magnitude.

Reference:
    Glosten, Jagannathan, Runkle (1993), "On the Relation between the Expected Value
    and the Volatility of the Nominal Excess Return on Stocks", Journal of Finance.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize


class GJRGARCH:
    """GJR-GARCH(1,1) model fitted via Maximum Likelihood Estimation (MLE)."""

    def __init__(self):
        self.params: Dict[str, float] = {}
        self.is_fitted = False
        self.conditional_vol: np.ndarray = np.array([])
        self.last_resid = 0.0
        self.last_var = 0.0

    def fit(self, returns: np.ndarray, annualize_factor: float = 252.0) -> "GJRGARCH":
        """Fits parameters [mu, omega, alpha, gamma, beta] on a series of log returns."""
        r = np.asarray(returns, dtype=np.float64)
        T = len(r)
        if T < 30:
            raise ValueError("At least 30 observations required to fit GJR-GARCH.")

        sample_var = float(np.var(r, ddof=1))
        sample_mean = float(np.mean(r))

        # Initial parameter guesses: [mu, omega, alpha, gamma, beta]
        # Starting with typical financial time-series values
        init_params = np.array([
            sample_mean,
            0.05 * sample_var,
            0.05,
            0.10,
            0.80
        ], dtype=np.float64)

        # Bounds: omega > 0, alpha >= 0, gamma >= 0, beta >= 0
        bounds = [
            (None, None),          # mu
            (1e-8, sample_var),    # omega
            (1e-6, 0.5),           # alpha
            (0.0, 0.5),            # gamma
            (0.2, 0.98),           # beta
        ]

        def neg_log_likelihood(params: np.ndarray) -> float:
            mu, omega, alpha, gamma, beta = params

            # Stationarity constraint penalty: alpha + beta + 0.5 * gamma < 1.0
            persistence = alpha + beta + 0.5 * gamma
            if persistence >= 0.9999 or omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
                return 1e10

            eps = r - mu
            sigma_sq = np.zeros(T, dtype=np.float64)
            sigma_sq[0] = sample_var

            for t in range(1, T):
                indicator = 1.0 if eps[t - 1] < 0.0 else 0.0
                sigma_sq[t] = omega + (alpha + gamma * indicator) * (eps[t - 1] ** 2) + beta * sigma_sq[t - 1]
                if sigma_sq[t] <= 1e-12:
                    sigma_sq[t] = 1e-12

            # Gaussian log-likelihood: -0.5 * sum(ln(2*pi) + ln(sigma^2) + eps^2 / sigma^2)
            nll = 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma_sq) + (eps ** 2) / sigma_sq)
            return float(nll) if np.isfinite(nll) else 1e10

        res = minimize(
            neg_log_likelihood,
            init_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-7}
        )

        mu, omega, alpha, gamma, beta = res.x
        self.params = {
            "mu": float(mu),
            "omega": float(omega),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "beta": float(beta),
            "persistence": float(alpha + beta + 0.5 * gamma),
        }

        # Filter out final conditional variances
        eps = r - mu
        sigma_sq = np.zeros(T, dtype=np.float64)
        sigma_sq[0] = sample_var
        for t in range(1, T):
            ind = 1.0 if eps[t - 1] < 0.0 else 0.0
            sigma_sq[t] = omega + (alpha + gamma * ind) * (eps[t - 1] ** 2) + beta * sigma_sq[t - 1]

        self.conditional_vol = np.sqrt(sigma_sq) * np.sqrt(annualize_factor)
        self.last_resid = float(eps[-1])
        self.last_var = float(sigma_sq[-1])
        self.is_fitted = True
        return self

    def forecast_next_vol(self, annualize: bool = True) -> float:
        """Forecasts 1-step ahead annualized volatility."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before forecasting.")

        ind = 1.0 if self.last_resid < 0.0 else 0.0
        var_next = (
            self.params["omega"]
            + (self.params["alpha"] + self.params["gamma"] * ind) * (self.last_resid ** 2)
            + self.params["beta"] * self.last_var
        )
        vol_next = np.sqrt(max(var_next, 1e-12))
        return float(vol_next * np.sqrt(252.0) if annualize else vol_next)
