# Portfolio Optimization & Regime-Aware Risk — Interview Study Guide
> **Complete technical reference, mathematical derivations, and verbal defense for interviews at Nomura, Goldman Sachs, and quantitative hedge funds.**

---

## 1. Quick Project Elevator Pitch (30 Seconds)
*"I engineered an institutional portfolio optimization and tail-risk engine evaluated on 50 liquid S&P 500 equities from 2018 to 2023 stored in a SQLite database. I replaced fragile sample covariance with Ledoit-Wolf analytical shrinkage and Marchenko-Pastur RMT denoising. To handle market regime shifts, I built a 3-layer trigger combining a 2-State Gaussian HMM and a GJR-GARCH leverage volatility forecast to condition both the covariance estimator and a multi-objective NSGA-II/MOEA/D Pareto optimizer minimizing CVaR and volatility while maximizing return. In out-of-sample walk-forward tests across 59 rebalances, the regime-aware framework delivered superior risk-adjusted return while automated regulatory audits verified zero lookahead bias."*

---

## 2. Core Mathematical Concepts

### Concept 1: Ledoit-Wolf Analytical Shrinkage (2004)
* **The Problem**: In sample covariance $\mathbf{S} = \frac{1}{T-1} \mathbf{X}^T \mathbf{X}$, small eigenvalues are underestimated and large eigenvalues are overestimated. Inverting $\mathbf{S}$ in Markowitz optimization amplifies these estimation errors, causing extreme, erratic weights.
* **The Solution**: Shrink $\mathbf{S}$ towards a structured, well-conditioned target matrix $\mathbf{F}$ (the constant-correlation model):
  $$\mathbf{\Sigma}^* = \delta^* \mathbf{F} + (1 - \delta^*) \mathbf{S}$$
* **Target Matrix $\mathbf{F}$**:
  * Diagonal elements: Sample variances $s_{ii}$.
  * Off-diagonal elements: $f_{ij} = \bar{r} \sqrt{s_{ii} s_{jj}}$, where $\bar{r} = \frac{2}{N(N-1)} \sum_{i < j} r_{ij}$ is the average correlation.
* **Optimal Intensity $\delta^*$**: Ledoit and Wolf derive the asymptotic minimum mean-squared error (MSE) intensity:
  $$\delta^* = \max\left(0, \, \min\left(1, \, \frac{\hat{\kappa}}{T}\right)\right)$$
  Where $\hat{\kappa} = \frac{\hat{\pi} - \hat{\rho}}{\hat{\gamma}}$.
* **Interview Punchline**: *"Ledoit-Wolf is non-parametric and analytically optimal—it doesn't require cross-validation or tuning parameters. It guarantees an invertible, positive definite covariance matrix even when $N \approx T$."*

---

### Concept 2: Random Matrix Theory (RMT) Marchenko-Pastur Denoising
* **The Problem**: Much of the empirical correlation between 50 stocks is pure random historical noise rather than genuine economic co-movement.
* **Marchenko-Pastur Theorem (1967)**: For a random matrix of size $T \times N$ with $q = T/N \ge 1$ and i.i.d. variance $\sigma^2$, the theoretical eigenvalue distribution of the correlation matrix is bounded strictly inside:
  $$\lambda_{\pm} = \sigma^2 \left( 1 \pm \sqrt{\frac{1}{q}} \right)^2$$
* **Signal vs. Noise Filtering**:
  * Any eigenvalue $\lambda_i > \lambda_+$ carries true macroeconomic, market-wide, or sector signal.
  * Any eigenvalue $\lambda_i \le \lambda_+$ is statistically indistinguishable from random noise!
* **Denoising Algorithm**:
  1. Compute spectral decomposition: $\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$.
  2. For all $i$ where $\lambda_i \le \lambda_+$, replace $\lambda_i$ with the average of the noise eigenvalues: $\bar{\lambda}_{\text{noise}}$.
  3. Reconstruct $\tilde{\mathbf{C}} = \mathbf{V} \tilde{\mathbf{\Lambda}} \mathbf{V}^T$ and re-scale diagonal elements to 1.
* **Interview Punchline**: *"RMT removes noisy eigenvalue fluctuations while preserving the dominant market mode and sector clusters. During market turbulence, this prevents the optimizer from taking massive levered bets on spurious negative correlations."*

---

### Concept 3: GJR-GARCH(1,1) Volatility with Leverage Effect
* **The Problem**: Standard GARCH(1,1) assumes positive returns and negative returns impact future volatility identically ($\alpha \epsilon_{t-1}^2$). In equities, market drops trigger panic selling and leverage surges, spiking volatility much more than rallies.
* **Mathematical Formula**:
  $$\sigma_t^2 = \omega + (\alpha + \gamma \cdot \mathbb{I}_{\{\epsilon_{t-1} < 0\}}) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
  * $\mathbb{I}_{\{\epsilon_{t-1} < 0\}} = 1$ if yesterday was down, $0$ if yesterday was up.
  * $\gamma > 0$ represents the **asymmetric leverage effect**.
  * Persistence condition: $\alpha + \beta + \frac{1}{2}\gamma < 1.0$.
* **Interview Punchline**: *"I calibrated GJR-GARCH via Maximum Likelihood Estimation. When $\gamma > 0$, negative shocks immediately ratchet up future volatility forecasts, triggering risk reduction before drawdowns worsen."*

---

### Concept 4: 2-State Gaussian Hidden Markov Model (HMM)
* **The Architecture**:
  * **State 0 (Calm / Bull)**: Higher mean drift $\mu_0 > 0$, lower volatility $\sigma_0$.
  * **State 1 (Turbulent / Crisis)**: Negative mean drift $\mu_1 < 0$, high volatility $\sigma_1$.
* **Identifying Constraint**: To prevent "label switching" across rolling calibration windows, we enforce $\sigma_0 < \sigma_1$.
* **Inference**:
  * **Baum-Welch (EM)** iteratively computes forward probabilities ($\alpha_t$) and backward probabilities ($\beta_t$) to maximize observation likelihood.
  * At each rebalance date, we compute the posterior probability:
    $$P(S_t = \text{Turbulent} \mid \text{Market Returns})$$
* **Interview Punchline**: *"The HMM acts as an regime classifier. When the posterior probability of State 1 exceeds 50%, the portfolio switches from an aggressive Sharpe-seeking stance to a defensive tail-risk mitigation stance."*

---

### Concept 5: CVaR (Expected Shortfall) & Multi-Objective Pareto Optimization
* **Why CVaR instead of VaR?**
  * **VaR is NOT a coherent risk measure** because it violates **subadditivity**: $\text{VaR}(X + Y)$ can be strictly greater than $\text{VaR}(X) + \text{VaR}(Y)$ for fat-tailed distributions. It discourages diversification in extreme tails.
  * **CVaR is coherent**: It measures the conditional mean of losses beyond VaR:
    $$\text{CVaR}_\alpha(w) = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha(L)]$$
* **NSGA-II vs MOEA/D**:
  * **NSGA-II**: Uses non-dominated sorting and crowding distance to maintain diversity along the Pareto frontier.
  * **MOEA/D**: Decomposes the multi-objective problem into $K$ scalar subproblems using Tchebycheff aggregation.
* **4 Pareto Weighting Schemes**:
  * `uniform`: Equal weight along the front.
  * `linear`: Linear Sharpe weighting.
  * `sigmoid`: Non-linear sigmoid weighting penalizing CVaR during turbulence.
  * `geometric`: Exponential decay favoring the knee point of the curve.

---

### Concept 6: Lookahead-Bias Governance & Transaction Costs
* **Lookahead-Bias Audit**:
  * In quantitative finance, data snooping occurs when day $t$ weights use prices from day $t$ or later.
  * I built an automated `PortfolioAuditor` class in `audit.py`. Before every rebalance, it queries the SQLite database and asserts that $\max(\text{lookback\_dates}) < \text{rebalance\_date}$. Any leakage immediately halts the pipeline with an assertion error.
* **Turnover & Drift**:
  * Between rebalances, asset weights naturally drift:
    $$w_i(t^-) = \frac{w_i(0)(1 + R_i)}{\sum_j w_j(0)(1 + R_j)}$$
  * Turnover is calculated against drifted weights: $\frac{1}{2} \sum |w_i^{\text{new}} - w_i(t^-)|$.
  * Two-way transaction cost of **10 basis points (0.10%)** is deducted directly from cash on each rebalance.
