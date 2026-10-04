# Regime-Aware Multi-Objective Portfolio Optimizer
> **Institutional Quantitative Portfolio & Tail-Risk Optimization Engine on 50 Real S&P 500 Equities with SQLite Database Storage**

---

## 1. Problem Statement & Motivation
Mean-variance portfolio optimization (Markowitz 1952) suffers from extreme sensitivity to estimation error ("error maximization"). In high-dimensional regimes ($N=50$ assets, $T=500$ days, $N/T = 0.1$), sample covariance matrices contain large amounts of random noise that corrupt portfolio weights. Furthermore, financial asset returns exhibit non-normality, asymmetric volatility clustering (leverage effect), and macro regime shifts (calm vs. crisis).

This repository implements an end-to-end, production-grade quantitative framework that:
1. Stores 100% real S&P 500 market data in an institutional **SQLite database** (`portfolio_risk.db`).
2. Evaluates **5 Covariance & Volatility Models** (Sample Covariance, Ledoit-Wolf Shrinkage, RMT Marchenko-Pastur Denoising, GJR-GARCH Asymmetric Volatility, and 2-State Gaussian Hidden Markov Models).
3. Solves multi-objective Pareto frontiers across **Expected Return, Volatility, and Conditional Value at Risk (CVaR)** using vectorized **NSGA-II** and **MOEA/D**.
4. Conducts walk-forward rolling out-of-sample backtesting with drift-based transaction costs (10 bps) and automated regulatory-style **lookahead-bias audits**.

---

## 2. Quantitative Methodology & Mathematical Foundations

### 2.1 Covariance Estimation & Denoising
* **Sample Covariance**: $\mathbf{S} = \frac{1}{T-1} (\mathbf{X} - \bar{\mathbf{X}})^T (\mathbf{X} - \bar{\mathbf{X}})$. Benchmark baseline.
* **Ledoit-Wolf Shrinkage (2004)**: Shrinks $\mathbf{S}$ towards a constant-correlation target $\mathbf{F}$:
  $$\mathbf{\Sigma}^* = \delta^* \mathbf{F} + (1 - \delta^*) \mathbf{S}$$
  The optimal shrinkage intensity $\delta^* \in [0, 1]$ is computed analytically, guaranteeing a well-conditioned, invertible matrix.
* **Random Matrix Theory (RMT) Marchenko-Pastur Denoising**:
  Given correlation matrix $\mathbf{C} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$ and ratio $q = T / N$, random Gaussian noise is bounded by:
  $$\lambda_{\pm} = \sigma^2 \left(1 \pm \sqrt{\frac{1}{q}}\right)^2$$
  Eigenvalues $\lambda_i \le \lambda_+$ represent empirical noise. We replace noise eigenvalues with their cross-sectional mean while preserving macroeconomic and sector signal modes ($\lambda_i > \lambda_+$).

### 2.2 Volatility & Regime Dynamics
* **GJR-GARCH(1,1)**: Captures asymmetric leverage where negative market returns increase future volatility more than positive returns:
  $$\sigma_t^2 = \omega + (\alpha + \gamma \cdot \mathbb{I}_{\{\epsilon_{t-1} < 0\}}) \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
* **2-State Gaussian Hidden Markov Model (HMM)**:
  Filters unobservable market regimes (State 0: Calm/Low-Vol vs. State 1: Turbulent/High-Vol) using Baum-Welch Expectation-Maximization and forward-backward posterior inference:
  $$P(S_t = k \mid y_{1:T})$$

### 2.3 Multi-Objective Pareto Optimization
Traditional single-objective quadratic programming produces corner solutions. We formulate a true 3-objective optimization problem:
$$\min_{\mathbf{w}} \quad f_1(\mathbf{w}) = -\mathbf{\mu}^T \mathbf{w}, \quad f_2(\mathbf{w}) = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}, \quad f_3(\mathbf{w}) = \text{CVaR}_{0.95}(\mathbf{w})$$
subject to:
$$\sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le w_{\max}$$

* **NSGA-II**: Non-dominated sorting, crowding distance preservation, and simulated binary crossover (SBX).
* **MOEA/D**: Tchebycheff scalar decomposition across uniform reference weight vectors.
* **Pareto Compromise Schemes**: Uniform, Linear (Sharpe-prioritized), Sigmoid (CVaR-penalized in turbulence), and Geometric knee-point selection.

---

## 3. SQL Database Architecture (`portfolio_risk.db`)

All market inputs, regime signals, allocations, daily P&L ledgers, and audit logs are recorded transactionally:

| Table | Purpose | Primary Key |
|---|---|---|
| `market_prices` | Daily OHLCV data for 50 S&P 500 stocks + SPY (76,000+ rows) | `(date, ticker)` |
| `daily_returns` | Forward log returns | `(date, ticker)` |
| `regime_signals` | Daily posterior probabilities and GJR-GARCH volatility forecasts | `date` |
| `rebalance_cycles` | Metadata per monthly cycle (regime detected, model used, turnover) | `cycle_id` |
| `portfolio_allocations` | Exact asset weights allocated per rebalance cycle | `allocation_id` |
| `backtest_ledger` | Daily marked-to-market portfolio value, return, drawdown, and cost | `(date, strategy_name)` |
| `strategy_metrics` | Final Sharpe, Sortino, Calmar, Max Drawdown, and CVaR summaries | `strategy_name` |
| `audit_log` | Timestamped governance verification of zero lookahead bias | `audit_id` |

---

## 4. Project Structure

```
regime_portfolio_optimizer/
├── data/
│   └── portfolio_risk.db        # SQLite database (76K+ market bars, tables, views)
├── sql/
│   ├── schema.sql               # Full DDL database schema
│   └── queries.sql              # Analytical SQL queries (P&L attribution, turnover)
├── src/
│   ├── database/
│   │   ├── db.py                # SQLite DatabaseManager with transactional API
│   │   └── fetch_data.py        # Real S&P 500 downloader (100% real Yahoo Finance bars)
│   ├── models/
│   │   ├── covariance.py        # Sample, Ledoit-Wolf (2004), RMT Marchenko-Pastur
│   │   ├── volatility.py        # GJR-GARCH(1,1) asymmetric volatility MLE
│   │   └── regime.py            # 2-State Gaussian HMM with Baum-Welch EM
│   ├── optimization/
│   │   ├── cvar.py              # Value at Risk & Conditional Value at Risk
│   │   ├── objectives.py        # Multi-objective Pareto evaluation & simplex projection
│   │   └── algorithms.py        # Vectorized NSGA-II, MOEA/D, and 4 weighting schemes
│   ├── backtest/
│   │   ├── engine.py            # 500-day rolling window walk-forward backtester
│   │   ├── transaction_costs.py # Drift-based turnover & slippage model (10 bps)
│   │   └── audit.py             # Strict t-1 temporal lookahead-bias auditor
│   └── reporting/
│       ├── performance.py       # SQL metric extractor & Markdown generator
│       └── plots.py             # Headless matplotlib visualizations
├── tests/
│   ├── test_covariance.py       # PSD, condition number, RMT eigenvalues
│   ├── test_cvar.py             # CVaR >= VaR, alpha monotonicity
│   ├── test_models.py           # HMM state separation, GJR-GARCH persistence
│   └── test_lookahead.py        # Automated verification of zero lookahead leakage
├── run_pipeline.py              # Master comparative benchmark driver
└── requirements.txt
```

---

## 5. Quick Start & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
python3 -m pytest -v tests/

# 3. Execute the full end-to-end pipeline (downloads data if needed, runs all 4 backtests)
python3 run_pipeline.py

# 4. View results
cat results/STRATEGY_PERFORMANCE.md
```

---

## 6. Regulatory Audit & Lookahead-Bias Governance

To ensure institutional compliance and eliminate data snooping / lookahead bias:
1. **Strict $t-1$ Barrier**: At each rebalance date $T_{\text{reb}}$, the estimation dataset is strictly bounded to dates $t < T_{\text{reb}}$. The automated `PortfolioAuditor` asserts this condition before every single rebalance cycle.
2. **Execution Timing**: Trades execute on day $T_{\text{reb}}$, and holdings are marked to market out-of-sample until the next cycle.
3. **Turnover Realism**: Drift between rebalances is explicitly tracked ($w_{i, t}^-$), and two-way transaction costs ($10\text{ bps}$) are deducted from portfolio cash immediately.
