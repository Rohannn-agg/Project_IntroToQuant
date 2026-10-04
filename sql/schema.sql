-- Database Schema for Regime-Aware Multi-Objective Portfolio Optimizer
-- Database: portfolio_risk.db

PRAGMA foreign_keys = ON;

-- 1. Daily Market Price Data (Real S&P 500 constituents)
CREATE TABLE IF NOT EXISTS market_prices (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    adj_close REAL NOT NULL,
    volume INTEGER NOT NULL,
    PRIMARY KEY (date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_market_prices_date ON market_prices (date);
CREATE INDEX IF NOT EXISTS idx_market_prices_ticker ON market_prices (ticker);

-- 2. Daily Log Returns (Derived and cached)
CREATE TABLE IF NOT EXISTS daily_returns (
    date DATE NOT NULL,
    ticker TEXT NOT NULL,
    log_return REAL NOT NULL,
    PRIMARY KEY (date, ticker),
    FOREIGN KEY (date, ticker) REFERENCES market_prices(date, ticker) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_daily_returns_date ON daily_returns (date);

-- 3. Regime Signals (from 2-State Gaussian HMM)
CREATE TABLE IF NOT EXISTS regime_signals (
    date DATE PRIMARY KEY,
    state INTEGER NOT NULL,             -- 0: Calm/Bull, 1: Turbulent/Bear
    prob_calm REAL NOT NULL,            -- Probability of State 0
    prob_turbulent REAL NOT NULL,       -- Probability of State 1
    market_vol_forecast REAL NOT NULL,  -- GJR-GARCH forecasted market volatility
    regime_label TEXT NOT NULL          -- 'Calm' or 'Turbulent'
);

-- 4. Rebalance Cycles
CREATE TABLE IF NOT EXISTS rebalance_cycles (
    cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_number INTEGER NOT NULL,
    rebalance_date DATE NOT NULL,
    lookback_start DATE NOT NULL,
    lookback_end DATE NOT NULL,
    regime_detected TEXT NOT NULL,
    covariance_model TEXT NOT NULL,
    optimization_method TEXT NOT NULL,
    objective_weights TEXT NOT NULL,    -- JSON/Text of Pareto weights
    turnover REAL NOT NULL,
    transaction_cost REAL NOT NULL,
    execution_status TEXT NOT NULL DEFAULT 'COMPLETED'
);

CREATE INDEX IF NOT EXISTS idx_rebalance_date ON rebalance_cycles (rebalance_date);

-- 5. Portfolio Allocations (Weights per rebalance cycle)
CREATE TABLE IF NOT EXISTS portfolio_allocations (
    allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL,
    rebalance_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    weight REAL NOT NULL,
    FOREIGN KEY (cycle_id) REFERENCES rebalance_cycles(cycle_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_allocations_cycle ON portfolio_allocations (cycle_id);
CREATE INDEX IF NOT EXISTS idx_allocations_ticker ON portfolio_allocations (ticker);

-- 6. Backtest Daily Ledger
CREATE TABLE IF NOT EXISTS backtest_ledger (
    ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    strategy_name TEXT NOT NULL,
    portfolio_value REAL NOT NULL,
    daily_return REAL NOT NULL,
    cumulative_return REAL NOT NULL,
    drawdown REAL NOT NULL,
    turnover REAL NOT NULL,
    transaction_cost_deducted REAL NOT NULL,
    regime_state INTEGER NOT NULL,
    UNIQUE (date, strategy_name)
);

CREATE INDEX IF NOT EXISTS idx_backtest_date_strat ON backtest_ledger (date, strategy_name);

-- 7. Audit Trail Log (for Lookahead-Bias Checks & Governance)
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    check_name TEXT NOT NULL,
    status TEXT NOT NULL,               -- 'PASS' or 'FAIL'
    records_checked INTEGER NOT NULL,
    violation_count INTEGER NOT NULL,
    message TEXT NOT NULL
);

-- 8. Benchmark Strategy Summary Metrics
CREATE TABLE IF NOT EXISTS strategy_metrics (
    strategy_name TEXT PRIMARY KEY,
    annualized_return REAL NOT NULL,
    annualized_volatility REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    sortino_ratio REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    calmar_ratio REAL NOT NULL,
    cvar_95 REAL NOT NULL,
    total_turnover REAL NOT NULL,
    rebalance_count INTEGER NOT NULL
);
