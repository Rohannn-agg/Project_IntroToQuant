-- Analytical SQL Queries for Portfolio Risk & Governance
-- Database: portfolio_risk.db

-- 1. Daily P&L and Drawdown Summary by Strategy
-- SELECT date, strategy_name, portfolio_value, daily_return, drawdown 
-- FROM backtest_ledger 
-- ORDER BY date DESC, strategy_name;

-- 2. Performance Comparison Across All Tested Strategies
SELECT 
    strategy_name,
    ROUND(annualized_return * 100, 2) || '%' AS ann_return,
    ROUND(annualized_volatility * 100, 2) || '%' AS ann_vol,
    ROUND(sharpe_ratio, 3) AS sharpe,
    ROUND(sortino_ratio, 3) AS sortino,
    ROUND(max_drawdown * 100, 2) || '%' AS max_drawdown,
    ROUND(cvar_95 * 100, 2) || '%' AS cvar_95,
    rebalance_count
FROM strategy_metrics
ORDER BY sharpe_ratio DESC;

-- 3. Top 10 Asset Allocations in the Latest Rebalance Cycle
SELECT 
    r.cycle_number,
    r.rebalance_date,
    r.regime_detected,
    r.covariance_model,
    a.ticker,
    ROUND(a.weight * 100, 2) AS weight_pct
FROM portfolio_allocations a
JOIN rebalance_cycles r ON a.cycle_id = r.cycle_id
WHERE r.cycle_id = (SELECT MAX(cycle_id) FROM rebalance_cycles)
  AND a.weight > 0.001
ORDER BY a.weight DESC
LIMIT 10;

-- 4. Regime Shift History and Impact on Portfolio Turnover
SELECT 
    cycle_number,
    rebalance_date,
    regime_detected,
    covariance_model,
    ROUND(turnover * 100, 2) || '%' AS turnover_pct,
    ROUND(transaction_cost, 4) AS cost_incurred
FROM rebalance_cycles
ORDER BY cycle_number;

-- 5. Audit Trail Verification: Lookahead-Bias & Data Integrity Status
SELECT 
    check_name,
    status,
    records_checked,
    violation_count,
    message,
    timestamp
FROM audit_log
ORDER BY audit_id DESC;

-- 6. Sector & Asset Concentration Check
SELECT 
    r.cycle_number,
    r.rebalance_date,
    COUNT(CASE WHEN a.weight > 0.005 THEN 1 END) AS active_assets_count,
    ROUND(MAX(a.weight) * 100, 2) AS max_single_weight_pct,
    ROUND(SUM(a.weight * a.weight), 4) AS herfindahl_hirschman_index
FROM portfolio_allocations a
JOIN rebalance_cycles r ON a.cycle_id = r.cycle_id
GROUP BY r.cycle_id, r.rebalance_date
ORDER BY r.cycle_number;
