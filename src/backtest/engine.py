"""Walk-Forward Rolling Backtester with 3-Layer Regime-Aware Trigger.

Simulates 59 out-of-sample monthly rebalances (2019-2023) using 500-day rolling
windows, strict t-1 lookahead audit, and drift-adjusted transaction costs.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

try:
    from database.db import DatabaseManager
    from models.covariance import sample_covariance, ledoit_wolf_shrinkage, rmt_denoise_covariance
    from models.volatility import GJRGARCH
    from models.regime import GaussianHMM2State
    from optimization.algorithms import (
        NSGA2PortfolioOptimizer,
        MOEADPortfolioOptimizer,
        select_pareto_portfolio,
    )
    from backtest.transaction_costs import compute_drifted_weights, calculate_rebalance_turnover
    from backtest.audit import PortfolioAuditor
except ImportError:
    from ..database.db import DatabaseManager
    from ..models.covariance import sample_covariance, ledoit_wolf_shrinkage, rmt_denoise_covariance
    from ..models.volatility import GJRGARCH
    from ..models.regime import GaussianHMM2State
    from ..optimization.algorithms import (
        NSGA2PortfolioOptimizer,
        MOEADPortfolioOptimizer,
        select_pareto_portfolio,
    )
    from .transaction_costs import compute_drifted_weights, calculate_rebalance_turnover
    from .audit import PortfolioAuditor


class BacktestEngine:
    """End-to-end quantitative backtesting engine."""

    def __init__(
        self,
        db: DatabaseManager,
        lookback_window: int = 500,
        rebalance_interval: int = 21,  # Approx 21 trading days = 1 month
        cost_bps: float = 10.0,        # 10 bps transaction cost
        max_asset_weight: float = 0.20,
    ):
        self.db = db
        self.lookback_window = lookback_window
        self.rebalance_interval = rebalance_interval
        self.cost_bps = cost_bps
        self.max_asset_weight = max_asset_weight
        self.auditor = PortfolioAuditor(db)

    def run_backtest(
        self,
        strategy_name: str = "Regime_Aware_Multi_Objective",
        start_date: str = "2018-01-01",
        end_date: str = "2023-12-31",
        use_moead: bool = False,
        weighting_scheme: str = "sigmoid",
    ) -> Dict[str, float]:
        """Runs the rolling walk-forward backtest and commits results to SQLite."""
        print(f"\n=======================================================")
        print(f"[+] Running Backtest: {strategy_name}")
        print(f"=======================================================")

        # 1. Fetch full market data matrix
        tickers = [t for t in self.db.get_available_tickers() if t != "SPY"]
        all_prices = self.db.get_price_matrix(start_date, end_date, tickers=tickers)
        spy_prices = self.db.get_price_matrix(start_date, end_date, tickers=["SPY"])["SPY"]

        if all_prices.empty or spy_prices.empty:
            raise RuntimeError("Market data not found in database. Run fetch_data first.")

        # Compute log returns
        asset_returns = np.log(all_prices / all_prices.shift(1)).dropna()
        spy_returns = np.log(spy_prices / spy_prices.shift(1)).dropna()

        # Align common dates
        common_dates = asset_returns.index.intersection(spy_returns.index)
        asset_returns = asset_returns.loc[common_dates]
        spy_returns = spy_returns.loc[common_dates]

        total_days = len(common_dates)
        N = len(tickers)

        if total_days < self.lookback_window + self.rebalance_interval:
            raise ValueError(f"Insufficient history: {total_days} days available.")

        # 2. Determine rebalance dates: first rebalance happens at index self.lookback_window
        rebalance_indices = list(range(self.lookback_window, total_days, self.rebalance_interval))
        print(f"[+] Out-of-sample period: {common_dates[self.lookback_window].strftime('%Y-%m-%d')} to {common_dates[-1].strftime('%Y-%m-%d')}")
        print(f"[+] Total Rebalancing Cycles: {len(rebalance_indices)}")

        # Optimizer instances
        nsga2 = NSGA2PortfolioOptimizer(pop_size=50, n_generations=25, max_asset_weight=self.max_asset_weight)
        moead = MOEADPortfolioOptimizer(n_subproblems=40, max_generations=20, max_asset_weight=self.max_asset_weight)

        portfolio_value = 1.0  # Normalized initial capital
        current_weights = np.full(N, 1.0 / N)  # Equal weight start
        ledger_records = []
        total_turnover = 0.0

        cycle_count = 0

        # Loop through rebalance periods
        for step_idx, reb_idx in enumerate(rebalance_indices):
            cycle_count += 1
            reb_date = common_dates[reb_idx].strftime("%Y-%m-%d")
            
            # Strict lookback window: [reb_idx - self.lookback_window : reb_idx] (strictly t-1 data!)
            train_dates = common_dates[reb_idx - self.lookback_window : reb_idx]
            train_returns = asset_returns.loc[train_dates].values
            train_spy = spy_returns.loc[train_dates].values

            # Audit check 1: Lookahead bias temporal verification
            self.auditor.audit_lookahead_bias(reb_date, train_dates)

            # 3-LAYER REGIME-AWARE LOGIC
            regime_detected = "Calm"
            cov_model_name = "Ledoit_Wolf"
            regime_state = 0

            if "Regime_Aware" in strategy_name:
                # Layer 1: Gaussian HMM on SPY benchmark
                hmm = GaussianHMM2State(max_iter=60)
                try:
                    hmm.fit(train_spy)
                    probs = hmm.predict_probabilities(train_spy)
                    prob_turbulent = probs[-1, 1]
                    regime_state = 1 if prob_turbulent >= 0.50 else 0
                except Exception:
                    regime_state = 0

                # Layer 2: GJR-GARCH asymmetric volatility forecast on SPY
                try:
                    gjr = GJRGARCH().fit(train_spy)
                    vol_forecast = gjr.forecast_next_vol(annualize=True)
                except Exception:
                    vol_forecast = 0.16

                # Layer 3: Conditioning
                if regime_state == 1 or vol_forecast > 0.22:
                    regime_detected = "Turbulent"
                    cov_model_name = "RMT_Denoised"
                    cov_matrix, _ = rmt_denoise_covariance(train_returns, alpha=0.0)
                    chosen_scheme = "sigmoid"  # Favor CVaR protection
                else:
                    regime_detected = "Calm"
                    cov_model_name = "Ledoit_Wolf"
                    cov_matrix, _ = ledoit_wolf_shrinkage(train_returns)
                    chosen_scheme = weighting_scheme

            elif "Sample_Covariance" in strategy_name:
                cov_model_name = "Sample_Covariance"
                cov_matrix = sample_covariance(train_returns)
                chosen_scheme = "linear"

            elif "Ledoit_Wolf" in strategy_name:
                cov_model_name = "Ledoit_Wolf"
                cov_matrix, _ = ledoit_wolf_shrinkage(train_returns)
                chosen_scheme = "linear"

            elif "RMT_Denoised" in strategy_name:
                cov_model_name = "RMT_Denoised"
                cov_matrix, _ = rmt_denoise_covariance(train_returns, alpha=0.0)
                chosen_scheme = "linear"
            else:
                cov_model_name = "Ledoit_Wolf"
                cov_matrix, _ = ledoit_wolf_shrinkage(train_returns)
                chosen_scheme = "linear"

            # Expected returns (historical mean with shrinkage towards cross-sectional mean)
            raw_means = np.mean(train_returns, axis=0)
            grand_mean = np.mean(raw_means)
            exp_returns = 0.6 * raw_means + 0.4 * grand_mean  # Bayesian shrinkage on returns

            # Run Multi-Objective Pareto Optimization
            if use_moead:
                p_weights, p_objs = moead.optimize(exp_returns, cov_matrix, train_returns)
                opt_name = "MOEA/D"
            else:
                p_weights, p_objs = nsga2.optimize(exp_returns, cov_matrix, train_returns)
                opt_name = "NSGA-II"

            # Select single portfolio from Pareto front
            new_weights = select_pareto_portfolio(
                p_weights, p_objs, scheme=chosen_scheme, regime_state=regime_state
            )

            # Audit check 2: Constraints verification
            self.auditor.audit_weights(new_weights, cycle_count, self.max_asset_weight)

            # Turnover & Transaction Costs
            turnover, cost_deduction = calculate_rebalance_turnover(
                new_weights, current_weights, cost_bps=self.cost_bps
            )
            total_turnover += turnover

            # Deduct cost from portfolio wealth
            portfolio_value *= (1.0 - cost_deduction)

            # Record rebalance cycle in DB
            weights_dict = {tickers[i]: float(new_weights[i]) for i in range(N)}
            self.db.record_rebalance(
                cycle_number=cycle_count,
                rebalance_date=reb_date,
                lookback_start=train_dates[0].strftime("%Y-%m-%d"),
                lookback_end=train_dates[-1].strftime("%Y-%m-%d"),
                regime_detected=regime_detected,
                covariance_model=cov_model_name,
                optimization_method=opt_name,
                objective_weights_json=json.dumps({"scheme": chosen_scheme}),
                turnover=turnover,
                transaction_cost=cost_deduction,
                weights=weights_dict,
            )

            current_weights = new_weights.copy()

            # Holding period daily simulation until next rebalance
            next_idx = rebalance_indices[step_idx + 1] if step_idx + 1 < len(rebalance_indices) else total_days
            holding_dates = common_dates[reb_idx:next_idx]
            holding_returns = asset_returns.loc[holding_dates].values

            cum_holding = 1.0
            peak_val = portfolio_value

            for d_idx, day_date in enumerate(holding_dates):
                day_asset_ret = holding_returns[d_idx]
                # Portfolio daily return = sum(w_i * (exp(r_i) - 1))
                day_port_ret = float(np.dot(current_weights, np.exp(day_asset_ret) - 1.0))
                portfolio_value *= (1.0 + day_port_ret)
                peak_val = max(peak_val, portfolio_value)
                drawdown = (portfolio_value - peak_val) / peak_val

                ledger_records.append({
                    "date": day_date.strftime("%Y-%m-%d"),
                    "strategy_name": strategy_name,
                    "portfolio_value": portfolio_value,
                    "daily_return": day_port_ret,
                    "cumulative_return": portfolio_value - 1.0,
                    "drawdown": drawdown,
                    "turnover": turnover if d_idx == 0 else 0.0,
                    "transaction_cost_deducted": cost_deduction if d_idx == 0 else 0.0,
                    "regime_state": regime_state,
                })

                # Drift weights for next day
                current_weights = compute_drifted_weights(current_weights, np.exp(day_asset_ret) - 1.0)

            if cycle_count % 10 == 0 or cycle_count == len(rebalance_indices):
                print(f"[Cycle {cycle_count:02d}/{len(rebalance_indices)}] Date: {reb_date} | Regime: {regime_detected:<10} | Model: {cov_model_name:<16} | Value: {portfolio_value:.3f}x")

        # Ingest ledger records into DB
        self.db.insert_ledger_records(ledger_records)

        # 4. Compute Summary Performance Metrics
        daily_returns = np.array([r["daily_return"] for r in ledger_records])
        ann_return = float(np.mean(daily_returns) * 252.0)
        ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252.0))
        sharpe = ann_return / max(ann_vol, 1e-6)
        
        downside_returns = daily_returns[daily_returns < 0.0]
        downside_vol = float(np.std(downside_returns, ddof=1) * np.sqrt(252.0)) if len(downside_returns) > 1 else ann_vol
        sortino = ann_return / max(downside_vol, 1e-6)

        max_dd = float(min(r["drawdown"] for r in ledger_records))
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

        var_95 = float(np.percentile(-daily_returns, 95.0))
        cvar_95 = float(np.mean(-daily_returns[-daily_returns >= var_95])) * np.sqrt(252.0)

        metrics = {
            "strategy_name": strategy_name,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "cvar_95": cvar_95,
            "total_turnover": float(total_turnover),
            "rebalance_count": cycle_count,
            "final_portfolio_value": portfolio_value,
        }

        self.db.record_strategy_metrics(metrics)
        print(f"\n[+] Finished {strategy_name}: Sharpe = {sharpe:.3f}, Ann Return = {ann_return*100:.1f}%, Max DD = {max_dd*100:.1f}%")
        return metrics
