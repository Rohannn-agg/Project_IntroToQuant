"""Master Execution Driver for Regime-Aware Multi-Objective Portfolio Optimizer.

Executes end-to-end walk-forward comparative experiments on 50 real S&P 500 stocks
(2019-2023, 59 rebalances) and records all outputs in SQLite database.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database.db import DatabaseManager
from backtest.engine import BacktestEngine
from reporting.performance import generate_performance_report
from reporting.plots import plot_backtest_results


def main():
    print("=" * 70)
    print("  QUANT PORTFOLIO RISK & MULTI-OBJECTIVE OPTIMIZATION ENGINE")
    print("  Universe: 50 Real S&P 500 Stocks + SPY (2018-2023)")
    print("  Database: SQLite (data/portfolio_risk.db)")
    print("=" * 70)

    db = DatabaseManager()
    
    # Check data readiness
    min_date, max_date = db.get_date_range()
    tickers = db.get_available_tickers()
    print(f"\n[+] Database Connected: {db.db_path.name}")
    print(f"[+] Date Range: {min_date} to {max_date} | Tickers: {len(tickers)}")

    if not tickers or len(tickers) < 10:
        print("[-] Data not found! Please run python3 src/database/fetch_data.py first.")
        sys.exit(1)

    # Initialize Engine (500-day rolling window, 21-day monthly rebalance, 10 bps cost)
    engine = BacktestEngine(
        db=db,
        lookback_window=500,
        rebalance_interval=21,
        cost_bps=10.0,
        max_asset_weight=0.20,
    )

    t0 = time.time()

    # 1. Benchmark 1: Sample Covariance + NSGA-II
    engine.run_backtest(
        strategy_name="Sample_Covariance_NSGA2",
        start_date="2018-01-01",
        end_date="2023-12-31",
    )

    # 2. Benchmark 2: Ledoit-Wolf Shrinkage (Static) + NSGA-II
    engine.run_backtest(
        strategy_name="Ledoit_Wolf_Static",
        start_date="2018-01-01",
        end_date="2023-12-31",
    )

    # 3. Benchmark 3: RMT Marchenko-Pastur Denoising (Static) + NSGA-II
    engine.run_backtest(
        strategy_name="RMT_Denoised_Static",
        start_date="2018-01-01",
        end_date="2023-12-31",
    )

    # 4. Core Proposed Model: 3-Layer Regime-Aware Trigger (HMM + GJR-GARCH + RMT/Ledoit-Wolf)
    engine.run_backtest(
        strategy_name="Regime_Aware_Multi_Objective",
        start_date="2018-01-01",
        end_date="2023-12-31",
        weighting_scheme="sigmoid",
    )

    elapsed = time.time() - t0
    print(f"\n[+] All 4 backtest strategies completed in {elapsed:.1f}s.")

    # 5. Generate Performance Reports & Visualizations
    report_df = generate_performance_report(db)
    plot_backtest_results(db)

    print("\n" + "=" * 70)
    print("  FINAL OUT-OF-SAMPLE PERFORMANCE SUMMARY (2019-2023, 59 REBALANCES)")
    print("=" * 70)
    print(report_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
