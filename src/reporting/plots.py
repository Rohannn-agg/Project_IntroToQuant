"""Visualization Generator for Cumulative Returns, Drawdowns, and Regimes."""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
# Headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from database.db import DatabaseManager
except ImportError:
    from ..database.db import DatabaseManager

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def plot_backtest_results(db: DatabaseManager) -> None:
    """Generates and saves performance charts to results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with db.get_connection() as conn:
        df_ledger = pd.read_sql_query("""
            SELECT date, strategy_name, portfolio_value, drawdown, regime_state
            FROM backtest_ledger
            ORDER BY date, strategy_name;
        """, conn)

    if df_ledger.empty:
        print("[-] No ledger records to plot.")
        return

    df_ledger["date"] = pd.to_datetime(df_ledger["date"])
    strategies = df_ledger["strategy_name"].unique()

    # 1. Cumulative Performance Chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    
    for strat in strategies:
        strat_data = df_ledger[df_ledger["strategy_name"] == strat]
        ax1.plot(strat_data["date"], strat_data["portfolio_value"], label=strat, lw=1.6)
        ax2.plot(strat_data["date"], strat_data["drawdown"] * 100.0, label=strat, lw=1.2)

    ax1.set_title("Out-of-Sample Portfolio Growth (2019–2023, 59 Rebalances)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (Base = 1.0)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True)

    ax2.set_title("Underwater Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown %", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = RESULTS_DIR / "01_cumulative_returns_and_drawdowns.png"
    plt.savefig(chart_path, dpi=200)
    plt.close(fig)
    print(f"[+] Performance chart saved to {chart_path}")
