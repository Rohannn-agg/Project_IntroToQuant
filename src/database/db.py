"""Database Manager for SQLite Portfolio Risk Database.

Handles schema migrations, transactional inserts, return matrix queries,
and audit logging for production-grade quantitative portfolio optimization.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "portfolio_risk.db"
SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "sql" / "schema.sql"


class DatabaseManager:
    """Encapsulates SQLite operations with connection pooling and schema management."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        """Executes DDL schema if tables do not exist."""
        if SCHEMA_PATH.exists():
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                ddl = f.read()
            with self.get_connection() as conn:
                conn.executescript(ddl)
                conn.commit()

    # --- Market Data Operations ---

    def insert_market_prices(self, df: pd.DataFrame) -> int:
        """Bulk inserts or replaces OHLCV price bars.

        Expected columns: ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        """
        required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns in price DataFrame: {missing}")

        records = [
            (
                str(row["date"])[:10],
                str(row["ticker"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["adj_close"]),
                int(row["volume"]),
            )
            for _, row in df.iterrows()
        ]

        sql = """
        INSERT OR REPLACE INTO market_prices 
        (date, ticker, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()
        return len(records)

    def insert_daily_returns(self, df: pd.DataFrame) -> int:
        """Inserts calculated daily log returns.
        Columns: ['date', 'ticker', 'log_return']
        """
        records = [
            (str(row["date"])[:10], str(row["ticker"]), float(row["log_return"]))
            for _, row in df.iterrows()
        ]
        sql = "INSERT OR REPLACE INTO daily_returns (date, ticker, log_return) VALUES (?, ?, ?);"
        with self.get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()
        return len(records)

    def get_price_matrix(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None,
        price_col: str = "adj_close",
    ) -> pd.DataFrame:
        """Returns pivoted matrix of prices: Index=date, Columns=ticker."""
        with self.get_connection() as conn:
            if tickers:
                placeholders = ",".join(["?"] * len(tickers))
                query = f"""
                SELECT date, ticker, {price_col} as price 
                FROM market_prices 
                WHERE date >= ? AND date <= ? AND ticker IN ({placeholders})
                ORDER BY date, ticker;
                """
                params = [start_date, end_date] + list(tickers)
            else:
                query = f"""
                SELECT date, ticker, {price_col} as price 
                FROM market_prices 
                WHERE date >= ? AND date <= ?
                ORDER BY date, ticker;
                """
                params = [start_date, end_date]

            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        pivoted = df.pivot(index="date", columns="ticker", values="price")
        pivoted.index = pd.to_datetime(pivoted.index)
        return pivoted

    def get_return_matrix(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Returns wide DataFrame of daily log returns: Index=date, Columns=ticker."""
        prices = self.get_price_matrix(start_date, end_date, tickers, price_col="adj_close")
        if prices.empty:
            return pd.DataFrame()
        # Compute forward log returns
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        return returns

    def get_available_tickers(self) -> List[str]:
        """Lists all distinct tickers in market_prices."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT DISTINCT ticker FROM market_prices ORDER BY ticker;")
            return [row[0] for row in cur.fetchall()]

    def get_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """Returns (min_date, max_date) present in market_prices."""
        with self.get_connection() as conn:
            cur = conn.execute("SELECT MIN(date), MAX(date) FROM market_prices;")
            row = cur.fetchone()
            return (row[0], row[1]) if row else (None, None)

    # --- Regime Signals ---

    def insert_regime_signals(self, signals_df: pd.DataFrame) -> None:
        """Records regime signals (date, state, prob_calm, prob_turbulent, market_vol_forecast, regime_label)."""
        records = [
            (
                str(row["date"])[:10],
                int(row["state"]),
                float(row["prob_calm"]),
                float(row["prob_turbulent"]),
                float(row["market_vol_forecast"]),
                str(row["regime_label"]),
            )
            for _, row in signals_df.iterrows()
        ]
        sql = """
        INSERT OR REPLACE INTO regime_signals 
        (date, state, prob_calm, prob_turbulent, market_vol_forecast, regime_label)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.executemany(sql, records)
            conn.commit()

    # --- Rebalance Cycles & Allocations ---

    def record_rebalance(
        self,
        cycle_number: int,
        rebalance_date: str,
        lookback_start: str,
        lookback_end: str,
        regime_detected: str,
        covariance_model: str,
        optimization_method: str,
        objective_weights_json: str,
        turnover: float,
        transaction_cost: float,
        weights: Dict[str, float],
    ) -> int:
        """Atomically inserts rebalance cycle and its constituent asset weights."""
        cycle_sql = """
        INSERT INTO rebalance_cycles 
        (cycle_number, rebalance_date, lookback_start, lookback_end, regime_detected,
         covariance_model, optimization_method, objective_weights, turnover, transaction_cost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            cur = conn.execute(
                cycle_sql,
                (
                    cycle_number,
                    rebalance_date,
                    lookback_start,
                    lookback_end,
                    regime_detected,
                    covariance_model,
                    optimization_method,
                    objective_weights_json,
                    turnover,
                    transaction_cost,
                ),
            )
            cycle_id = cur.lastrowid

            alloc_sql = """
            INSERT INTO portfolio_allocations (cycle_id, rebalance_date, ticker, weight)
            VALUES (?, ?, ?, ?);
            """
            alloc_records = [
                (cycle_id, rebalance_date, ticker, float(w))
                for ticker, w in weights.items()
                if abs(w) > 1e-6
            ]
            conn.executemany(alloc_sql, alloc_records)
            conn.commit()
            return cycle_id

    # --- Backtest Ledger & Metrics ---

    def insert_ledger_records(self, records: List[Dict[str, Any]]) -> None:
        """Bulk inserts daily backtest ledger rows."""
        sql = """
        INSERT OR REPLACE INTO backtest_ledger 
        (date, strategy_name, portfolio_value, daily_return, cumulative_return, drawdown, turnover, transaction_cost_deducted, regime_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        rows = [
            (
                r["date"],
                r["strategy_name"],
                float(r["portfolio_value"]),
                float(r["daily_return"]),
                float(r["cumulative_return"]),
                float(r["drawdown"]),
                float(r["turnover"]),
                float(r["transaction_cost_deducted"]),
                int(r["regime_state"]),
            )
            for r in records
        ]
        with self.get_connection() as conn:
            conn.executemany(sql, rows)
            conn.commit()

    def record_strategy_metrics(self, metrics: Dict[str, Any]) -> None:
        """Stores strategy summary statistics."""
        sql = """
        INSERT OR REPLACE INTO strategy_metrics 
        (strategy_name, annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, cvar_95, total_turnover, rebalance_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.execute(
                sql,
                (
                    metrics["strategy_name"],
                    metrics["annualized_return"],
                    metrics["annualized_volatility"],
                    metrics["sharpe_ratio"],
                    metrics["sortino_ratio"],
                    metrics["max_drawdown"],
                    metrics["calmar_ratio"],
                    metrics["cvar_95"],
                    metrics["total_turnover"],
                    metrics["rebalance_count"],
                ),
            )
            conn.commit()

    # --- Audit Trail ---

    def log_audit(
        self,
        check_name: str,
        status: str,
        records_checked: int,
        violation_count: int,
        message: str,
    ) -> None:
        """Appends audit verification result to audit_log table."""
        sql = """
        INSERT INTO audit_log (check_name, status, records_checked, violation_count, message)
        VALUES (?, ?, ?, ?, ?);
        """
        with self.get_connection() as conn:
            conn.execute(sql, (check_name, status, records_checked, violation_count, message))
            conn.commit()
