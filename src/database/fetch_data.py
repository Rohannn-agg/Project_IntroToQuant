"""Real Market Data Downloader for S&P 500 Constituents.

Fetches 100% real OHLCV daily data directly from Yahoo Finance for 50 liquid
S&P 500 constituents across all 11 GICS sectors (2018-01-01 to 2023-12-31)
and stores them into the SQLite database.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime
import sys
from pathlib import Path

# Allow execution as standalone script
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from database.db import DatabaseManager

# 50 S&P 500 stocks spanning all 11 GICS sectors + SPY benchmark
SP50_TICKERS = [
    # Information Technology (10)
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "CSCO", "ADBE", "CRM",
    # Financials (7)
    "JPM", "BAC", "WFC", "MS", "GS", "BLK", "AXP",
    # Health Care (7)
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO",
    # Consumer Discretionary (5)
    "TSLA", "HD", "MCD", "NKE", "SBUX",
    # Consumer Staples (5)
    "PG", "KO", "PEP", "COST", "WMT",
    # Industrials (5)
    "CAT", "GE", "UNP", "HON", "BA",
    # Energy (4)
    "XOM", "CVX", "COP", "SLB",
    # Communication Services (3)
    "CMCSA", "DIS", "NFLX",
    # Utilities, Real Estate & Materials (4)
    "NEE", "DUK", "PLD", "LIN"
]

BENCHMARK_TICKER = "SPY"


def fetch_yahoo_chart_data(
    ticker: str,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31"
) -> Optional[pd.DataFrame]:
    """Downloads daily OHLCV from Yahoo Finance API via HTTP request."""
    dt_start = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    dt_end = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={dt_start}&period2={dt_end}&interval=1d&events=history"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        indicators = result["indicators"]
        quote = indicators["quote"][0]
        adjclose = indicators.get("adjclose", [{}])[0].get("adjclose", quote["close"])
        
        df = pd.DataFrame({
            "date": [datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps],
            "ticker": ticker,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adj_close": adjclose,
            "volume": quote["volume"],
        }).dropna()
        
        return df
    except Exception as e:
        print(f"[-] Failed to fetch {ticker}: {e}")
        return None


def populate_market_data(
    db: DatabaseManager,
    tickers: List[str] = SP50_TICKERS,
    include_benchmark: bool = True,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31"
) -> int:
    """Fetches and inserts real historical market data for all tickers into SQLite."""
    all_tickers = (tickers + [BENCHMARK_TICKER]) if include_benchmark else tickers
    total_bars = 0
    print(f"[+] Starting market data download for {len(all_tickers)} tickers ({start_date} to {end_date})...")
    
    for i, ticker in enumerate(all_tickers, 1):
        df = fetch_yahoo_chart_data(ticker, start_date, end_date)
        if df is not None and not df.empty:
            count = db.insert_market_prices(df)
            total_bars += count
            print(f"[{i}/{len(all_tickers)}] {ticker}: Ingested {count} daily bars.")
        else:
            print(f"[{i}/{len(all_tickers)}] {ticker}: Warning - No data returned.")
        time.sleep(0.15)  # Rate limiting
        
    print(f"[+] Market data ingestion complete. Total rows in database: {total_bars}")
    return total_bars


if __name__ == "__main__":
    db = DatabaseManager()
    populate_market_data(db)
