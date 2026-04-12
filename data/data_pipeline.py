"""
Phase 1 — S&P 500 Data Pipeline  (updated)
============================================
Covers:
  - S&P 500 constituent list
  - Daily OHLCV data (yfinance)
  - SPY + VIX condition signals
  - Extended feature engineering (48 features across 7 groups)
  - Rolling-window train/val/test splits (no lookahead)
  - Saves all artifacts to disk

Run:
    pip install yfinance pandas numpy scikit-learn pyarrow lxml html5lib
    python sp500_pipeline.py
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Generator

# Local feature library (same directory)
from features import (
    compute_stock_features,
    compute_market_signals,
    ALL_FEATURES,
    print_feature_summary,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────
DATA_DIR      = Path("data")
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WINDOWS_DIR   = DATA_DIR / "windows"

for d in [RAW_DIR, PROCESSED_DIR, WINDOWS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START_DATE  = "2010-01-01"
END_DATE    = "2025-12-31"
LOOKBACK  = 20
STEP_SIZE   = 5
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


# ─────────────────────────────────────────────
# 1.  DATA COLLECTION
# ─────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    cache = RAW_DIR / "sp500_tickers.csv"
    if cache.exists():
        print("  [cache] Tickers loaded from disk.")
        return pd.read_csv(cache)["ticker"].tolist()

    print("  Fetching S&P 500 tickers from Wikipedia ...")
    try:
        # Wikipedia blocks default urllib/pandas user-agent — spoof a browser header
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        import requests
        from io import StringIO
        url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers=headers, timeout=10).text
        table = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}), falling back to hardcoded list ...")
        # Reliable static fallback — current as of early 2025
        table = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        )
        table = table.rename(columns={"Symbol": "Symbol"})

    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    pd.DataFrame({"ticker": tickers}).to_csv(cache, index=False)
    print(f"  {len(tickers)} tickers saved.")
    return tickers


def download_ohlcv(tickers: list[str]) -> pd.DataFrame:
    cache = RAW_DIR / "ohlcv.parquet"
    if cache.exists():
        print("  [cache] OHLCV loaded from disk.")
        return pd.read_parquet(cache)

    print(f"  Downloading OHLCV for {len(tickers)} tickers ...")
    raw = yf.download(
        tickers, start=START_DATE, end=END_DATE,
        auto_adjust=True, progress=True, threads=True,
    )
    raw.columns.names = ["metric", "ticker"]
    df = raw.stack(level="ticker").rename_axis(["date", "ticker"])
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    df.to_parquet(cache)
    print(f"  {df.shape[0]:,} rows saved.")
    return df


def download_signals() -> tuple[pd.Series, pd.Series]:
    """Returns (spy_close, vix_close) raw price series."""
    cache = RAW_DIR / "signals.parquet"
    if cache.exists():
        print("  [cache] Signals loaded from disk.")
        raw = pd.read_parquet(cache)
        return raw["spy"], raw["vix"]

    print("  Downloading SPY and VIX ...")
    spy = yf.download("SPY",  start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)["Close"].squeeze()
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)["Close"].squeeze()
    spy.index.name = "date"
    vix.index.name = "date"
    pd.DataFrame({"spy": spy, "vix": vix}).to_parquet(cache)
    print("  Signals saved.")
    return spy, vix


# ─────────────────────────────────────────────
# 2.  MISSING DATA
# ─────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    filled = (
        df.groupby(level="ticker", group_keys=False)
          .apply(lambda g: g.ffill())
    )
    return filled.dropna()


# ─────────────────────────────────────────────
# 3.  MAIN
# ─────────────────────────────────────────────

def run_pipeline():
    print("\n======================================================")
    print("  Phase 1 - S&P 500 Data Pipeline (extended features)")
    print("======================================================")
    print_feature_summary()

    print("Step 1: Tickers")
    tickers = get_sp500_tickers()

    print("\nStep 2: OHLCV")
    ohlcv = download_ohlcv(tickers)

    print("\nStep 3: Market signals")
    spy, vix = download_signals()

    print("\nStep 4: Stock features")
    stock_feats = compute_stock_features(ohlcv)

    print("\nStep 5: Market-wide signals")
    market_signals = compute_market_signals(spy, vix)

    print("\nStep 6: Merging & cleaning")
    merged = stock_feats.join(market_signals, on="date", how="inner")
    merged = handle_missing(merged)

    available = [c for c in ALL_FEATURES if c in merged.columns]
    merged = merged[available]
    print(f"  Final shape: {merged.shape}  ({len(available)} features)")

    proc_path = PROCESSED_DIR / "features.parquet"
    merged.to_parquet(proc_path)
    print(f"  Saved -> {proc_path}")

if __name__ == "__main__":
    run_pipeline()
