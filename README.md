data/
  raw/               ← original downloads
    sp500_tickers.csv
    ohlcv.parquet      # OHLCV for ~500 S&P 500 stocks, 2010–2025
    signals.parquet    # SPY close + VIX close (raw)
  processed/
    features.parquet   # Feature matrix — start here for modeling
    
data_pipeline.py       ← run this to reproduce everything from scratch
features.py            ← feature definitions (imported by pipeline, don't run directly)
