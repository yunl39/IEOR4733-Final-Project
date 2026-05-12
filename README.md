# IEOR4733 Final Project

## Repository Structure

```
data/
  raw/                        # Original downloads (do not edit)
    sp500_tickers.csv
    ohlcv.parquet             # OHLCV for ~500 S&P 500 stocks, 2010–2025
    signals.parquet           # SPY close + VIX close
  processed/
    features.parquet          # Feature matrix — start here for modeling

data_pipeline.py              # Run this to reproduce everything from scratch
features.py                   # Feature definitions (imported by pipeline)

model/
  uWNcWN.py                   # Unconditional and conditional model and usage example
  lorenz_test2.py              # Lorenz example in the paper with model usage example
```



