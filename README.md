# IEOR4733 Final Project - Conditional WaveNet Portfolio Backtesting Framework

This project extends Borovykh et al. (2018), *Conditional Time Series Forecasting with Convolutional Neural Networks*, by applying the conditional WaveNet framework to a cross-sectional equity return prediction and portfolio construction setting. Building on the paper’s comparison between unconditional and conditional convolutional architectures, this project evaluates whether conditioning on market-level signals improves stock-level return forecasts and trading performance.

The project develops a complete Python-based quantitative research pipeline including:

- Data preparation and feature engineering
- uWN (unconditional WaveNet)
- cWN (conditional WaveNet)
- LSTM benchmark model
- Walk-forward out-of-sample backtesting
- Long-short and long-only portfolio construction
- Regime analysis
- Transaction cost modeling
- Interactive Streamlit dashboard

The framework is modular and designed for reproducible empirical research and portfolio evaluation.

---

# Project Structure

```text
.
├── app.py                                  # Streamlit app entry point
├── pages/
│   ├── 01_data_exploration.py
│   ├── 02_backtest_results.py
│   ├── 03_regime_analysis.py
│   └── 04_run_simulation.py
│
├── results/
│   ├── uWNcWN_LSTM2/
│   │   ├── predictions_uWN.csv
│   │   ├── predictions_cWN.csv
│   │   ├── predictions_LSTM.csv
│   │   ├── cumulative_returns.csv
│   │   ├── daily_returns.csv
│   │   ├── metrics.csv
│   │   ├── regime_Bull_Market_2010_2019.csv
│   │   ├── regime_COVID_Crash_Feb_Apr_2020.csv
│   │   ├── regime_Bear_Market_2022.csv
│   │   ├── regime_cumulative.png
│   │   └── regime_sharpe.png
│   │
│   └── transaction_cost/
│       ├── cumulative_returns.csv
│       ├── daily_returns.csv
│       ├── metrics.csv
│       ├── long_short_comparison.png
│       └── long_only_comparison.png
│
├── robustness/
│   └── regime_analysis.py
│
├── transaction_cost/
│   └── transactioncost.py
│
└── Main_Pipeline.ipynb


