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
├── data/
│   ├── raw/
│   │   ├── ohlcv.parquet             # Raw stock-level OHLCV data
│   │   ├── signals.parquet           # Raw market signal data
│   │   └── sp500_tickers.csv         # S&P 500 ticker universe
│   │
│   ├── processed/
│   │   └── features.parquet          # Processed feature dataset
│   │
│   ├── data_pipeline.py              # Data loading, cleaning, processing, and saving
│   └── features.py                   # Feature engineering for stock-level and market-level features
│
├── model/
│   ├── uWNcWN2.py                    # Main model architecture for uWN and cWN
│   ├── lorenz_test2.py               # Lorenz test and model validation
│   └── backtest_uWNcWN_LSTM2.py      # Training, validation, prediction, portfolio construction, and backtesting
│
├── robustness/
│   └── regime_analysis.py            # Regime-level performance analysis
│
├── transaction_cost/
│   ├── transactioncost.py            # Portfolio comparison with transaction cost modeling
│   ├── metrics.csv                   # Transaction-cost-adjusted performance metrics
│   ├── daily_returns.csv             # Daily strategy returns under transaction costs
│   ├── cumulative_returns.csv        # Cumulative strategy returns under transaction costs
│   ├── long_short_comparison.png     # Long-short portfolio comparison plot
│   └── long_only_comparison.png      # Long-only portfolio comparison plot
│
├── pages/
│   ├── 01_data_explorer.py           # Streamlit page for data and feature exploration
│   ├── 02_backtest_results.py        # Streamlit page for backtest result visualization
│   ├── 03_regime_analysis.py         # Streamlit page for regime analysis
│   └── 04_run_simulation.py          # Streamlit page for interactive portfolio simulation
│
├── results/
│   ├── lorenz_test/                  # Outputs from the Lorenz test
│   └── uWNcWN_LSTM2/                 # Backtesting and Regime analy sisoutputs, metrics, predictions, returns, and plots
│
├── app.py                            # Streamlit dashboard entry point
├── Main_Pipeline.ipynb               # Main notebook pipeline that coordinates all Python modules
└── README.md                         # Project documentation


