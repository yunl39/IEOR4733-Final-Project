"""
app.py  —  Streamlit home page
Run from project root: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title  = "TCN Equity Strategy",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

ROOT = Path(__file__).resolve().parent

# Auto-detect results folder (latest uWNcWN results)
results_options = sorted(
    [d for d in (ROOT / "results").iterdir() if d.is_dir()
     and (d / "daily_returns.csv").exists()],
    key=lambda d: d.stat().st_mtime, reverse=True,
)

if results_options:
    results_labels = [d.name for d in results_options]
    chosen = st.sidebar.selectbox("Results folder", results_labels)
    results_dir = ROOT / "results" / chosen
else:
    results_dir = ROOT / "results"
    st.sidebar.warning("No results found yet.")

st.session_state["results_dir"] = results_dir
st.session_state["root"]        = ROOT

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("📈 Conditional WaveNet — Equity Strategy Dashboard")
st.markdown(
    """
    Reproduction and extension of **Borovykh et al. (2018)** —
    *Conditional time series forecasting with convolutional neural networks.*

    Cross-sectional long-short equity strategy on the S&P 500,
    comparing **uWN**, **cWN**, and **LSTM** models.
    """
)

# Status cards
c1, c2, c3, c4 = st.columns(4)

feat_path = ROOT / "data" / "processed" / "features.parquet"
ret_path  = results_dir / "daily_returns.csv"
met_path  = results_dir / "metrics.csv"
reg_path  = results_dir / "regime_Bull_Market_2010-2019.csv"

c1.metric("Features parquet",   "✅ Ready"   if feat_path.exists() else "❌ Missing")
c2.metric("Daily returns",      "✅ Ready"   if ret_path.exists()  else "❌ Missing")
c3.metric("Metrics table",      "✅ Ready"   if met_path.exists()  else "❌ Missing")
c4.metric("Regime analysis",    "✅ Ready"   if reg_path.exists()  else "⚠️ Run regime_analysis.py")

st.markdown("---")

# Quick metrics snapshot
if ret_path.exists() and met_path.exists():
    ret = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    met = pd.read_csv(met_path)

    st.subheader("Quick Snapshot — Out-of-Sample Performance")
    cols = st.columns(len(met))
    for col, (_, row) in zip(cols, met.iterrows()):
        col.metric(
            label    = row["name"],
            value    = f"{row['sharpe']:.3f}",
            delta    = f"Ann Ret {row['ann_return']:.1f}%",
            help     = f"Max DD: {row['max_dd']:.1f}%  |  Hit Rate: {row['hit_rate']:.1f}%",
        )

st.markdown("---")
st.caption(f"Results loaded from: `{results_dir}`")
