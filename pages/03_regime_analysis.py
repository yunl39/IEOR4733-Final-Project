"""
pages/03_regime_analysis.py  —  Regime Analysis
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Regime Analysis", page_icon="🔍", layout="wide")
st.title("🔍 Regime Analysis")

ANNUAL_DAYS = 252
results_dir = st.session_state.get("results_dir", Path("results"))

REGIMES = {
    "Bull Market (2010–2019)": ("2010-01-01", "2019-12-31"),
    "COVID Crash (Feb–Apr 2020)": ("2020-02-01", "2020-04-30"),
    "Bear Market 2022": ("2022-01-01", "2022-12-31"),
}

LS_COLS = {
    "uWN L/S":  ("uWN_ls",  "#4E8FD6"),
    "cWN L/S":  ("cWN_ls",  "#F97316"),
    "LSTM L/S": ("LSTM_ls", "#22C55E"),
    "SPY B&H":  ("spy",     "#94A3B8"),
}

# ── Load returns ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading returns …")
def load_returns(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

ret_path = results_dir / "daily_returns.csv"
if not ret_path.exists():
    st.error("daily_returns.csv not found.")
    st.stop()
returns = load_returns(ret_path)

# ── Metrics helper ────────────────────────────────────────────────────────────
def regime_metrics(rets, start, end):
    r = rets.loc[start:end].dropna()
    if len(r) < 5:
        return {"sharpe": np.nan, "ann_ret": np.nan, "max_dd": np.nan}
    ann_r = r.mean() * ANNUAL_DAYS * 100
    ann_v = r.std()  * np.sqrt(ANNUAL_DAYS) * 100
    cum   = (1 + r).cumprod()
    mdd   = (cum / cum.cummax() - 1).min() * 100
    return {"sharpe": ann_r / (ann_v + 1e-9), "ann_ret": ann_r, "max_dd": mdd}

# ── Sharpe summary table ──────────────────────────────────────────────────────
st.subheader("Sharpe Ratio by Regime — Long-Short Strategies")

sharpe_rows = {}
for label, (col, _) in LS_COLS.items():
    if col not in returns.columns:
        continue
    sharpe_rows[label] = {
        regime: regime_metrics(returns[col], s, e)["sharpe"]
        for regime, (s, e) in REGIMES.items()
    }

sharpe_df = pd.DataFrame(sharpe_rows).T
sharpe_df.columns = [r.split("(")[0].strip() for r in sharpe_df.columns]

# Color-coded table
def color_sharpe(val):
    if pd.isna(val):   return ""
    if val > 0:        return "background-color: #DCFCE7; color: #16A34A"
    return                    "background-color: #FEE2E2; color: #DC2626"

st.dataframe(
    sharpe_df.style.applymap(color_sharpe).format("{:.3f}"),
    use_container_width=True,
)

# ── Cumulative returns by regime ──────────────────────────────────────────────
st.subheader("Cumulative Returns by Regime")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, (regime_name, (start, end)) in zip(axes, REGIMES.items()):
    slice_ = returns.loc[start:end]
    if slice_.empty:
        ax.set_title(f"{regime_name}\n(no data)")
        continue
    for label, (col, color) in LS_COLS.items():
        if col not in slice_.columns:
            continue
        cum = (1 + slice_[col].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, color=color, lw=2, label=label)
    ax.axhline(1, color="black", lw=0.5, ls="--")
    ax.set_title(regime_name, fontsize=10)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.suptitle("Cumulative Returns by Market Regime", fontsize=12)
plt.tight_layout()
st.pyplot(fig); plt.close(fig)

# ── Sharpe bar chart ──────────────────────────────────────────────────────────
st.subheader("Sharpe Ratio Comparison")
regime_labels  = list(sharpe_df.columns)
strategy_labels = list(sharpe_df.index)
colors = [v[1] for v in LS_COLS.values() if v[0] in returns.columns]

x, w = np.arange(len(regime_labels)), 0.18
fig, ax = plt.subplots(figsize=(11, 4))
for i, (strat, color) in enumerate(zip(strategy_labels, colors)):
    offset = (i - len(strategy_labels) / 2 + 0.5) * w
    ax.bar(x + offset, sharpe_df.loc[strat], width=w, color=color, alpha=0.85, label=strat)
ax.axhline(0, color="black", lw=0.7, ls="--")
ax.set_xticks(x); ax.set_xticklabels(regime_labels, fontsize=10)
ax.set_ylabel("Annualised Sharpe Ratio")
ax.legend(fontsize=9)
plt.tight_layout()
st.pyplot(fig); plt.close(fig)

# ── Load pre-computed regime CSVs if available ────────────────────────────────
st.subheader("Detailed Regime Tables")
regime_csvs = list(results_dir.glob("regime_*.csv"))
if regime_csvs:
    for f in sorted(regime_csvs):
        label = f.stem.replace("regime_", "").replace("_", " ")
        with st.expander(label):
            st.dataframe(pd.read_csv(f, index_col=0), use_container_width=True)
else:
    st.info(
        "Detailed tables not found. Run:\n"
        "```\npython robustness/regime_analysis.py "
        f"--results_dir {results_dir}\n```"
    )
