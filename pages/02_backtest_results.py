"""
pages/02_backtest_results.py  —  Backtest Dashboard
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Backtest Results", page_icon="📊", layout="wide")
st.title("📊 Backtest Results")

ANNUAL_DAYS = 252
results_dir = Path(st.session_state.get("results_dir", Path("results")))

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading returns …")
def load_returns(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

@st.cache_data(show_spinner="Loading metrics …")
def load_metrics(path):
    return pd.read_csv(path)

ret_path = results_dir / "daily_returns.csv"
met_path = results_dir / "metrics.csv"

if not ret_path.exists():
    st.error("daily_returns.csv not found. Run the backtest first.")
    st.stop()

returns_full = load_returns(ret_path)
metrics = load_metrics(met_path) if met_path.exists() else None

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Chart Controls")

all_dates = returns_full.index
min_date = all_dates.min().date()
max_date = all_dates.max().date()

st.sidebar.caption(f"Available range: {min_date} → {max_date}")

# Use text inputs instead of st.date_input because the calendar widget can be hard
# to navigate for long backtest ranges in some Streamlit/browser setups.
start_text = st.sidebar.text_input("Start date (YYYY-MM-DD)", value=str(min_date))
end_text = st.sidebar.text_input("End date (YYYY-MM-DD)", value=str(max_date))

try:
    start_date = pd.to_datetime(start_text).date()
    end_date = pd.to_datetime(end_text).date()

    if start_date < min_date:
        st.sidebar.warning(f"Start date is before available data; using {min_date}.")
        start_date = min_date
    if end_date > max_date:
        st.sidebar.warning(f"End date is after available data; using {max_date}.")
        end_date = max_date
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

except Exception:
    st.sidebar.error("Please enter dates in YYYY-MM-DD format, e.g. 2020-01-01.")
    st.stop()

returns = returns_full.loc[str(start_date): str(end_date)].copy()

roll_win = st.sidebar.slider("Rolling Sharpe window (days)", 21, 126, 63, step=21)
show_long = st.sidebar.checkbox("Show long-only leg", value=False)

STRATS = {
    "uWN L/S":  ("uWN_ls",  "#4E8FD6"),
    "cWN L/S":  ("cWN_ls",  "#F97316"),
    "LSTM L/S": ("LSTM_ls", "#22C55E"),
    "SPY B&H":  ("spy",     "#94A3B8"),
}
STRATS_LONG = {
    "uWN Long":  ("uWN_long",  "#4E8FD6"),
    "cWN Long":  ("cWN_long",  "#F97316"),
    "LSTM Long": ("LSTM_long", "#22C55E"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def cum(r):
    return (1 + r.fillna(0)).cumprod()

def roll_sh(r, w):
    return r.rolling(w).mean() / (r.rolling(w).std() + 1e-9) * np.sqrt(ANNUAL_DAYS)

def drawdown(r):
    c = cum(r)
    return c / c.cummax() - 1

# ── Basic info ────────────────────────────────────────────────────────────────
st.caption(f"Results loaded from: `{results_dir}`")
st.caption(f"Displaying: {start_date} → {end_date} ({len(returns):,} trading days)")

# ── Cumulative return ─────────────────────────────────────────────────────────
st.subheader("Cumulative Return")
fig, ax = plt.subplots(figsize=(12, 4))
for label, (col, color) in STRATS.items():
    if col in returns.columns:
        lw = 1.5 if "SPY" in label else 2
        ls = "--" if "SPY" in label else "-"
        ax.plot(returns.index, cum(returns[col]), color=color, lw=lw, ls=ls, label=label)
if show_long:
    for label, (col, color) in STRATS_LONG.items():
        if col in returns.columns:
            ax.plot(returns.index, cum(returns[col]), color=color, lw=1.5, ls=":", label=label)
ax.axhline(1, color="black", lw=0.5, ls="--")
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=9)
ax.set_title("Out-of-sample walk-forward cumulative return")
st.pyplot(fig)
plt.close(fig)

# ── Rolling Sharpe + Drawdown side by side ────────────────────────────────────
c_left, c_right = st.columns(2)

with c_left:
    st.subheader(f"Rolling {roll_win}-Day Sharpe")
    fig, ax = plt.subplots(figsize=(6, 3))
    for label, (col, color) in STRATS.items():
        if col in returns.columns:
            ax.plot(returns.index, roll_sh(returns[col], roll_win), color=color, lw=1.5, label=label)
    if show_long:
        for label, (col, color) in STRATS_LONG.items():
            if col in returns.columns:
                ax.plot(returns.index, roll_sh(returns[col], roll_win), color=color, lw=1.2, ls=":", label=label)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Annualised Sharpe")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

with c_right:
    st.subheader("Drawdown")
    fig, ax = plt.subplots(figsize=(6, 3))
    for label, (col, color) in STRATS.items():
        if col in returns.columns:
            dd = drawdown(returns[col])
            ax.fill_between(returns.index, dd, 0, alpha=0.3, color=color, label=label)
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

# ── Annual returns ────────────────────────────────────────────────────────────
st.subheader("Annual Returns (%)")
ann_data = {}
for label, (col, _) in STRATS.items():
    if col in returns.columns:
        ann_data[label] = returns[col].resample("YE").apply(lambda r: (1 + r).prod() - 1) * 100
ann_df = pd.DataFrame(ann_data)
ann_df.index = ann_df.index.year

if not ann_df.empty:
    fig, ax = plt.subplots(figsize=(12, 3.5))
    x, w = np.arange(len(ann_df)), 0.2
    for i, label in enumerate(ann_df.columns):
        color = STRATS[label][1]
        offset = (i - len(ann_df.columns) / 2 + 0.5) * w
        ax.bar(x + offset, ann_df[label], width=w, color=color, alpha=0.85, label=label)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ann_df.index, rotation=45, fontsize=9)
    ax.set_ylabel("Annual Return (%)")
    ax.legend(fontsize=9)
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("No annual return data available for the selected date range.")

# ── Metrics table ─────────────────────────────────────────────────────────────
st.subheader("Performance Metrics")

# Recompute metrics for the selected date range so the table matches the chart.
def calc_metrics(r, name):
    r = r.dropna()
    if len(r) == 0:
        return None
    ann_r = r.mean() * ANNUAL_DAYS * 100
    ann_v = r.std() * np.sqrt(ANNUAL_DAYS) * 100
    sh = ann_r / (ann_v + 1e-9)
    c = cum(r)
    mdd = (c / c.cummax() - 1).min() * 100
    hit = (r > 0).mean() * 100
    return {
        "Strategy": name,
        "Ann Ret %": ann_r,
        "Ann Vol %": ann_v,
        "Sharpe": sh,
        "Max DD %": mdd,
        "Hit Rate %": hit,
    }

rows = []
for label, (col, _) in STRATS.items():
    if col in returns.columns:
        item = calc_metrics(returns[col], label)
        if item is not None:
            rows.append(item)
if show_long:
    for label, (col, _) in STRATS_LONG.items():
        if col in returns.columns:
            item = calc_metrics(returns[col], label)
            if item is not None:
                rows.append(item)

if rows:
    st.dataframe(
        pd.DataFrame(rows).set_index("Strategy").style.format({
            "Ann Ret %": "{:.1f}%",
            "Ann Vol %": "{:.1f}%",
            "Sharpe": "{:.3f}",
            "Max DD %": "{:.1f}%",
            "Hit Rate %": "{:.1f}%",
        }),
        use_container_width=True,
    )
else:
    st.info("No performance metrics available for the selected date range.")
