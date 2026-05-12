"""
pages/01_data_explorer.py  —  Data Explorer
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

st.set_page_config(page_title="Data Explorer", page_icon="🗂", layout="wide")
st.title("🗂 Data Explorer")

DATA_DIR = ROOT / "data"

# ── Load features ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading features parquet …")
def load_features():
    path = DATA_DIR / "processed" / "features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

df = load_features()

if df is None:
    st.error("features.parquet not found. Run `data/data_pipeline.py` first.")
    st.stop()

# ── Dataset overview ──────────────────────────────────────────────────────────
st.subheader("Dataset Overview")

n_dates   = df.index.get_level_values("date").nunique()
n_tickers = df.index.get_level_values("ticker").nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows",   f"{len(df):,}")
c2.metric("Trading days", f"{n_dates:,}")
c3.metric("Tickers",      f"{n_tickers:,}")
c4.metric("Features",     f"{df.shape[1]}")

date_range = df.index.get_level_values("date")
st.caption(f"Date range: {date_range.min().date()}  →  {date_range.max().date()}")

# ── Train / Val / Test timeline ───────────────────────────────────────────────
st.subheader("Walk-Forward Split Timeline")

all_dates = pd.to_datetime(df.index.get_level_values("date").unique()).sort_values()
fig, ax   = plt.subplots(figsize=(12, 1.4))
ax.axvspan(all_dates[0],              pd.Timestamp("2019-12-31"), alpha=0.4, color="#1E2761", label="Train (2yr rolling)")
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), alpha=0.4, color="#0D9488", label="Validation (6mo)")
ax.axvspan(pd.Timestamp("2022-01-01"), all_dates[-1],              alpha=0.4, color="#F97316", label="Test (3mo steps)")
ax.set_xlim(all_dates[0], all_dates[-1])
ax.set_yticks([])
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Illustrative split — actual folds roll every 3 months")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Feature explorer ──────────────────────────────────────────────────────────
st.subheader("Feature Distribution")

try:
    from features import FEATURE_GROUPS
    group_names = list(FEATURE_GROUPS.keys())
except Exception:
    group_names = []
    FEATURE_GROUPS = {}

col_left, col_right = st.columns([1, 3])
with col_left:
    if group_names:
        selected_group = st.selectbox("Feature group", group_names)
        feat_cols = [c for c in FEATURE_GROUPS.get(selected_group, []) if c in df.columns]
    else:
        feat_cols = df.columns.tolist()[:8]

    selected_ticker = st.selectbox(
        "Ticker (for time series)",
        sorted(df.index.get_level_values("ticker").unique())[:50],
    )

with col_right:
    if feat_cols:
        n_cols = 4
        n_rows = (len(feat_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
        axes = np.array(axes).flatten()
        sample = df[feat_cols].sample(min(30_000, len(df)), random_state=42)

        for i, col in enumerate(feat_cols):
            data = sample[col].dropna()
            lo, hi = data.quantile(0.01), data.quantile(0.99)
            axes[i].hist(data.clip(lo, hi), bins=50, color="#0D9488", alpha=0.75, edgecolor="none")
            axes[i].set_title(col, fontsize=8)
            axes[i].set_yticks([])
        for j in range(len(feat_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── Ticker time series ────────────────────────────────────────────────────────
st.subheader(f"Feature Time Series — {selected_ticker}")

try:
    ticker_df = df.xs(selected_ticker, level="ticker")
    num_cols  = [c for c in ticker_df.columns if ticker_df[c].dtype in [np.float32, np.float64, float]]
    chosen_feat = st.selectbox("Feature to plot", num_cols[:20])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(ticker_df.index, ticker_df[chosen_feat], color="#1E2761", lw=1)
    ax.set_title(f"{selected_ticker} — {chosen_feat}")
    ax.set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.warning(f"Could not load ticker data: {e}")

# ── Missing data ──────────────────────────────────────────────────────────────
st.subheader("Missing Data Summary")
missing = df.isnull().mean().sort_values(ascending=False)
missing_df = missing[missing > 0]
if missing_df.empty:
    st.success("No missing values.")
else:
    st.dataframe(
        (missing_df * 100).round(2).rename("Missing %").to_frame(),
        use_container_width=True,
    )
