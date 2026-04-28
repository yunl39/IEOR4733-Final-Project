"""
Backtest: TCN / LSTM cross-sectional equity model
=================================================
Walk-forward out-of-sample evaluation of:
    - uWN
    - cWN
    - LSTM

Strategy
--------
At each rebalancing date (daily), use the trained model to predict each
stock's next-day return. Go long top-20% and short bottom-20% by
predicted return (equal-weighted within each leg).
"""

import sys
import warnings
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(DATA_DIR))
sys.path.insert(0, str(MODEL_DIR))

from features import (
    compute_stock_features,
    compute_market_signals,
    FEATURE_GROUPS,
    ALL_STOCK_FEATURES,
    ALL_FEATURES,
)
from uWNcWN2 import UnconditionalTCN, ConditionalTCNSeparate

# ── feature column lists ───────────────────────────────────────────────────
STOCK_FEAT_COLS = ALL_STOCK_FEATURES
MARKET_FEAT_COLS = FEATURE_GROUPS["G_market_signals"]
ALL_FEAT_COLS = ALL_FEATURES

ANNUAL_DAYS = 252
TARGET_COL = "target"


# ══════════════════════════════════════════════════════════════════════════
# 0. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def load_features(max_tickers: int | None = None) -> pd.DataFrame:
    print("\n[1/4] Loading OHLCV …")
    ohlcv = pd.read_parquet(DATA_DIR / "raw" / "ohlcv.parquet")

    print("[2/4] Loading market signals …")
    sigs_raw = pd.read_parquet(DATA_DIR / "raw" / "signals.parquet")
    spy = sigs_raw["spy"]
    vix = sigs_raw["vix"]

    if max_tickers:
        tickers = sorted(ohlcv.index.get_level_values("ticker").unique())[:max_tickers]
        ohlcv = ohlcv[ohlcv.index.get_level_values("ticker").isin(tickers)]
        print(f"  Restricting to {max_tickers} tickers for speed.")

    print("[3/4] Computing stock features …")
    stock_feats = compute_stock_features(ohlcv)

    print("[4/4] Computing market signals …")
    market_signals = compute_market_signals(spy, vix)

    print("Merging & cleaning …")
    merged = stock_feats.join(market_signals, on="date", how="inner")

    close = ohlcv["Close"].unstack("ticker")
    fwd = np.log(close / close.shift(1)).shift(-1)
    fwd = fwd.stack()
    fwd.index.names = ["date", "ticker"]
    fwd.name = TARGET_COL

    merged = merged.join(fwd, how="inner")

    keep = [c for c in ALL_FEAT_COLS if c in merged.columns] + [TARGET_COL]
    merged = merged[keep]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna()

    n_dates = merged.index.get_level_values("date").nunique()
    n_tickers = merged.index.get_level_values("ticker").nunique()
    print(f"  Dataset: {merged.shape[0]:,} rows | {n_dates} dates | {n_tickers} tickers")
    return merged


# ══════════════════════════════════════════════════════════════════════════
# 2. ROLLING WINDOW GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def rolling_windows(sorted_dates, train_days=504, val_days=126, test_days=63, step_days=63):
    dates = np.asarray(sorted_dates)
    n = len(dates)
    start = 0
    while start + train_days + val_days + test_days <= n:
        tr = dates[start : start + train_days]
        vl = dates[start + train_days : start + train_days + val_days]
        te = dates[start + train_days + val_days : start + train_days + val_days + test_days]
        yield tr, vl, te
        start += step_days


# ══════════════════════════════════════════════════════════════════════════
# 3. SAMPLE BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_samples(df: pd.DataFrame, dates, feat_cols: list[str], window: int, subsample: float = 1.0):
    date_set = set(pd.Timestamp(d) for d in dates)
    tickers = df.index.get_level_values("ticker").unique()

    X_list, y_list, meta = [], [], []

    for ticker in tickers:
        try:
            grp = df.xs(ticker, level="ticker").sort_index()
        except KeyError:
            continue

        grp = grp[grp.index.isin(date_set)]
        if len(grp) < window:
            continue

        vals = grp[feat_cols].values.astype(np.float32)
        target = grp[TARGET_COL].values.astype(np.float32)

        for i in range(window - 1, len(grp)):
            x_win = vals[i - window + 1 : i + 1]
            y_t = np.array([target[i]], dtype=np.float32)

            X_list.append(x_win)
            y_list.append(y_t)
            meta.append((grp.index[i], ticker))

    if not X_list:
        return np.empty((0, window, len(feat_cols))), np.empty((0, 1)), []

    X = np.stack(X_list)
    y = np.stack(y_list)

    if subsample < 1.0:
        n = len(X)
        idx = np.random.choice(n, int(n * subsample), replace=False)
        X, y, meta = X[idx], y[idx], [meta[k] for k in idx]

    return X, y, meta


# ══════════════════════════════════════════════════════════════════════════
# 4. NORMALISATION (NO CLIP)
# ══════════════════════════════════════════════════════════════════════════

def fit_normalize(X_train):
    N, W, F = X_train.shape
    flat = X_train.reshape(-1, F)
    scaler = StandardScaler().fit(flat)
    X_norm = scaler.transform(flat)
    X_norm = X_norm.reshape(N, W, F).astype(np.float32)
    return scaler, X_norm


def apply_normalize(X, scaler):
    N, W, F = X.shape
    X_norm = scaler.transform(X.reshape(-1, F))
    return X_norm.reshape(N, W, F).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# 5. EXTRA BASELINE: LSTM
# ══════════════════════════════════════════════════════════════════════════

class LSTMBaseline(nn.Module):
    """
    Unconditional LSTM baseline.
    Input:  (batch, window, F)
    Output: (batch, 1)
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.0):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


# ══════════════════════════════════════════════════════════════════════════
# 6. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════

def _make_loader(X, y, batch_size=512, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model, X_tr, y_tr, X_vl, y_vl,
    n_iter=300, lr=3e-4, wd=1e-4, batch_size=512,
    patience=20, device="cpu",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    loader = _make_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)

    X_vl_t = torch.tensor(X_vl, dtype=torch.float32, device=device)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_wts = None
    no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(n_iter):
        model.train()
        ep_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss += loss.item() * len(xb)

        ep_loss /= len(X_tr)
        train_hist.append(ep_loss)

        with torch.no_grad():
            model.eval()
            val_pred = model(X_vl_t)
            val_loss = loss_fn(val_pred, y_vl_t).item()

        val_hist.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_wts is not None:
        model.load_state_dict(best_wts)

    return train_hist, val_hist


class ConditionalTCNWrapper:
    """
    cWN wrapper:
    - main input: stock features
    - conditions: each market feature as one separate condition series
    """
    def __init__(self, n_stock_feats, n_market_feats, **model_kwargs):
        self.n_stock = n_stock_feats
        self.n_market = n_market_feats

        self.model = ConditionalTCNSeparate(
            input_dim=n_stock_feats,
            num_conditions=n_market_feats,
            **model_kwargs,
        )

    def __call__(self, X_tensor):
        x = X_tensor[:, :, :self.n_stock]
        c_list = [
            X_tensor[:, :, self.n_stock + j : self.n_stock + j + 1]
            for j in range(self.n_market)
        ]
        return self.model(x, c_list)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, d):
        return self.model.load_state_dict(d)

    def parameters(self):
        return self.model.parameters()


def train_conditional(
    wrapper: ConditionalTCNWrapper,
    X_tr, y_tr, X_vl, y_vl,
    n_iter=300, lr=3e-4, wd=1e-4, batch_size=512,
    patience=20, device="cpu",
):
    wrapper.to(device)

    optimizer = torch.optim.Adam(wrapper.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    loader = _make_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)

    X_vl_t = torch.tensor(X_vl, dtype=torch.float32, device=device)
    y_vl_t = torch.tensor(y_vl, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_wts = None
    no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(n_iter):
        wrapper.train()
        ep_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = wrapper(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss += loss.item() * len(xb)

        ep_loss /= len(X_tr)
        train_hist.append(ep_loss)

        with torch.no_grad():
            wrapper.eval()
            val_pred = wrapper(X_vl_t)
            val_loss = loss_fn(val_pred, y_vl_t).item()

        val_hist.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_wts = {k: v.cpu().clone() for k, v in wrapper.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_wts is not None:
        wrapper.load_state_dict(best_wts)

    return train_hist, val_hist


# ══════════════════════════════════════════════════════════════════════════
# 7. PREDICTION
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_torch(model_or_wrapper, X: np.ndarray, device="cpu", batch_size=1024) -> np.ndarray:
    is_cond = isinstance(model_or_wrapper, ConditionalTCNWrapper)

    if is_cond:
        model_or_wrapper.eval()
    else:
        model_or_wrapper.eval()
        model_or_wrapper.to(device)

    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)

        if is_cond:
            out = model_or_wrapper(xb)
        else:
            out = model_or_wrapper(xb)

        preds.append(out.cpu().numpy())

    return np.concatenate(preds).reshape(-1)


# ══════════════════════════════════════════════════════════════════════════
# 8. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════

def form_portfolio(pred_df: pd.DataFrame, top_frac=0.20, bot_frac=0.20) -> pd.DataFrame:
    rows = []
    for date, group in pred_df.groupby(level="date"):
        n = len(group)
        if n < 10:
            continue

        k_top = max(1, int(n * top_frac))
        k_bot = max(1, int(n * bot_frac))

        ranked = group["pred"].rank(ascending=False)
        long_mask = ranked <= k_top
        short_mask = ranked > (n - k_bot)

        long_ret = group.loc[long_mask, "realized"].mean()
        short_ret = -group.loc[short_mask, "realized"].mean()
        ls_ret = 0.5 * long_ret + 0.5 * short_ret

        rows.append({
            "date": date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": int(long_mask.sum()),
            "n_short": int(short_mask.sum()),
        })

    return pd.DataFrame(rows).set_index("date")


# ══════════════════════════════════════════════════════════════════════════
# 9. PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(daily_returns: pd.Series, name="strategy") -> dict:
    r = daily_returns.dropna()
    ann = ANNUAL_DAYS

    cum_ret = (1 + r).prod() - 1
    ann_ret = (1 + r).prod() ** (ann / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = ann_ret / (ann_vol + 1e-9)

    roll_max = (1 + r).cumprod().cummax()
    drawdown = (1 + r).cumprod() / roll_max - 1
    max_dd = drawdown.min()
    hit_rate = (r > 0).mean()

    return {
        "name": name,
        "cum_return": round(cum_ret * 100, 2),
        "ann_return": round(ann_ret * 100, 2),
        "ann_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd * 100, 2),
        "hit_rate": round(hit_rate * 100, 2),
        "n_days": len(r),
    }


# ══════════════════════════════════════════════════════════════════════════
# 10. MAIN BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    train_days: int = 504,
    val_days: int = 126,
    test_days: int = 63,
    step_days: int = 63,
    window: int = 20,
    channels: int = 16,
    kernel_size: int = 2,
    num_layers: int = 4,
    dropout: float = 0.1,
    n_iter: int = 300,
    lr: float = 3e-4,
    wd: float = 1e-4,
    batch_size: int = 512,
    patience: int = 20,
    top_frac: float = 0.20,
    bot_frac: float = 0.20,
    subsample: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
):
    all_dates = sorted(df.index.get_level_values("date").unique())
    n_stock = len(STOCK_FEAT_COLS)
    n_market = len(MARKET_FEAT_COLS)
    n_all = n_stock + n_market

    all_preds_u, all_preds_c, all_preds_lstm = [], [], []
    fold = 0

    for tr_dates, vl_dates, te_dates in rolling_windows(
        all_dates, train_days, val_days, test_days, step_days
    ):
        fold += 1
        te_start = pd.Timestamp(te_dates[0]).strftime("%Y-%m-%d")
        te_end = pd.Timestamp(te_dates[-1]).strftime("%Y-%m-%d")
        if verbose:
            print(f"\n── Fold {fold}  test: {te_start} → {te_end} ──")

        X_tr, y_tr, _ = build_samples(df, tr_dates, ALL_FEAT_COLS, window, subsample)
        X_vl, y_vl, _ = build_samples(df, vl_dates, ALL_FEAT_COLS, window)

        if len(X_tr) < 100 or len(X_vl) < 10:
            if verbose:
                print("  Skipping – not enough samples.")
            continue

        scaler, X_tr_n = fit_normalize(X_tr)
        X_vl_n = apply_normalize(X_vl, scaler)

        y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-9
        y_tr_n = ((y_tr - y_mean) / y_std).astype(np.float32)
        y_vl_n = ((y_vl - y_mean) / y_std).astype(np.float32)

        if verbose:
            print(f"  Train: {len(X_tr):,} samples | Val: {len(X_vl):,} samples")
            print(f"  X_tr_n finite: {np.isfinite(X_tr_n).all()} | max abs: {np.max(np.abs(X_tr_n)):.4f}")
            print(f"  X_vl_n finite: {np.isfinite(X_vl_n).all()} | max abs: {np.max(np.abs(X_vl_n)):.4f}")

        # ── uWN ───────────────────────────────────────────────────────
        model_u = UnconditionalTCN(
            input_dim=n_all,
            channels=channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        t_hist_u, v_hist_u = train_model(
            model_u, X_tr_n, y_tr_n, X_vl_n, y_vl_n,
            n_iter=n_iter, lr=lr, wd=wd,
            batch_size=batch_size, patience=patience, device=device,
        )
        if verbose:
            print(f"  uWN  – trained {len(t_hist_u)} epochs | best val loss: {min(v_hist_u):.6f}")

        # ── cWN ───────────────────────────────────────────────────────
        wrapper_c = ConditionalTCNWrapper(
            n_stock_feats=n_stock,
            n_market_feats=n_market,
            channels=channels,
            kernel_size=kernel_size,
            cond_kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        t_hist_c, v_hist_c = train_conditional(
            wrapper_c, X_tr_n, y_tr_n, X_vl_n, y_vl_n,
            n_iter=n_iter, lr=lr, wd=wd,
            batch_size=batch_size, patience=patience, device=device,
        )
        if verbose:
            print(f"  cWN  – trained {len(t_hist_c)} epochs | best val loss: {min(v_hist_c):.6f}")

        # ── LSTM ──────────────────────────────────────────────────────
        model_lstm = LSTMBaseline(
            input_dim=n_all,
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
        ).to(device)

        t_hist_lstm, v_hist_lstm = train_model(
            model_lstm, X_tr_n, y_tr_n, X_vl_n, y_vl_n,
            n_iter=n_iter, lr=lr, wd=wd,
            batch_size=batch_size, patience=patience, device=device,
        )
        if verbose:
            print(f"  LSTM – trained {len(t_hist_lstm)} epochs | best val loss: {min(v_hist_lstm):.6f}")

        # ── out-of-sample predictions ────────────────────────────────
        X_te, y_te, meta_te = build_samples(df, te_dates, ALL_FEAT_COLS, window)
        if len(X_te) == 0:
            continue

        X_te_n = apply_normalize(X_te, scaler)

        pred_u = predict_torch(model_u, X_te_n, device=device)
        pred_c = predict_torch(wrapper_c, X_te_n, device=device)
        pred_lstm = predict_torch(model_lstm, X_te_n, device=device)

        pred_u = pred_u * y_std + y_mean
        pred_c = pred_c * y_std + y_mean
        pred_lstm = pred_lstm * y_std + y_mean

        realized = y_te.reshape(-1)

        dates_te = [m[0] for m in meta_te]
        tickers_te = [m[1] for m in meta_te]
        idx = pd.MultiIndex.from_arrays([dates_te, tickers_te], names=["date", "ticker"])

        all_preds_u.append(pd.DataFrame({"pred": pred_u, "realized": realized}, index=idx))
        all_preds_c.append(pd.DataFrame({"pred": pred_c, "realized": realized}, index=idx))
        all_preds_lstm.append(pd.DataFrame({"pred": pred_lstm, "realized": realized}, index=idx))

    if not all_preds_u:
        raise RuntimeError("No folds completed – check date ranges or subsample settings.")

    pred_df_u = pd.concat(all_preds_u).sort_index()
    pred_df_c = pd.concat(all_preds_c).sort_index()
    pred_df_lstm = pd.concat(all_preds_lstm).sort_index()

    port_u = form_portfolio(pred_df_u, top_frac, bot_frac)
    port_c = form_portfolio(pred_df_c, top_frac, bot_frac)
    port_lstm = form_portfolio(pred_df_lstm, top_frac, bot_frac)

    return port_u, port_c, port_lstm, pred_df_u, pred_df_c, pred_df_lstm


# ══════════════════════════════════════════════════════════════════════════
# 11. SPY BENCHMARK
# ══════════════════════════════════════════════════════════════════════════

def load_spy_returns(dates_needed) -> pd.Series:
    sigs = pd.read_parquet(DATA_DIR / "raw" / "signals.parquet")
    spy = sigs["spy"]
    spy_r = np.log(spy / spy.shift(1)).dropna()
    spy_r = spy_r[spy_r.index.isin(dates_needed)]
    spy_r.name = "SPY"
    return spy_r


# ══════════════════════════════════════════════════════════════════════════
# 12. PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_results_without_lstm(port_u, port_c, spy_ret, save_path=None):
    ret_u = port_u["ls_ret"]
    ret_c = port_c["ls_ret"]

    common = ret_u.index.intersection(ret_c.index).intersection(spy_ret.index)
    ret_u = ret_u.reindex(common)
    ret_c = ret_c.reindex(common)
    spy_r = spy_ret.reindex(common)

    cum_u = (1 + ret_u).cumprod()
    cum_c = (1 + ret_c).cumprod()
    cum_spy = (1 + spy_r).cumprod()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cum_spy.index, cum_spy.values, color="grey", lw=1.5, label="SPY")
    ax1.plot(cum_u.index, cum_u.values, lw=2, label="uWN")
    ax1.plot(cum_c.index, cum_c.values, lw=2, label="cWN")
    ax1.axhline(1, color="black", lw=0.5, ls="--")
    ax1.set_title("Cumulative Return – uWN vs cWN")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    dd_u = (1 + ret_u).cumprod() / (1 + ret_u).cumprod().cummax() - 1
    dd_c = (1 + ret_c).cumprod() / (1 + ret_c).cumprod().cummax() - 1

    ax2.plot(dd_u.index, dd_u.values, label="uWN")
    ax2.plot(dd_c.index, dd_c.values, label="cWN")
    ax2.set_title("Drawdown – uWN vs cWN")
    ax2.legend()

    plt.suptitle("Cross-Sectional Equity Backtest Comparison (without LSTM)", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_results_with_lstm(port_u, port_c, port_lstm, spy_ret, save_path=None):
    ret_u = port_u["ls_ret"]
    ret_c = port_c["ls_ret"]
    ret_lstm = port_lstm["ls_ret"]

    common = ret_u.index.intersection(ret_c.index).intersection(ret_lstm.index).intersection(spy_ret.index)
    ret_u = ret_u.reindex(common)
    ret_c = ret_c.reindex(common)
    ret_lstm = ret_lstm.reindex(common)
    spy_r = spy_ret.reindex(common)

    cum_u = (1 + ret_u).cumprod()
    cum_c = (1 + ret_c).cumprod()
    cum_lstm = (1 + ret_lstm).cumprod()
    cum_spy = (1 + spy_r).cumprod()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cum_spy.index, cum_spy.values, color="grey", lw=1.5, label="SPY")
    ax1.plot(cum_u.index, cum_u.values, lw=2, label="uWN")
    ax1.plot(cum_c.index, cum_c.values, lw=2, label="cWN")
    ax1.plot(cum_lstm.index, cum_lstm.values, lw=2, label="LSTM")
    ax1.axhline(1, color="black", lw=0.5, ls="--")
    ax1.set_title("Cumulative Return – uWN vs cWN vs LSTM")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    dd_u = (1 + ret_u).cumprod() / (1 + ret_u).cumprod().cummax() - 1
    dd_c = (1 + ret_c).cumprod() / (1 + ret_c).cumprod().cummax() - 1
    dd_lstm = (1 + ret_lstm).cumprod() / (1 + ret_lstm).cumprod().cummax() - 1

    ax2.plot(dd_u.index, dd_u.values, label="uWN")
    ax2.plot(dd_c.index, dd_c.values, label="cWN")
    ax2.plot(dd_lstm.index, dd_lstm.values, label="LSTM")
    ax2.set_title("Drawdown – uWN vs cWN vs LSTM")
    ax2.legend()

    plt.suptitle("Cross-Sectional Equity Backtest Comparison (with LSTM)", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 13. METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════

def print_metrics_table(port_u, port_c, port_lstm, spy_ret, save_dir=None):
    common = (
        port_u.index
        .intersection(port_c.index)
        .intersection(port_lstm.index)
        .intersection(spy_ret.index)
    )

    metrics = [
        compute_metrics(port_u.loc[common, "ls_ret"], "uWN (L/S)"),
        compute_metrics(port_c.loc[common, "ls_ret"], "cWN (L/S)"),
        compute_metrics(port_lstm.loc[common, "ls_ret"], "LSTM (L/S)"),
        compute_metrics(spy_ret.reindex(common), "SPY buy-hold"),
    ]

    header = f"{'Strategy':<18} {'CumRet%':>8} {'AnnRet%':>8} {'AnnVol%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'HitRate%':>9}"
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  BACKTEST RESULTS – OUT-OF-SAMPLE")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for m in metrics:
        print(f"{m['name']:<18} {m['cum_return']:>8.1f} {m['ann_return']:>8.1f} "
              f"{m['ann_vol']:>8.1f} {m['sharpe']:>7.3f} {m['max_dd']:>8.1f} {m['hit_rate']:>9.1f}")
    print(sep)

    if save_dir is not None:
        pd.DataFrame(metrics).to_csv(save_dir / "metrics.csv", index=False)

    return metrics, common


# ══════════════════════════════════════════════════════════════════════════
# 14. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="TCN/LSTM equity backtest")
    p.add_argument("--tickers", type=int, default=None)
    p.add_argument("--train_years", type=float, default=2.0)
    p.add_argument("--val_months", type=int, default=6)
    p.add_argument("--test_months", type=int, default=3)
    p.add_argument("--step_months", type=int, default=3)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--iters", type=int, default=700)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--top_frac", type=float, default=0.20)
    p.add_argument("--bot_frac", type=float, default=0.20)
    p.add_argument("--save_plot", type=str, default=None)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--results_dir", type=str, default="results_uWNcWN_LSTM2")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_days = int(args.train_years * ANNUAL_DAYS)
    val_days = int(args.val_months * ANNUAL_DAYS / 12)
    test_days = int(args.test_months * ANNUAL_DAYS / 12)
    step_days = int(args.step_months * ANNUAL_DAYS / 12)

    df = load_features(max_tickers=args.tickers)

    port_u, port_c, port_lstm, pred_df_u, pred_df_c, pred_df_lstm = run_backtest(
        df,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
        step_days=step_days,
        window=args.window,
        channels=args.channels,
        num_layers=args.layers,
        n_iter=args.iters,
        subsample=args.subsample,
        top_frac=args.top_frac,
        bot_frac=args.bot_frac,
        device=device,
    )

    all_backtest_dates = port_u.index.union(port_c.index).union(port_lstm.index)
    spy_ret = load_spy_returns(all_backtest_dates)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_metrics_table(port_u, port_c, port_lstm, spy_ret, save_dir=out_dir)

    returns_df = pd.DataFrame({
        "uWN_ls": port_u["ls_ret"],
        "cWN_ls": port_c["ls_ret"],
        "LSTM_ls": port_lstm["ls_ret"],
        "spy": spy_ret,
    }).sort_index()
    returns_df.to_csv(out_dir / "daily_returns.csv")

    cum_df = (1 + returns_df.fillna(0)).cumprod()
    cum_df.to_csv(out_dir / "cumulative_returns.csv")

    pred_df_u.to_csv(out_dir / "predictions_uWN.csv")
    pred_df_c.to_csv(out_dir / "predictions_cWN.csv")
    pred_df_lstm.to_csv(out_dir / "predictions_LSTM.csv")

    if not args.no_plot:
        if args.save_plot:
            base_path = Path(args.save_plot)
            base_path.parent.mkdir(parents=True, exist_ok=True)

            stem = base_path.stem
            suffix = base_path.suffix if base_path.suffix else ".png"

            save_path_without = base_path.with_name(f"{stem}_without_lstm{suffix}")
            save_path_with = base_path.with_name(f"{stem}_with_lstm{suffix}")
        else:
            save_path_without = out_dir / "backtest_without_lstm.png"
            save_path_with = out_dir / "backtest_with_lstm.png"

        plot_results_without_lstm(port_u, port_c, spy_ret, save_path=save_path_without)
        plot_results_with_lstm(port_u, port_c, port_lstm, spy_ret, save_path=save_path_with)


if __name__ == "__main__":
    main()