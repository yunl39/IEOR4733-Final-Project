import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── paths ───────────────────────────────────────────────────────────────
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

ANNUAL_DAYS = 252
TARGET_COL = "target"

STOCK_FEAT_COLS = ALL_STOCK_FEATURES
MARKET_FEAT_COLS = FEATURE_GROUPS["G_market_signals"]
ALL_FEAT_COLS = ALL_FEATURES


# ════════════════════════════════════════════════════════════════════════
# 1. Load features exactly like backtest
# ════════════════════════════════════════════════════════════════════════
def load_features(max_tickers: int | None = None) -> pd.DataFrame:
    print("[1/4] Loading OHLCV …")
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


# ════════════════════════════════════════════════════════════════════════
# 2. Same rolling windows as backtest
# ════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════
# 3. Build samples with metadata
# ════════════════════════════════════════════════════════════════════════
def build_samples_with_meta(df: pd.DataFrame, dates, feat_cols: list[str], window: int):
    """
    Build rolling-window samples and keep metadata.

    Returns
    -------
    X : (N, window, F)
    y : (N, 1)
    sample_meta : list of dict
    """
    date_set = set(pd.Timestamp(d) for d in dates)
    tickers = df.index.get_level_values("ticker").unique()

    X_list, y_list, sample_meta = [], [], []

    for ticker in tickers:
        try:
            grp = df.xs(ticker, level="ticker").sort_index()
        except KeyError:
            continue

        grp = grp[grp.index.isin(date_set)]
        if len(grp) < window:
            continue

        vals = grp[feat_cols].values.astype(np.float64)
        target = grp[TARGET_COL].values.astype(np.float64)
        idx = grp.index.to_list()

        for i in range(window - 1, len(grp)):
            x_win = vals[i - window + 1 : i + 1]
            y_t = np.array([target[i]], dtype=np.float64)

            X_list.append(x_win)
            y_list.append(y_t)
            sample_meta.append({
                "date": idx[i],
                "ticker": ticker,
                "window_dates": idx[i - window + 1 : i + 1],
            })

    if not X_list:
        return np.empty((0, window, len(feat_cols))), np.empty((0, 1)), []

    X = np.stack(X_list)
    y = np.stack(y_list)
    return X, y, sample_meta


# ════════════════════════════════════════════════════════════════════════
# 4. Diagnostics helpers
# ════════════════════════════════════════════════════════════════════════
def summarize_raw_feature_stats(X_train, X_val, feat_cols):
    Ntr, W, F = X_train.shape
    Nvl, _, _ = X_val.shape

    tr_flat = X_train.reshape(-1, F)
    vl_flat = X_val.reshape(-1, F)

    rows = []
    for j, feat in enumerate(feat_cols):
        tr_col = tr_flat[:, j]
        vl_col = vl_flat[:, j]

        rows.append({
            "feature": feat,
            "train_mean": np.mean(tr_col),
            "train_std": np.std(tr_col),
            "train_min": np.min(tr_col),
            "train_p01": np.percentile(tr_col, 1),
            "train_p50": np.percentile(tr_col, 50),
            "train_p99": np.percentile(tr_col, 99),
            "train_max": np.max(tr_col),
            "val_mean": np.mean(vl_col),
            "val_std": np.std(vl_col),
            "val_min": np.min(vl_col),
            "val_p01": np.percentile(vl_col, 1),
            "val_p50": np.percentile(vl_col, 50),
            "val_p99": np.percentile(vl_col, 99),
            "val_max": np.max(vl_col),
        })

    return pd.DataFrame(rows)


def summarize_normalized_feature_stats(X_train_n, X_val_n, feat_cols):
    F = X_train_n.shape[2]
    tr_max = np.max(np.abs(X_train_n), axis=(0, 1))
    vl_max = np.max(np.abs(X_val_n), axis=(0, 1))
    tr_mean_abs = np.mean(np.abs(X_train_n), axis=(0, 1))
    vl_mean_abs = np.mean(np.abs(X_val_n), axis=(0, 1))

    rows = []
    for j in range(F):
        rows.append({
            "feature": feat_cols[j],
            "train_max_abs_z": tr_max[j],
            "val_max_abs_z": vl_max[j],
            "train_mean_abs_z": tr_mean_abs[j],
            "val_mean_abs_z": vl_mean_abs[j],
            "val_to_train_max_ratio": vl_max[j] / (tr_max[j] + 1e-12),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("val_max_abs_z", ascending=False)
    return df


def find_top_extreme_samples(X_val, X_val_n, sample_meta, feat_cols, top_k_features=10, top_m_samples=5):
    """
    For the most extreme normalized validation features, find the exact samples causing the issue.
    """
    val_max = np.max(np.abs(X_val_n), axis=(0, 1))
    feature_order = np.argsort(-val_max)[:top_k_features]

    reports = {}

    for j in feature_order:
        feat = feat_cols[j]

        abs_z = np.abs(X_val_n[:, :, j])  # (N, W)
        flat_idx = np.argpartition(abs_z.ravel(), -top_m_samples)[-top_m_samples:]
        flat_idx = flat_idx[np.argsort(-abs_z.ravel()[flat_idx])]

        rows = []
        for idx_flat in flat_idx:
            sample_idx, time_pos = np.unravel_index(idx_flat, abs_z.shape)

            meta = sample_meta[sample_idx]
            raw_val = X_val[sample_idx, time_pos, j]
            z_val = X_val_n[sample_idx, time_pos, j]
            source_date = meta["window_dates"][time_pos]

            rows.append({
                "feature": feat,
                "sample_idx": sample_idx,
                "ticker": meta["ticker"],
                "prediction_date": meta["date"],
                "window_position": time_pos,
                "source_date": source_date,
                "raw_value": raw_val,
                "z_value": z_val,
            })

        reports[feat] = pd.DataFrame(rows)

    return reports


def print_top_problem_features(raw_df, norm_df, top_k=15):
    merged = norm_df.merge(raw_df, on="feature", how="left")
    print("\nTop problematic features in validation:")
    cols = [
        "feature",
        "train_max_abs_z",
        "val_max_abs_z",
        "val_to_train_max_ratio",
        "train_std",
        "train_min",
        "train_p01",
        "train_p50",
        "train_p99",
        "train_max",
        "val_min",
        "val_p01",
        "val_p50",
        "val_p99",
        "val_max",
    ]
    print(merged[cols].head(top_k).to_string(index=False))


# ════════════════════════════════════════════════════════════════════════
# 5. Analyze one fold
# ════════════════════════════════════════════════════════════════════════
def analyze_one_fold(
    df,
    fold_idx,
    tr_dates,
    vl_dates,
    te_dates,
    window,
    out_dir,
):
    print("\n" + "=" * 90)
    print(
        f"Fold {fold_idx} | "
        f"train: {pd.Timestamp(tr_dates[0]).date()} → {pd.Timestamp(tr_dates[-1]).date()} | "
        f"val:   {pd.Timestamp(vl_dates[0]).date()} → {pd.Timestamp(vl_dates[-1]).date()} | "
        f"test:  {pd.Timestamp(te_dates[0]).date()} → {pd.Timestamp(te_dates[-1]).date()}"
    )
    print("=" * 90)

    X_tr, y_tr, meta_tr = build_samples_with_meta(df, tr_dates, ALL_FEAT_COLS, window)
    X_vl, y_vl, meta_vl = build_samples_with_meta(df, vl_dates, ALL_FEAT_COLS, window)

    print(f"Train samples: {len(X_tr):,}")
    print(f"Val samples:   {len(X_vl):,}")
    print(f"X_tr shape: {X_tr.shape}")
    print(f"X_vl shape: {X_vl.shape}")
    print(f"y_tr shape: {y_tr.shape}")
    print(f"y_vl shape: {y_vl.shape}")

    # standardize exactly like backtest
    Ntr, W, F = X_tr.shape
    Nvl, _, _ = X_vl.shape

    tr_flat = X_tr.reshape(-1, F)
    vl_flat = X_vl.reshape(-1, F)

    scaler = StandardScaler().fit(tr_flat)

    X_tr_n = scaler.transform(tr_flat).reshape(Ntr, W, F)
    X_vl_n = scaler.transform(vl_flat).reshape(Nvl, W, F)

    print(f"X_tr_n finite: {np.isfinite(X_tr_n).all()} | max abs: {np.max(np.abs(X_tr_n)):.6f}")
    print(f"X_vl_n finite: {np.isfinite(X_vl_n).all()} | max abs: {np.max(np.abs(X_vl_n)):.6f}")

    raw_df = summarize_raw_feature_stats(X_tr, X_vl, ALL_FEAT_COLS)
    norm_df = summarize_normalized_feature_stats(X_tr_n, X_vl_n, ALL_FEAT_COLS)

    print_top_problem_features(raw_df, norm_df, top_k=15)

    extreme_reports = find_top_extreme_samples(
        X_val=X_vl,
        X_val_n=X_vl_n,
        sample_meta=meta_vl,
        feat_cols=ALL_FEAT_COLS,
        top_k_features=10,
        top_m_samples=10,
    )

    # save outputs
    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(fold_dir / "raw_feature_stats.csv", index=False)
    norm_df.to_csv(fold_dir / "normalized_feature_stats.csv", index=False)

    for feat, rep_df in extreme_reports.items():
        safe_name = feat.replace("/", "_").replace("\\", "_").replace(" ", "_")
        rep_df.to_csv(fold_dir / f"extreme_samples__{safe_name}.csv", index=False)

    print(f"\nSaved diagnostic files to: {fold_dir}")


# ════════════════════════════════════════════════════════════════════════
# 6. Main
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Analyze problematic folds in backtest")
    parser.add_argument("--tickers", type=int, default=None, help="Restrict to first N tickers")
    parser.add_argument("--window", type=int, default=20, help="Rolling window length")
    parser.add_argument("--train_years", type=float, default=2.0)
    parser.add_argument("--val_months", type=int, default=6)
    parser.add_argument("--test_months", type=int, default=3)
    parser.add_argument("--step_months", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(os.getcwd()) / "result" / "fold_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_days = int(args.train_years * ANNUAL_DAYS)
    val_days = int(args.val_months * ANNUAL_DAYS / 12)
    test_days = int(args.test_months * ANNUAL_DAYS / 12)
    step_days = int(args.step_months * ANNUAL_DAYS / 12)

    df = load_features(max_tickers=args.tickers)
    all_dates = sorted(df.index.get_level_values("date").unique())

    folds = list(rolling_windows(all_dates, train_days, val_days, test_days, step_days))

    # only analyze folds 3 and 4
    target_fold_numbers = [3, 4]

    for fold_number in target_fold_numbers:
        if fold_number < 1 or fold_number > len(folds):
            print(f"Fold {fold_number} does not exist. Total folds: {len(folds)}")
            continue

        tr_dates, vl_dates, te_dates = folds[fold_number - 1]
        analyze_one_fold(
            df=df,
            fold_idx=fold_number,
            tr_dates=tr_dates,
            vl_dates=vl_dates,
            te_dates=te_dates,
            window=args.window,
            out_dir=out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()