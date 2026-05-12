"""
robustness/regime_analysis.py
==============================
Phase 5 — Regime Testing
Loads daily returns from results/ and reports per-strategy performance
broken out by three regimes:
  • Bull market  : 2010-01-01 → 2019-12-31
  • COVID crash  : 2020-02-01 → 2020-04-30
  • 2022 bear    : 2022-01-01 → 2022-12-31

Usage
-----
    cd <project_root>
    python robustness/regime_analysis.py
    python robustness/regime_analysis.py --results_dir results/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ANNUAL_DAYS = 252

# ── Regime definitions ────────────────────────────────────────────────────────
REGIMES = {
    "Bull Market (2010–2019)": ("2010-01-01", "2019-12-31"),
    "COVID Crash (Feb–Apr 2020)": ("2020-02-01", "2020-04-30"),
    "Bear Market 2022": ("2022-01-01", "2022-12-31"),
}

STRATEGY_COLS = {
    "uWN L/S":   "uWN_ls",
    "cWN L/S":   "cWN_ls",
    "LSTM L/S":  "LSTM_ls",
    "uWN Long":  "uWN_long",
    "cWN Long":  "cWN_long",
    "LSTM Long": "LSTM_long",
    "SPY B&H":   "spy",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(rets: pd.Series, name: str) -> dict:
    """Standard performance metrics for a daily-return series."""
    rets = rets.dropna()
    if len(rets) == 0:
        return dict(name=name, n_days=0, ann_return=np.nan, ann_vol=np.nan,
                    sharpe=np.nan, max_dd=np.nan, hit_rate=np.nan, calmar=np.nan)

    ann_ret  = rets.mean() * ANNUAL_DAYS
    ann_vol  = rets.std()  * np.sqrt(ANNUAL_DAYS)
    sharpe   = ann_ret / (ann_vol + 1e-9)
    cum      = (1 + rets).cumprod()
    max_dd   = ((cum / cum.cummax()) - 1).min() * 100
    calmar   = ann_ret / (abs(max_dd / 100) + 1e-9)
    hit_rate = (rets > 0).mean() * 100

    return dict(
        name     = name,
        n_days   = len(rets),
        ann_return = ann_ret * 100,
        ann_vol    = ann_vol  * 100,
        sharpe     = sharpe,
        max_dd     = max_dd,
        calmar     = calmar,
        hit_rate   = hit_rate,
    )


# ── Core analysis ─────────────────────────────────────────────────────────────

def load_returns(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "daily_returns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run backtest.py first to generate results."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def analyse_regime(returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return a metrics DataFrame for all strategies within a date range."""
    mask = (returns.index >= start) & (returns.index <= end)
    slice_ = returns.loc[mask]
    rows = []
    for label, col in STRATEGY_COLS.items():
        if col in slice_.columns:
            rows.append(compute_metrics(slice_[col], label))
    return pd.DataFrame(rows).set_index("name")


def run_regime_analysis(results_dir: Path) -> dict[str, pd.DataFrame]:
    returns = load_returns(results_dir)
    all_regime_tables = {}

    print("\n" + "=" * 70)
    print("  REGIME PERFORMANCE ANALYSIS")
    print("=" * 70)

    for regime_name, (start, end) in REGIMES.items():
        tbl = analyse_regime(returns, start, end)
        all_regime_tables[regime_name] = tbl

        actual_start = returns.loc[start:end].index.min()
        actual_end   = returns.loc[start:end].index.max()
        n_days       = len(returns.loc[start:end].dropna(how="all"))

        print(f"\n{'─'*70}")
        print(f"  {regime_name}  [{actual_start.date()} → {actual_end.date()}]  ({n_days} trading days)")
        print(f"{'─'*70}")
        header = f"  {'Strategy':<20} {'AnnRet%':>8} {'AnnVol%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Calmar':>8} {'Hit%':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, row in tbl.iterrows():
            print(
                f"  {name:<20} {row['ann_return']:>8.1f} {row['ann_vol']:>8.1f} "
                f"{row['sharpe']:>8.3f} {row['max_dd']:>8.1f} {row['calmar']:>8.2f} "
                f"{row['hit_rate']:>7.1f}"
            )

    return all_regime_tables


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_regime_cumulative(returns: pd.DataFrame, save_path: Path | None = None):
    """Plot cumulative returns for each regime side by side."""
    n_regimes = len(REGIMES)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6 * n_regimes, 5), sharey=False)
    if n_regimes == 1:
        axes = [axes]

    colors = {
        "uWN_ls":   "steelblue",
        "cWN_ls":   "darkorange",
        "LSTM_ls":  "green",
        "uWN_long": "cornflowerblue",
        "cWN_long": "sandybrown",
        "LSTM_long":"mediumseagreen",
        "spy":      "grey",
    }
    labels = {v: k for k, v in STRATEGY_COLS.items()}

    for ax, (regime_name, (start, end)) in zip(axes, REGIMES.items()):
        slice_ = returns.loc[start:end].dropna(how="all")
        if slice_.empty:
            ax.set_title(f"{regime_name}\n(no data)")
            continue

        for col in ["uWN_ls", "cWN_ls", "LSTM_ls", "spy"]:
            if col not in slice_.columns:
                continue
            cum = (1 + slice_[col].fillna(0)).cumprod()
            ax.plot(cum.index, cum.values, color=colors[col],
                    lw=2, label=labels[col])

        ax.axhline(1, color="black", lw=0.5, ls="--")
        ax.set_title(regime_name, fontsize=10)
        ax.set_ylabel("Growth of $1")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.suptitle("Cumulative Returns by Market Regime", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Regime plot saved → {save_path}")
    else:
        plt.show()


def plot_regime_sharpe_bar(regime_tables: dict[str, pd.DataFrame],
                           save_path: Path | None = None):
    """Grouped bar chart of Sharpe ratios across regimes."""
    strategies = list(STRATEGY_COLS.keys())
    regime_names = list(regime_tables.keys())
    n_reg = len(regime_names)
    n_str = len(strategies)

    sharpe_matrix = np.zeros((n_str, n_reg))
    for j, regime_name in enumerate(regime_names):
        tbl = regime_tables[regime_name]
        for i, strat in enumerate(strategies):
            if strat in tbl.index:
                sharpe_matrix[i, j] = tbl.loc[strat, "sharpe"]

    x    = np.arange(n_reg)
    w    = 0.15
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["steelblue", "darkorange", "green", "cornflowerblue", "sandybrown", "mediumseagreen", "grey"]
    for i, strat in enumerate(strategies):
        offset = (i - n_str / 2 + 0.5) * w
        ax.bar(x + offset, sharpe_matrix[i], width=w,
               color=palette[i], alpha=0.85, label=strat)

    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(regime_names, fontsize=9)
    ax.set_ylabel("Annualised Sharpe Ratio")
    ax.set_title("Sharpe Ratio by Market Regime")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sharpe bar plot saved → {save_path}")
    else:
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Regime performance analysis")
    p.add_argument("--results_dir", type=str, default="results/uWNcWN_LSTM2",
                   help="Directory containing daily_returns.csv (default: results/uWNcWN_LSTM2)")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    regime_tables = run_regime_analysis(results_dir)

    # Save per-regime CSVs
    for regime_name, tbl in regime_tables.items():
        safe_name = regime_name.replace(" ", "_").replace("(", "").replace(")", "").replace("–", "-")
        out_path  = results_dir / f"regime_{safe_name}.csv"
        tbl.to_csv(out_path)
        print(f"Saved → {out_path}")

    if not args.no_plot:
        returns = load_returns(results_dir)
        plot_regime_cumulative(returns, save_path=results_dir / "regime_cumulative.png")
        plot_regime_sharpe_bar(regime_tables,  save_path=results_dir / "regime_sharpe.png")


if __name__ == "__main__":
    main()