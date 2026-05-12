"""
Portfolio comparison with transaction cost modeling
==================================================

This script assumes model prediction files have already been generated and saved.
It does NOT train any model.

Expected input files in --pred_dir:
    predictions_uWN.csv
    predictions_cWN.csv
    predictions_LSTM.csv

Each prediction file should contain columns:
    date, ticker, pred, realized

Outputs in --results_dir:
    metrics.csv
    daily_returns.csv
    cumulative_returns.csv
    long_short_comparison.png
    long_only_comparison.png

Example:
    python portfolio_comparison_transaction_cost.py \
        --pred_dir results2 \
        --results_dir results_portfolio \
        --cost_bps 10 \
        --holding 1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


ANNUAL_DAYS = 252


# =============================================================================
# 1. LOAD PREDICTION FILES
# =============================================================================

def load_prediction_files(pred_dir: str | Path = "results2") -> dict[str, pd.DataFrame]:
    """
    Load saved prediction files.

    Expected files:
        predictions_uWN.csv
        predictions_cWN.csv
        predictions_LSTM.csv

    Each file should contain:
        date, ticker, pred, realized

    Returns
    -------
    pred_dfs : dict[str, pd.DataFrame]
        Dictionary mapping model name to prediction DataFrame.
        Each DataFrame has MultiIndex [date, ticker] and columns [pred, realized].
    """
    pred_dir = Path(pred_dir)

    files = {
        "uWN": pred_dir / "predictions_uWN.csv",
        "cWN": pred_dir / "predictions_cWN.csv",
        "LSTM": pred_dir / "predictions_LSTM.csv",
    }

    pred_dfs = {}

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {name}: {path}")

        df = pd.read_csv(path)

        required_cols = {"date", "ticker", "pred", "realized"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "ticker"]).sort_index()
        df = df[["pred", "realized"]].replace([np.inf, -np.inf], np.nan).dropna()

        pred_dfs[name] = df
        print(f"Loaded {name}: {len(df):,} rows from {path}")

    return pred_dfs


# =============================================================================
# 2. PORTFOLIO CONSTRUCTION WITH TRANSACTION COSTS
# =============================================================================

def form_portfolio(
    pred_df: pd.DataFrame,
    top_frac: float = 0.20,
    bot_frac: float = 0.20,
    cost_bps: float = 10.0,
    holding_days: int = 1,
) -> pd.DataFrame:
    """
    Build portfolio returns with transaction cost and holding period.

    This revised version deducts transaction costs from BOTH:
        1. the long-short net portfolio
        2. the long-only net portfolio

    Long-short:
        ls_ret_gross = 0.5 * long_ret_gross + 0.5 * short_ret
        ls_ret_net   = ls_ret_gross - ls_turnover * cost_rate

    Long-only:
        long_ret_net = long_ret_gross - long_turnover * cost_rate
    """
    cost_rate = cost_bps / 10_000
    all_dates = sorted(pred_df.index.get_level_values("date").unique())
    rebal_set = set(all_dates[::holding_days])

    prev_long, prev_short = set(), set()
    rows = []

    for date, group in pred_df.groupby(level="date"):
        n = len(group)
        if n < 10:
            continue

        k_top = max(1, int(n * top_frac))
        k_bot = max(1, int(n * bot_frac))
        tickers_today = group.index.get_level_values("ticker")

        if date in rebal_set or not prev_long:
            ranked = group["pred"].rank(ascending=False, method="first")

            curr_long = set(tickers_today[ranked <= k_top])
            curr_short = set(tickers_today[ranked > (n - k_bot)])

            long_turnover = len(curr_long ^ prev_long) / max(len(curr_long), 1)
            short_turnover = len(curr_short ^ prev_short) / max(len(curr_short), 1)

            # Same convention as the original long-short strategy:
            # average long-leg and short-leg turnover.
            ls_turnover = (long_turnover + short_turnover) / 2.0

            prev_long, prev_short = curr_long, curr_short
        else:
            long_turnover = 0.0
            short_turnover = 0.0
            ls_turnover = 0.0

        long_mask = tickers_today.isin(prev_long)
        short_mask = tickers_today.isin(prev_short)

        long_ret_gross = group.loc[long_mask, "realized"].mean() if long_mask.any() else 0.0
        short_ret = (-group.loc[short_mask, "realized"]).mean() if short_mask.any() else 0.0

        ls_ret_gross = 0.5 * long_ret_gross + 0.5 * short_ret

        ls_transaction_cost = ls_turnover * cost_rate
        long_transaction_cost = long_turnover * cost_rate

        ls_ret_net = ls_ret_gross - ls_transaction_cost
        long_ret_net = long_ret_gross - long_transaction_cost

        rows.append({
            "date": date,

            # Long-short returns
            "ls_ret": ls_ret_net,          # backward compatible: net L/S return
            "ls_ret_net": ls_ret_net,
            "ls_ret_gross": ls_ret_gross,
            "short_ret": short_ret,

            # Long-only returns
            "long_ret": long_ret_net,      # backward compatible: now net long-only return
            "long_ret_net": long_ret_net,
            "long_ret_gross": long_ret_gross,

            # Costs and turnover
            "transaction_cost": ls_transaction_cost,
            "ls_transaction_cost": ls_transaction_cost,
            "long_transaction_cost": long_transaction_cost,
            "turnover": ls_turnover,
            "ls_turnover": ls_turnover,
            "long_turnover": long_turnover,
            "short_turnover": short_turnover,

            # Holdings
            "n_long": int(long_mask.sum()),
            "n_short": int(short_mask.sum()),
        })

    return pd.DataFrame(rows).set_index("date").sort_index()

# =============================================================================
# 3. PERFORMANCE METRICS
# =============================================================================

def compute_metrics(daily_returns: pd.Series, name: str = "strategy") -> dict:
    """
    Compute standard portfolio performance metrics.
    """
    r = daily_returns.dropna()

    if len(r) == 0:
        raise ValueError(f"No valid returns for {name}")

    cum_ret = (1 + r).prod() - 1
    ann_ret = (1 + r).prod() ** (ANNUAL_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ANNUAL_DAYS)
    sharpe = ann_ret / (ann_vol + 1e-9)

    equity = (1 + r).cumprod()
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_dd = drawdown.min()
    hit_rate = (r > 0).mean()

    return {
        "name": name,
        "cum_return_pct": round(cum_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "hit_rate_pct": round(hit_rate * 100, 2),
        "n_days": len(r),
    }


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """
    Print a readable metrics table.
    """
    display_cols = [
        "name",
        "cum_return_pct",
        "ann_return_pct",
        "ann_vol_pct",
        "sharpe",
        "max_drawdown_pct",
        "hit_rate_pct",
        "n_days",
    ]

    print("\n" + "=" * 100)
    print("PORTFOLIO PERFORMANCE COMPARISON")
    print("=" * 100)
    print(metrics_df[display_cols].to_string(index=False))
    print("=" * 100)


# =============================================================================
# 4. PLOTTING HELPERS
# =============================================================================

def _annual_returns(series: pd.Series) -> pd.Series:
    return series.resample("YE").apply(lambda r: (1 + r).prod() - 1) * 100


def _rolling_sharpe(series: pd.Series, window: int = 63) -> pd.Series:
    return series.rolling(window).apply(
        lambda r: r.mean() / (r.std() + 1e-9) * np.sqrt(ANNUAL_DAYS),
        raw=True,
    )


def _drawdown(series: pd.Series) -> pd.Series:
    equity = (1 + series).cumprod()
    return equity / equity.cummax() - 1


def plot_four_panel(
    portfolios: dict[str, pd.Series],
    title: str,
    save_path: str | Path | None = None,
):
    """
    Four-panel performance plot:
        1. Cumulative return
        2. Rolling Sharpe
        3. Drawdown
        4. Annual returns
    """
    common = None
    for s in portfolios.values():
        common = s.index if common is None else common.intersection(s.index)

    aligned = {k: v.reindex(common).dropna() for k, v in portfolios.items()}

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])
    for name, series in aligned.items():
        cum = (1 + series).cumprod()
        ax1.plot(cum.index, cum.values, lw=2, label=name)
    ax1.axhline(1, color="black", lw=0.5, ls="--")
    ax1.set_title("Cumulative Return")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(fontsize=10)

    ax2 = fig.add_subplot(gs[1, :])
    for name, series in aligned.items():
        rsh = _rolling_sharpe(series, window=63)
        ax2.plot(rsh.index, rsh.values, lw=1.5, label=f"{name} 63d Sharpe")
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    ax2.set_title("Rolling 63-Day Sharpe Ratio")
    ax2.set_ylabel("Annualized Sharpe")
    ax2.legend(fontsize=10)

    ax3 = fig.add_subplot(gs[2, 0])
    for name, series in aligned.items():
        dd = _drawdown(series)
        ax3.fill_between(dd.index, dd.values, 0, alpha=0.28, label=name)
    ax3.set_title("Drawdown")
    ax3.set_ylabel("Drawdown")
    ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[2, 1])
    annual_dict = {k: _annual_returns(v) for k, v in aligned.items()}
    years = sorted(set().union(*[s.index.year.tolist() for s in annual_dict.values()]))

    x = np.arange(len(years))
    n_series = len(annual_dict)
    width = 0.8 / max(n_series, 1)

    for i, (name, annual_s) in enumerate(annual_dict.items()):
        vals = [
            annual_s[annual_s.index.year == y].iloc[0]
            if y in annual_s.index.year
            else 0.0
            for y in years
        ]
        offset = (i - (n_series - 1) / 2) * width
        ax4.bar(x + offset, vals, width=width, alpha=0.85, label=name)

    ax4.axhline(0, color="black", lw=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(years, rotation=45, fontsize=8)
    ax4.set_title("Annual Returns (%)")
    ax4.legend(fontsize=9)

    plt.suptitle(title, fontsize=13)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()


# =============================================================================
# 5. MAIN COMPARISON FUNCTION
# =============================================================================

def run_portfolio_comparison(
    pred_dfs: dict[str, pd.DataFrame],
    cost_bps: float = 10.0,
    holding_days: int = 1,
    top_frac: float = 0.20,
    bot_frac: float = 0.20,
    results_dir: str | Path = "transaction_cost",
    no_plot: bool = False,
):
    """
    Compare portfolio performance across already-loaded prediction results.

    Revised version:
        - Reports L/S net and gross.
        - Reports Long-only net and gross.
        - Saves long-only transaction costs and turnover.
        - Plots long-only NET performance.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    portfolios = {}

    for name, pred_df in pred_dfs.items():
        portfolios[name] = form_portfolio(
            pred_df=pred_df,
            top_frac=top_frac,
            bot_frac=bot_frac,
            cost_bps=cost_bps,
            holding_days=holding_days,
        )
        print(f"Formed portfolio for {name}: {len(portfolios[name]):,} days")

    common = None
    for port in portfolios.values():
        common = port.index if common is None else common.intersection(port.index)

    metrics = []

    for name, port in portfolios.items():
        metrics.append(compute_metrics(port.loc[common, "ls_ret_net"], f"{name} L/S net"))
        metrics.append(compute_metrics(port.loc[common, "ls_ret_gross"], f"{name} L/S gross"))
        metrics.append(compute_metrics(port.loc[common, "long_ret_net"], f"{name} Long net"))
        metrics.append(compute_metrics(port.loc[common, "long_ret_gross"], f"{name} Long gross"))

    metrics_df = pd.DataFrame(metrics)
    print_metrics_table(metrics_df)
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)

    returns_df = pd.DataFrame(index=common)

    for name, port in portfolios.items():
        returns_df[f"{name}_ls_net"] = port.loc[common, "ls_ret_net"]
        returns_df[f"{name}_ls_gross"] = port.loc[common, "ls_ret_gross"]
        returns_df[f"{name}_long_net"] = port.loc[common, "long_ret_net"]
        returns_df[f"{name}_long_gross"] = port.loc[common, "long_ret_gross"]
        returns_df[f"{name}_short"] = port.loc[common, "short_ret"]

        returns_df[f"{name}_ls_turnover"] = port.loc[common, "ls_turnover"]
        returns_df[f"{name}_long_turnover"] = port.loc[common, "long_turnover"]
        returns_df[f"{name}_short_turnover"] = port.loc[common, "short_turnover"]

        returns_df[f"{name}_ls_transaction_cost"] = port.loc[common, "ls_transaction_cost"]
        returns_df[f"{name}_long_transaction_cost"] = port.loc[common, "long_transaction_cost"]

    returns_df.to_csv(results_dir / "daily_returns.csv")

    cumulative_cols = [
        c for c in returns_df.columns
        if "turnover" not in c and "transaction_cost" not in c
    ]
    cumulative_df = (1 + returns_df[cumulative_cols].fillna(0)).cumprod()
    cumulative_df.to_csv(results_dir / "cumulative_returns.csv")

    if not no_plot:
        plot_four_panel(
            portfolios={
                f"{name} L/S net": port.loc[common, "ls_ret_net"]
                for name, port in portfolios.items()
            },
            title=f"Long-Short Portfolio Comparison, {cost_bps} bps Cost",
            save_path=results_dir / "long_short_comparison.png",
        )

        plot_four_panel(
            portfolios={
                f"{name} Long net": port.loc[common, "long_ret_net"]
                for name, port in portfolios.items()
            },
            title=f"Long-Only Portfolio Comparison, {cost_bps} bps Cost",
            save_path=results_dir / "long_only_comparison.png",
        )

    print(f"\nSaved outputs to: {results_dir}")

    return portfolios, metrics_df, returns_df

# =============================================================================
# 6. ENTRY POINT
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Transaction-cost portfolio comparison using saved prediction files."
    )

    parser.add_argument(
        "--pred_dir",
        type=str,
        default="results/uWNcWN_LSTM2",
        help="Directory containing prediction CSV files.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="transaction_cost",
        help="Directory to save portfolio comparison outputs.",
    )
    parser.add_argument(
        "--cost_bps",
        type=float,
        default=10.0,
        help="One-way transaction cost in basis points.",
    )
    parser.add_argument(
        "--holding",
        type=int,
        default=1,
        help="Rebalance every N trading days. 1=daily, 5=weekly.",
    )
    parser.add_argument(
        "--top_frac",
        type=float,
        default=0.20,
        help="Fraction of stocks to long.",
    )
    parser.add_argument(
        "--bot_frac",
        type=float,
        default=0.20,
        help="Fraction of stocks to short.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plot generation.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    pred_dfs = load_prediction_files(pred_dir=args.pred_dir)

    run_portfolio_comparison(
        pred_dfs=pred_dfs,
        cost_bps=args.cost_bps,
        holding_days=args.holding,
        top_frac=args.top_frac,
        bot_frac=args.bot_frac,
        results_dir=args.results_dir,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
