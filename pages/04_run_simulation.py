"""
pages/04_run_simulation.py  —  Simulation

Loads pre-computed model predictions from the selected results folder and applies
the same portfolio construction logic used in portfolio_comparison_transaction_cost.py.

Important:
- No model retraining.
- Portfolio results are generated only after clicking Run Simulation.
- Short group size can be 0%.
- Long-short and long-only strategies are plotted separately.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st


ROOT = st.session_state.get("root", Path(__file__).resolve().parent.parent)
ANNUAL_DAYS = 252

st.set_page_config(page_title="Simulation", page_icon="▶️", layout="wide")
st.title("▶️ Simulation")
st.caption(
    "Applies portfolio rules to pre-trained model predictions. "
    "No retraining — results are generated only after you click Run Simulation."
)

results_dir = Path(st.session_state.get("results_dir", ROOT / "results"))


@st.cache_data(show_spinner="Loading saved predictions …")
def load_prediction_files(pred_dir_str: str) -> dict[str, pd.DataFrame]:
    pred_dir = Path(pred_dir_str)
    files = {
        "uWN": pred_dir / "predictions_uWN.csv",
        "cWN": pred_dir / "predictions_cWN.csv",
        "LSTM": pred_dir / "predictions_LSTM.csv",
    }

    pred_dfs = {}

    for name, path in files.items():
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if {"date", "ticker", "pred", "realized"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "ticker"]).sort_index()

        elif {"pred", "realized"}.issubset(df.columns) and df.shape[1] >= 4:
            first_col = df.columns[0]
            second_col = df.columns[1]
            df = df.rename(columns={first_col: "date", second_col: "ticker"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "ticker"]).sort_index()

        else:
            st.warning(
                f"Skipping {name}: `{path.name}` does not contain the required "
                "columns date, ticker, pred, realized."
            )
            continue

        df = df[["pred", "realized"]].replace([np.inf, -np.inf], np.nan).dropna()
        pred_dfs[name] = df

    return pred_dfs


preds = load_prediction_files(str(results_dir))

if not preds:
    st.error(
        f"No prediction files found in `{results_dir}`.\n\n"
        "Expected files include `predictions_uWN.csv`, `predictions_cWN.csv`, "
        "and optionally `predictions_LSTM.csv`."
    )
    st.stop()

st.caption(f"Prediction files loaded from: `{results_dir}`")


def form_portfolio(
    pred_df: pd.DataFrame,
    top_frac: float = 0.20,
    bot_frac: float = 0.20,
    cost_bps: float = 10.0,
    holding_days: int = 1,
) -> pd.DataFrame:
    """
    Portfolio construction with transaction costs for BOTH long-short and long-only.

    Long-short:
        ls_ret_gross = 0.5 * long_ret_gross + 0.5 * short_ret
        ls_ret_net   = ls_ret_gross - ls_turnover * cost_rate

    Long-only:
        long_ret_net = long_ret_gross - long_turnover * cost_rate

    Notes:
    - bot_frac can be 0 for long-only-only simulations.
    - long_ret is kept as the NET long-only return for backward compatibility.
    - ls_ret is kept as the NET long-short return for backward compatibility.
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
        k_bot = int(n * bot_frac) if bot_frac > 0 else 0
        tickers_today = group.index.get_level_values("ticker")

        if date in rebal_set or not prev_long:
            ranked = group["pred"].rank(ascending=False, method="first")

            curr_long = set(tickers_today[ranked <= k_top])

            if k_bot > 0:
                curr_short = set(tickers_today[ranked > (n - k_bot)])
            else:
                curr_short = set()

            long_turnover = len(curr_long ^ prev_long) / max(len(curr_long), 1)

            if k_bot > 0:
                short_turnover = len(curr_short ^ prev_short) / max(len(curr_short), 1)
                ls_turnover = (long_turnover + short_turnover) / 2.0
            else:
                short_turnover = 0.0
                ls_turnover = long_turnover

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
            "ls_ret": ls_ret_net,
            "ls_ret_net": ls_ret_net,
            "ls_ret_gross": ls_ret_gross,
            "short_ret": short_ret,

            # Long-only returns
            "long_ret": long_ret_net,
            "long_ret_net": long_ret_net,
            "long_ret_gross": long_ret_gross,

            # Transaction costs and turnover
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

def compute_metrics(daily_returns: pd.Series, name: str = "strategy") -> dict:
    r = daily_returns.dropna()

    if len(r) == 0:
        return {
            "Strategy": name,
            "Cum Return %": np.nan,
            "Ann Ret %": np.nan,
            "Ann Vol %": np.nan,
            "Sharpe": np.nan,
            "Max DD %": np.nan,
            "Hit Rate %": np.nan,
            "N Days": 0,
        }

    cum_ret = (1 + r).prod() - 1
    ann_ret = (1 + r).prod() ** (ANNUAL_DAYS / len(r)) - 1
    ann_vol = r.std() * np.sqrt(ANNUAL_DAYS)
    sharpe = ann_ret / (ann_vol + 1e-9)

    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    hit_rate = (r > 0).mean()

    return {
        "Strategy": name,
        "Cum Return %": round(cum_ret * 100, 2),
        "Ann Ret %": round(ann_ret * 100, 2),
        "Ann Vol %": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Max DD %": round(max_dd * 100, 2),
        "Hit Rate %": round(hit_rate * 100, 2),
        "N Days": len(r),
    }


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


def plot_four_panel(portfolios: dict[str, pd.Series], title: str):
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
    return fig


st.subheader("Portfolio Parameters")

with st.form("simulation_form"):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        strategy_type = st.radio(
            "Strategy type",
            ["Long-short", "Long-only", "Both"],
            index=2,
            help="Choose which portfolio strategy to simulate.",
        )

    with c2:
        top_pct = st.select_slider(
            "Long group size",
            options=[5, 10, 15, 20, 25, 30],
            value=20,
            format_func=lambda x: f"Top {x}%",
            help="Top X% of predicted returns forms the long leg.",
        )

    with c3:
        bot_pct = st.select_slider(
            "Short group size",
            options=[0, 5, 10, 15, 20, 25, 30],
            value=20,
            format_func=lambda x: "No short leg" if x == 0 else f"Bottom {x}%",
            help="Set to 0% to disable shorts.",
        )

    with c4:
        holding_days = st.select_slider(
            "Holding period",
            options=[1, 5, 10, 21],
            value=1,
            format_func=lambda x: {
                1: "1 day",
                5: "5 days",
                10: "10 days",
                21: "21 days",
            }[x],
            help="Rebalance every N trading days.",
        )

    c5, c6 = st.columns([1, 3])

    with c5:
        cost_bps = st.select_slider(
            "Transaction cost (bps per side)",
            options=[0, 5, 10, 15, 20, 30],
            value=10,
            help="One-way cost in basis points.",
        )

    with c6:
        st.write("")
        st.write("")
        run_clicked = st.form_submit_button("▶️ Run Simulation", use_container_width=True)


if not run_clicked:
    st.info("Choose portfolio parameters, then click **Run Simulation** to generate results.")
    st.stop()


with st.spinner("Applying portfolio rules …"):
    ports = {
        name: form_portfolio(
            df,
            top_frac=top_pct / 100,
            bot_frac=bot_pct / 100,
            cost_bps=cost_bps,
            holding_days=holding_days,
        )
        for name, df in preds.items()
    }

common = None
for port in ports.values():
    common = port.index if common is None else common.intersection(port.index)

st.success(
    f"Simulation complete: {strategy_type}, top {top_pct}%, bottom {bot_pct}%, "
    f"rebalance every {holding_days} trading day(s), cost {cost_bps} bps/side."
)

show_long_short = strategy_type in ["Long-short", "Both"] and bot_pct > 0
show_long_only = strategy_type in ["Long-only", "Both"]

if strategy_type == "Long-short" and bot_pct == 0:
    st.warning(
        "Short group size is 0%, so a long-short strategy cannot be formed. "
        "Increase the short group size or choose Long-only."
    )

if show_long_short:
    st.subheader("Long-Short Strategy — Portfolio Performance")
    fig_ls = plot_four_panel(
        portfolios={
            f"{name} L/S net": port.loc[common, "ls_ret"]
            for name, port in ports.items()
        },
        title=f"Long-Short Portfolio Comparison, {cost_bps} bps Cost",
    )
    st.pyplot(fig_ls)
    plt.close(fig_ls)

if show_long_only:
    st.subheader("Long-Only Strategy — Portfolio Performance")
    fig_long = plot_four_panel(
        portfolios={
            f"{name} Long net": port.loc[common, "long_ret_net"]
            for name, port in ports.items()
        },
        title=f"Long-Only Net Portfolio Comparison, {cost_bps} bps Cost",
    )
    st.pyplot(fig_long)
    plt.close(fig_long)

st.subheader("Performance Metrics")

rows = []

for name, port in ports.items():
    if show_long_short:
        rows.append(compute_metrics(port.loc[common, "ls_ret"], f"{name} L/S net"))
        rows.append(compute_metrics(port.loc[common, "ls_ret_gross"], f"{name} L/S gross"))

    if show_long_only:
        rows.append(compute_metrics(port.loc[common, "long_ret_net"], f"{name} Long net"))
        rows.append(compute_metrics(port.loc[common, "long_ret_gross"], f"{name} Long gross"))

metrics_df = pd.DataFrame(rows)

st.dataframe(
    metrics_df.set_index("Strategy").style.format({
        "Cum Return %": "{:.2f}",
        "Ann Ret %": "{:.2f}",
        "Ann Vol %": "{:.2f}",
        "Sharpe": "{:.3f}",
        "Max DD %": "{:.2f}",
        "Hit Rate %": "{:.2f}",
        "N Days": "{:.0f}",
    }),
    use_container_width=True,
)

if show_long_short or show_long_only:
    st.subheader("Transaction Cost Sensitivity — Sharpe Ratio")
    st.caption("Holding period and group sizes are fixed; only transaction cost changes.")

    cost_rows = []

    for c in [0, 5, 10, 15, 20, 30]:
        row = {"Cost (bps/side)": c}

        for name, df in preds.items():
            p = form_portfolio(
                df,
                top_frac=top_pct / 100,
                bot_frac=bot_pct / 100,
                cost_bps=c,
                holding_days=holding_days,
            )

            if show_long_short:
                r_ls = p["ls_ret_net"].reindex(common).dropna()
                sh_ls = r_ls.mean() / (r_ls.std() + 1e-9) * np.sqrt(ANNUAL_DAYS)
                row[f"{name} L/S net"] = round(sh_ls, 3)

            if show_long_only:
                r_long = p["long_ret_net"].reindex(common).dropna()
                sh_long = r_long.mean() / (r_long.std() + 1e-9) * np.sqrt(ANNUAL_DAYS)
                row[f"{name} Long net"] = round(sh_long, 3)

        cost_rows.append(row)

    cost_df = pd.DataFrame(cost_rows).set_index("Cost (bps/side)")

    def color_sharpe(val):
        try:
            v = float(val)
            if v > 0:
                return "color: #16A34A; font-weight: bold"
            if v < -0.1:
                return "color: #DC2626"
            return ""
        except Exception:
            return ""

    st.dataframe(
        cost_df.style.applymap(color_sharpe),
        use_container_width=True,
    )

if show_long_short:
    st.subheader("Average Turnover on Rebalance Days — Long-Short")
    cols = st.columns(len(ports))

    for col, (name, port) in zip(cols, ports.items()):
        rebal_turn = port[port["ls_turnover"] > 0]["ls_turnover"]
        avg_t = rebal_turn.mean() * 100 if len(rebal_turn) > 0 else 0
        n_rebal = len(rebal_turn)

        col.metric(
            f"{name} L/S",
            f"{avg_t:.1f}%",
            delta=f"{n_rebal} rebalances",
            help="Average turnover used to deduct transaction cost in the long-short strategy.",
        )

if show_long_only:
    st.subheader("Average Turnover on Rebalance Days — Long-Only")
    cols = st.columns(len(ports))

    for col, (name, port) in zip(cols, ports.items()):
        rebal_turn = port[port["long_turnover"] > 0]["long_turnover"]
        avg_t = rebal_turn.mean() * 100 if len(rebal_turn) > 0 else 0
        n_rebal = len(rebal_turn)

        col.metric(
            f"{name} Long",
            f"{avg_t:.1f}%",
            delta=f"{n_rebal} rebalances",
            help="Average turnover used to deduct transaction cost in the long-only strategy.",
        )
