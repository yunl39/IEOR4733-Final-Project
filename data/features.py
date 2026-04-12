"""
features.py — Extended Feature Library
========================================
Organized into 6 groups:

  Group A  │ Returns & Momentum         (9 features)
  Group B  │ Trend / Moving Averages    (8 features)
  Group C  │ Volatility                 (5 features)
  Group D  │ Oscillators                (7 features)
  Group E  │ Volume & Liquidity         (6 features)
  Group F  │ Price Structure            (6 features)
  ─────────┼────────────────────────────────────────
  Group G  │ Market-Wide Signals        (8 features)
  ─────────┴────────────────────────────────────────
             TOTAL                      49 features

All computations are look-back only — no future data leaks in.
Normalization (StandardScaler) is applied later in the pipeline,
per rolling window, using train-set statistics only.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


# ══════════════════════════════════════════════════════
#  GROUP A — RETURNS & MOMENTUM
# ══════════════════════════════════════════════════════

def add_returns_momentum(df: pd.DataFrame, close: pd.Series) -> None:
    """
    9 features:
      ret_1d, ret_2d, ret_3d, ret_5d, ret_10d, ret_20d
      mom_60d, mom_126d, mom_252d
    """
    log_close = np.log(close)

    df["ret_1d"]  = log_close.diff(1)
    df["ret_2d"]  = log_close.diff(2)
    df["ret_3d"]  = log_close.diff(3)
    df["ret_5d"]  = log_close.diff(5)
    df["ret_10d"] = log_close.diff(10)
    df["ret_20d"] = log_close.diff(20)

    # Skip-1 momentum: log(P_{t-1} / P_{t-1-N})
    log_close_lag1 = log_close.shift(1)
    df["mom_60d"]  = log_close_lag1.diff(59)
    df["mom_126d"] = log_close_lag1.diff(125)   # ~6-month
    df["mom_252d"] = log_close_lag1.diff(251)   # ~12-month


# ══════════════════════════════════════════════════════
#  GROUP B — TREND / MOVING AVERAGES
# ══════════════════════════════════════════════════════

def add_trend(df: pd.DataFrame, close: pd.Series) -> None:
    """
    8 features:
      price_to_sma20, price_to_sma50, price_to_sma200
      sma20_to_sma50  (golden/death cross signal)
      sma50_to_sma200
      ema12_to_ema26  (fast/slow EMA spread, unnormalized MACD signal)
      price_to_ema20
      slope_20d       (linear regression slope of close over 20d, normalized)
    """
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    ema20  = _ema(close, 20)

    eps = 1e-9
    df["price_to_sma20"]  = close / (sma20  + eps) - 1
    df["price_to_sma50"]  = close / (sma50  + eps) - 1
    df["price_to_sma200"] = close / (sma200 + eps) - 1
    df["sma20_to_sma50"]  = sma20  / (sma50  + eps) - 1
    df["sma50_to_sma200"] = sma50  / (sma200 + eps) - 1
    df["ema12_to_ema26"]  = ema12  / (ema26  + eps) - 1
    df["price_to_ema20"]  = close  / (ema20  + eps) - 1

    # Rolling 20-day linear regression slope (normalized by close)
    def linreg_slope(s: pd.Series) -> float:
        y = s.values
        x = np.arange(len(y))
        if np.any(np.isnan(y)):
            return np.nan
        slope = np.polyfit(x, y, 1)[0]
        return slope / (y[-1] + eps)

    df["slope_20d"] = close.rolling(20).apply(linreg_slope, raw=False)


# ══════════════════════════════════════════════════════
#  GROUP C — VOLATILITY
# ══════════════════════════════════════════════════════

def add_volatility(df: pd.DataFrame, close: pd.Series,
                   high: pd.Series, low: pd.Series) -> None:
    """
    5 features:
      realized_vol_5d, realized_vol_20d, realized_vol_60d
        — annualized rolling std of log-returns
      vol_regime      — ratio of 5d vol to 20d vol (>1 = rising vol)
      atr_pct         — ATR(14) normalized by close
    """
    log_ret = np.log(close / close.shift(1))
    ann     = np.sqrt(252)

    rv5  = log_ret.rolling(5).std()  * ann
    rv20 = log_ret.rolling(20).std() * ann
    rv60 = log_ret.rolling(60).std() * ann

    df["realized_vol_5d"]  = rv5
    df["realized_vol_20d"] = rv20
    df["realized_vol_60d"] = rv60
    df["vol_regime"]       = rv5 / (rv20 + 1e-9)

    atr = _atr(high, low, close, 14)
    df["atr_pct"] = atr / (close + 1e-9)


# ══════════════════════════════════════════════════════
#  GROUP D — OSCILLATORS
# ══════════════════════════════════════════════════════

def add_oscillators(df: pd.DataFrame, close: pd.Series,
                    high: pd.Series, low: pd.Series) -> None:
    """
    7 features:
      rsi_14
      macd_line, macd_signal, macd_hist
      bb_pct          — where price sits within Bollinger Bands (0=lower, 1=upper)
      bb_width        — (upper-lower) / sma20, measures band squeeze
      stoch_k         — fast stochastic %K (14-day)
    """
    # RSI
    df["rsi_14"] = _rsi(close, 14)

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line   = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    df["macd_line"]   = macd_line   / (close + 1e-9)   # normalize by price
    df["macd_signal"] = macd_signal / (close + 1e-9)
    df["macd_hist"]   = (macd_line - macd_signal) / (close + 1e-9)

    # Bollinger Bands (20-day, 2σ)
    sma20   = close.rolling(20).mean()
    std20   = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    band_width = bb_upper - bb_lower
    df["bb_pct"]   = (close - bb_lower) / (band_width + 1e-9)
    df["bb_width"] = band_width / (sma20 + 1e-9)

    # Stochastic %K
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = (close - low14) / (high14 - low14 + 1e-9)


# ══════════════════════════════════════════════════════
#  GROUP E — VOLUME & LIQUIDITY
# ══════════════════════════════════════════════════════

def add_volume_liquidity(df: pd.DataFrame, close: pd.Series,
                         volume: pd.Series, ret: pd.Series) -> None:
    """
    6 features:
      vol_norm        — volume / 20d avg volume
      vol_momentum    — 5d avg volume / 20d avg volume
      obv_norm        — OBV momentum: (OBV - OBV_20d_ago) / OBV_std_20d
      vpt             — Volume Price Trend (cumulative, normalized)
      amihud          — Amihud illiquidity: |ret| / (volume × close)
      hl_spread       — (High - Low) / Close  — intraday spread proxy
    """
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5  = volume.rolling(5).mean()

    df["vol_norm"]     = volume / (vol_ma20 + 1e-9)
    df["vol_momentum"] = vol_ma5 / (vol_ma20 + 1e-9)

    obv     = _obv(close, volume)
    obv_std = obv.rolling(20).std()
    df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv_std + 1e-9)

    vpt = (ret * volume).cumsum()
    vpt_std = vpt.rolling(20).std()
    df["vpt"] = (vpt - vpt.rolling(20).mean()) / (vpt_std + 1e-9)

    dollar_vol = volume * close
    df["amihud"] = ret.abs() / (dollar_vol + 1e-9) * 1e6   # scale up small numbers

    # hl_spread requires High/Low — passed in via close proxy if not available
    # actual H/L supplied in pipeline
    df["hl_spread"] = np.nan   # filled in add_price_structure when H/L available


# ══════════════════════════════════════════════════════
#  GROUP F — PRICE STRUCTURE
# ══════════════════════════════════════════════════════

def add_price_structure(df: pd.DataFrame, open_: pd.Series, close: pd.Series,
                        high: pd.Series, low: pd.Series) -> None:
    """
    6 features:
      gap             — overnight gap: Open / prev Close - 1
      intraday_ret    — Close / Open - 1
      hl_range        — (High - Low) / ATR(14) — normalized intraday range
      dist_52w_high   — Close / rolling 252d High - 1  (≤0)
      dist_52w_low    — Close / rolling 252d Low  - 1  (≥0)
      hl_spread       — (High - Low) / Close  — fills slot from Group E
    """
    eps = 1e-9
    df["gap"]          = open_ / (close.shift(1) + eps) - 1
    df["intraday_ret"] = close / (open_  + eps) - 1

    atr = _atr(high, low, close, 14)
    df["hl_range"] = (high - low) / (atr + eps)

    df["dist_52w_high"] = close / (close.rolling(252).max() + eps) - 1
    df["dist_52w_low"]  = close / (close.rolling(252).min() + eps) - 1

    df["hl_spread"] = (high - low) / (close + eps)   # overwrite NaN from Group E


# ══════════════════════════════════════════════════════
#  GROUP G — MARKET-WIDE CONDITION SIGNALS
# ══════════════════════════════════════════════════════

def compute_market_signals(spy: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """
    8 market-wide features broadcast to every stock on every date:

      spy_ret_1d, spy_ret_5d, spy_ret_20d  — SPY momentum
      spy_vol_20d                           — SPY realized vol (regime)
      vix_close_norm                        — VIX / 20d rolling mean  (relative level)
      vix_change_1d                         — daily VIX pct change
      vix_regime                            — 1 if VIX > 20, else 0  (fear signal)
      vix_mom_5d                            — 5-day VIX momentum
    """
    spy_log = np.log(spy / spy.shift(1))

    signals = pd.DataFrame(index=spy.index)
    signals["spy_ret_1d"]      = spy.pct_change(1)
    signals["spy_ret_5d"]      = spy.pct_change(5)
    signals["spy_ret_20d"]     = spy.pct_change(20)
    signals["spy_vol_20d"]     = spy_log.rolling(20).std() * np.sqrt(252)
    signals["vix_close_norm"]  = vix / (vix.rolling(20).mean() + 1e-9)
    signals["vix_change_1d"]   = vix.pct_change(1)
    signals["vix_regime"]      = (vix > 20).astype(float)
    signals["vix_mom_5d"]      = vix.pct_change(5)

    return signals.dropna()


# ══════════════════════════════════════════════════════
#  MASTER ENTRY POINT
# ══════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "A_returns_momentum": [
        "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
        "mom_60d", "mom_126d", "mom_252d",
    ],
    "B_trend": [
        "price_to_sma20", "price_to_sma50", "price_to_sma200",
        "sma20_to_sma50", "sma50_to_sma200",
        "ema12_to_ema26", "price_to_ema20", "slope_20d",
    ],
    "C_volatility": [
        "realized_vol_5d", "realized_vol_20d", "realized_vol_60d",
        "vol_regime", "atr_pct",
    ],
    "D_oscillators": [
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "bb_pct", "bb_width", "stoch_k",
    ],
    "E_volume_liquidity": [
        "vol_norm", "vol_momentum", "obv_norm", "vpt", "amihud", "hl_spread",
    ],
    "F_price_structure": [
        "gap", "intraday_ret", "hl_range",
        "dist_52w_high", "dist_52w_low",
        # hl_spread also lives here but is already counted in Group E
    ],
    "G_market_signals": [
        "spy_ret_1d", "spy_ret_5d", "spy_ret_20d", "spy_vol_20d",
        "vix_close_norm", "vix_change_1d", "vix_regime", "vix_mom_5d",
    ],
}

ALL_STOCK_FEATURES = (
    FEATURE_GROUPS["A_returns_momentum"] +
    FEATURE_GROUPS["B_trend"] +
    FEATURE_GROUPS["C_volatility"] +
    FEATURE_GROUPS["D_oscillators"] +
    FEATURE_GROUPS["E_volume_liquidity"] +
    FEATURE_GROUPS["F_price_structure"]
)   # 9+8+5+7+6+5 = 40  (hl_spread counted once)

ALL_FEATURES = ALL_STOCK_FEATURES + FEATURE_GROUPS["G_market_signals"]  # 48 total


def compute_stock_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all per-stock features from raw OHLCV data.

    Parameters
    ----------
    ohlcv : (date, ticker) MultiIndex DataFrame with columns
            [Open, High, Low, Close, Volume]

    Returns
    -------
    (date, ticker) MultiIndex DataFrame with ALL_STOCK_FEATURES columns
    """
    all_feats = []

    tickers = ohlcv.index.get_level_values("ticker").unique()
    n = len(tickers)
    print(f"  Computing features for {n} tickers …")

    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"    … {i}/{n}")

        grp    = ohlcv.xs(ticker, level="ticker").sort_index()
        close  = grp["Close"]
        open_  = grp["Open"]
        high   = grp["High"]
        low    = grp["Low"]
        volume = grp["Volume"]
        ret    = close.pct_change(1)

        f = pd.DataFrame(index=grp.index)

        add_returns_momentum(f, close)
        add_trend(f, close)
        add_volatility(f, close, high, low)
        add_oscillators(f, close, high, low)
        add_volume_liquidity(f, close, volume, ret)
        add_price_structure(f, open_, close, high, low)

        f["ticker"] = ticker
        all_feats.append(f)

    features = (
        pd.concat(all_feats)
          .reset_index()
          .set_index(["ticker", "date"])
          .sort_index()
    )
    # Swap to (date, ticker) to match original pipeline convention
    features = features.swaplevel("ticker", "date").sort_index()
    return features


def print_feature_summary() -> None:
    total_stock   = len(ALL_STOCK_FEATURES)
    total_market  = len(FEATURE_GROUPS["G_market_signals"])
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          Feature Library Summary                     ║")
    print("╠══════════════════════════════════════════════════════╣")
    for group, cols in FEATURE_GROUPS.items():
        label = group.replace("_", " ").title()
        print(f"║  {label:<30} {len(cols):>3} features  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  {'Total (per-stock)':<30} {total_stock:>3} features  ║")
    print(f"║  {'Total (incl. market signals)':<30} {total_stock + total_market:>3} features  ║")
    print("╚══════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    print_feature_summary()
    for group, cols in FEATURE_GROUPS.items():
        print(f"\n  {group}")
        for c in cols:
            print(f"    • {c}")