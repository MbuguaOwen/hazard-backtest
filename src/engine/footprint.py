import pandas as pd
import numpy as np

def ticks_to_microbars(ticks: pd.DataFrame, micro_tf: str = "1s") -> pd.DataFrame:
    """
    Vectorized microbar aggregation using resample for speed.
    ticks: requires columns ts (datetime64[ns, UTC]), price, qty, optional is_buyer_maker.
    """
    if ticks.empty:
        return pd.DataFrame(columns=["open","high","low","close","buy_vol","sell_vol","delta","tr"])  
    t = ticks[["ts","price","qty"] + (["is_buyer_maker"] if "is_buyer_maker" in ticks.columns else [])].copy()
    t = t.sort_values("ts").set_index("ts")
    if "is_buyer_maker" not in t.columns:
        sgn = np.sign(t["price"].diff().fillna(0.0))
        t["is_buyer_maker"] = (sgn <= 0).astype("int8")
    else:
        t["is_buyer_maker"] = t["is_buyer_maker"].astype("int8")
    # OHLC from trade price
    ohlc = t["price"].resample(micro_tf).ohlc()
    # Buy/Sell volumes via masks
    buy_qty = (1 - t["is_buyer_maker"]) * t["qty"]
    sell_qty = (t["is_buyer_maker"]) * t["qty"]
    buy = buy_qty.resample(micro_tf).sum().rename("buy_vol")
    sell = sell_qty.resample(micro_tf).sum().rename("sell_vol")
    df = ohlc.join([buy, sell], how="outer")
    df[["buy_vol","sell_vol"]] = df[["buy_vol","sell_vol"]].fillna(0.0)
    df[["open","high","low","close"]] = df[["open","high","low","close"]].ffill()
    df["delta"] = df["buy_vol"] - df["sell_vol"]
    df["tr"] = (df["high"] - df["low"]).fillna(0.0)
    return df.dropna(subset=["open","high","low","close"])  

def attach_minute_ATR60_to_micro(micro: pd.DataFrame, minute_ohlcv: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    """
    Align 1m ATR60 to microbar index via forward-fill.
    minute_ohlcv indexed by minute ts. atr_series indexed same as minute_ohlcv.
    Output micro with column 'ATR60' aligned.
    """
    atr = atr_series.reindex(minute_ohlcv.index).ffill()
    s = atr.reindex(micro.index, method="ffill")
    out = micro.copy()
    out["ATR60"] = s
    return out

def rolling_z(series: pd.Series, lookback: int) -> pd.Series:
    r = series.rolling(lookback, min_periods=max(5, lookback//3))
    mu = r.mean(); sd = r.std(ddof=1)
    z = (series - mu) / sd.replace({0.0: np.nan})
    return z

def rolling_lambda(micro: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Kyle’s lambda proxy over microbars: cov(Δp, ΔCVD)/var(ΔCVD).
    Vectorized implementation using rolling means to avoid per-window apply.
    ΔCVD approximated by per-microbar delta, Δp is price change per microbar.
    """
    dp = micro["close"].diff()
    dcvd = micro["delta"]
    # Precompute products
    xy = dp * dcvd
    y2 = dcvd * dcvd
    r = lookback
    mp = max(10, r // 2)
    mean_x = dp.rolling(r, min_periods=mp).mean()
    mean_y = dcvd.rolling(r, min_periods=mp).mean()
    mean_xy = xy.rolling(r, min_periods=mp).mean()
    mean_y2 = y2.rolling(r, min_periods=mp).mean()
    cov_xy = mean_xy - (mean_x * mean_y)
    var_y = mean_y2 - (mean_y * mean_y)
    lam = cov_xy / var_y.replace(0.0, np.nan)
    return lam
