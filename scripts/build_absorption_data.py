import argparse, os, glob, hashlib
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm  # progress bar
except Exception:  # fallback no-op tqdm
    def tqdm(it, total=None, desc=None):
        it = list(it)
        n = len(it) if total is None else total
        print(f"{desc or 'Progress'}: 0/{n}", end="\r")
        for i, x in enumerate(it, 1):
            if i % max(1, n // 10 or 1) == 0:
                print(f"{desc or 'Progress'}: {i}/{n}", end="\r")
            yield x
        print(f"{desc or 'Progress'}: {n}/{n}")

# ---- CONFIG-LITE ----
DEFAULT_MICRO_TF = "5s"  # use 5s to stabilize; switch to "1s" for final pass
ATR_WINDOW = 60
GATE_PAD_S = 15

# ---- HELPERS ----
def to_utc(ts_series: pd.Series) -> pd.Series:
    """
    Robust timestamp parser:
    - numeric >= 1e14 and <1e17 -> epoch microseconds
    - numeric >= 1e11 and <1e14 -> epoch milliseconds
    - numeric <  1e11            -> epoch seconds
    - else parse as ISO strings
    """
    s_num = pd.to_numeric(ts_series, errors="coerce")
    is_num = s_num.notna()
    # initialize UTC-aware datetime series
    out = pd.Series(pd.NaT, index=ts_series.index, dtype="datetime64[ns, UTC]")

    # numeric epochs
    sn = s_num[is_num]
    us_mask = (sn >= 1e14) & (sn < 1e17)
    ms_mask = (sn >= 1e11) & (sn < 1e14)
    s_mask  = ~us_mask & ~ms_mask
    if us_mask.any():
        out.loc[sn.index[us_mask]] = pd.to_datetime(sn[us_mask].astype("int64"), unit="us", utc=True)
    if ms_mask.any():
        out.loc[sn.index[ms_mask]] = pd.to_datetime(sn[ms_mask].astype("int64"), unit="ms", utc=True)
    if s_mask.any():
        out.loc[sn.index[s_mask]] = pd.to_datetime(sn[s_mask].astype("int64"), unit="s", utc=True)

    # strings (ISO and variants)
    is_str = ~is_num
    if is_str.any():
        out.loc[ts_series.index[is_str]] = pd.to_datetime(ts_series[is_str], utc=True, errors="coerce")
    return out
def read_alerts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t_alert" not in df.columns:
        raise ValueError("alerts CSV must have column 't_alert'")
    df["t_alert"] = pd.to_datetime(df["t_alert"], utc=True)
    df = df.dropna(subset=["t_alert"]).drop_duplicates(subset=["t_alert"]).sort_values("t_alert")
    return df

def coalesce(intervals):
    intervals = sorted(intervals)
    out = []
    for s, e in intervals:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return [(a, b) for a, b in out]

def build_gate_windows(alerts: pd.DataFrame, horizon_min: int, pad_s: int = GATE_PAD_S):
    pad = pd.Timedelta(seconds=pad_s)
    raw = [(t, t + pd.Timedelta(minutes=horizon_min)) for t in alerts["t_alert"]]
    return coalesce([(s - pad, e + pad) for s, e in raw])

# tick IO & normalization
TICK_TS_ALIASES = {"ts","timestamp","time","date","open_time","open_time_ms","ts_ms"}
TICK_PX_ALIASES = {"price","p","last","close","trade_price"}
TICK_QTY_ALIASES= {"qty","size","amount","volume","q","quantity","base_volume","vol","v"}
TICK_SIDE_ALIASES={"is_buyer_maker","is_buyer_maker_flag","isBuyerMaker","is_buy_maker","buyer_maker","is_maker","maker_flag","side"}

def _pick(df, cands):
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in cols: return cols[c.lower()]
    return None

def _norm_ticks(df: pd.DataFrame) -> pd.DataFrame:
    ts = _pick(df, TICK_TS_ALIASES); px = _pick(df, TICK_PX_ALIASES); q = _pick(df, TICK_QTY_ALIASES)
    ib = _pick(df, TICK_SIDE_ALIASES)
    if ts is None or px is None:
        raise ValueError("Tick CSV missing ts/price columns")
    out = df.rename(columns={ts:"ts", px:"price"})
    if q is None:
        # derive naive unit size
        out["qty"] = 1.0
    else:
        out = out.rename(columns={q:"qty"})
    if ib is not None:
        out = out.rename(columns={ib:"is_buyer_maker"})
        # map SELL→1, BUY→0 if column contains strings
        if out["is_buyer_maker"].dtype == object:
            out["is_buyer_maker"] = out["is_buyer_maker"].str.upper().map({"SELL":1,"BUY":0}).fillna(1).astype("int8")
        else:
            out["is_buyer_maker"] = pd.to_numeric(out["is_buyer_maker"], errors="coerce").fillna(1).astype("int8")
    else:
        out["is_buyer_maker"] = np.nan  # will infer later
    out["price"] = pd.to_numeric(out["price"], errors="coerce").astype("float32")
    out["qty"]   = pd.to_numeric(out["qty"],   errors="coerce").astype("float32")
    out["ts"] = to_utc(out["ts"])
    out = out.dropna(subset=["ts","price","qty"]).sort_values("ts")
    # infer maker flag if missing
    if out["is_buyer_maker"].isna().all():
        sgn = np.sign(out["price"].diff().fillna(0.0))
        out["is_buyer_maker"] = (sgn <= 0).astype("int8")  # down/same → seller
    return out[["ts","price","qty","is_buyer_maker"]]

def read_ticks(glob_pat: str) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No tick CSVs matched glob: {glob_pat}")
    print(f"[Load] tick files matched: {len(paths)}")
    parts = []
    for p in tqdm(paths, desc="Reading ticks", total=len(paths)):
        df = pd.read_csv(p, low_memory=False)
        parts.append(_norm_ticks(df))
    t = pd.concat(parts, ignore_index=True).sort_values("ts")
    return t

def mask_ticks_to_windows(ticks: pd.DataFrame, windows) -> pd.Series:
    m = pd.Series(False, index=ticks.index)
    for s, e in windows:
        m |= (ticks["ts"] >= s) & (ticks["ts"] <= e)
    return m

# microbars via resample (fast)
def ticks_to_microbars(ticks: pd.DataFrame, micro_tf: str = DEFAULT_MICRO_TF) -> pd.DataFrame:
    t = ticks.set_index("ts")
    ohlc = t["price"].resample(micro_tf).ohlc()
    buy  = ((1 - t["is_buyer_maker"]) * t["qty"]).resample(micro_tf).sum().rename("buy_vol")
    sell = ((    t["is_buyer_maker"]) * t["qty"]).resample(micro_tf).sum().rename("sell_vol")
    df = ohlc.join([buy, sell], how="outer").sort_index()
    df[["buy_vol","sell_vol"]] = df[["buy_vol","sell_vol"]].fillna(0.0)
    df[["open","high","low","close"]] = df[["open","high","low","close"]].ffill()
    df["delta"] = df["buy_vol"] - df["sell_vol"]
    df["tr"] = (df["high"] - df["low"]).fillna(0.0)
    return df.dropna(subset=["open","high","low","close"])

# minute build + ATR
def build_minutes_from_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    t = ticks.set_index("ts")
    ohlc = t["price"].resample("1min").ohlc().rename(columns=str)
    return ohlc

def compute_atr60(minute_bars: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    # classic Wilder-ish ATR using close to close TR proxy
    hi = minute_bars["high"]; lo = minute_bars["low"]; cl = minute_bars["close"]
    prev_close = cl.shift(1)
    tr = pd.concat([
        (hi - lo).abs(),
        (hi - prev_close).abs(),
        (lo - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=max(5, window//3)).mean().rename("ATR")

def attach_atr60_to_micro(micro: pd.DataFrame, minute_atr: pd.Series) -> pd.DataFrame:
    atr = minute_atr.reindex(minute_atr.index).ffill()
    micro = micro.copy()
    micro["ATR60"] = atr.reindex(micro.index, method="ffill")
    return micro

def hash_windows(windows) -> str:
    s = "".join([f"{int(s.value)}-{int(e.value)};" for s,e in windows])  # ns to int
    return hashlib.md5(s.encode()).hexdigest()[:12]

# ---- MAIN BUILD ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks_glob", required=True)
    ap.add_argument("--alerts_csv", required=True)
    ap.add_argument("--out_dir", default="results/cache")
    ap.add_argument("--micro_tf", default=DEFAULT_MICRO_TF)
    ap.add_argument("--horizon_min", type=int, default=180)
    ap.add_argument("--flip_maker_flag", action="store_true", help="try if your SELL/BUY semantics are reversed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_min = os.path.join(args.out_dir, "minutes")
    out_micro = os.path.join(args.out_dir, "microbars")
    os.makedirs(out_min, exist_ok=True); os.makedirs(out_micro, exist_ok=True)

    alerts = read_alerts(args.alerts_csv)
    windows = build_gate_windows(alerts, args.horizon_min, GATE_PAD_S)
    print(f"[Alerts] {len(alerts)} rows → {len(windows)} windows (coalesced)")

    ticks = read_ticks(args.ticks_glob)
    if args.flip_maker_flag:
        ticks["is_buyer_maker"] = 1 - ticks["is_buyer_maker"]
    print(f"[Ticks] {len(ticks):,} rows")
    # Range sanity
    if len(ticks):
        try:
            print(f"[Range] tick ts: {ticks['ts'].min()}  →  {ticks['ts'].max()}")
        except Exception:
            pass
    if len(windows):
        print(f"[Range] alert windows: {windows[0][0]}  →  {windows[-1][1]}")

    gate_mask = mask_ticks_to_windows(ticks, windows)
    tg = ticks.loc[gate_mask].copy()
    print(f"[Gate] ticks in windows: {len(tg):,} ({100*len(tg)/max(1,len(ticks)):.1f}%)")

    # Minutes + ATR from gated ticks (guarantees coverage)
    minutes = build_minutes_from_ticks(tg)
    minutes["ATR"] = compute_atr60(minutes, ATR_WINDOW)
    # small forward fill to bridge sparse minutes inside pads
    minutes[["open","high","low","close"]] = minutes[["open","high","low","close"]].ffill()
    cov = minutes["ATR"].notna().mean()
    print(f"[ATR] minute ATR coverage: {cov:.0%}")

    # Save minutes
    min_path = os.path.join(out_min, f"minutes_{hash_windows(windows)}.parquet")
    minutes.to_parquet(min_path)
    print(f"[Write] minutes+ATR → {min_path}")

    # Microbars inside gates + ATR60
    micro = ticks_to_microbars(tg, args.micro_tf)
    micro = attach_atr60_to_micro(micro, minutes["ATR"])
    cov_micro = micro["ATR60"].notna().mean() if len(micro) else 0.0
    print(f"[ATR] micro ATR60 coverage: {cov_micro:.0%}  | micro rows: {len(micro):,}")

    micro_path = os.path.join(out_micro, f"micro_{args.micro_tf}_{hash_windows(windows)}.parquet")
    micro.to_parquet(micro_path)
    print(f"[Write] microbars → {micro_path}")

    # Windows manifest (for reproducibility)
    wdf = pd.DataFrame({"start":[s for s,_ in windows], "end":[e for _,e in windows]})
    win_path = os.path.join(args.out_dir, f"windows_{hash_windows(windows)}.csv")
    wdf.to_csv(win_path, index=False)
    print(f"[Write] windows manifest → {win_path}")

if __name__ == "__main__":
    main()
