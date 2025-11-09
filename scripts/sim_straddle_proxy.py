import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse, json, os, glob
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, total=None, desc=None):
        try:
            n = len(iterable) if total is None else total
        except Exception:
            n = None
        count = 0
        for item in iterable:
            count += 1
            if n is not None:
                print(f"\r{desc + ': ' if desc else ''}{count}/{n}", end="", flush=True)
            elif count % 100 == 0:
                print(f"\r{desc + ': ' if desc else ''}{count}", end="", flush=True)
            yield item
        print()

# ---- shared helpers ----
def _force_ts_cols(df: pd.DataFrame, time_candidates=None, out_col="ts") -> pd.DataFrame:
    """Normalize a DataFrame to contain a UTC datetime column named `out_col`."""
    if time_candidates is None:
        time_candidates = ["ts","t_alert","timestamp","open_time","date","datetime","time","open_time_ms"]
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in time_candidates if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"No recognizable time column in {list(df.columns)}")
    s = df[tcol]
    if s.dtype.kind in "iuf":  # numeric epoch
        ser = pd.to_numeric(s, errors="coerce")
        med = float(ser.dropna().median())
        unit = "ms" if med > 1e11 else "s"
        ts = pd.to_datetime(ser, unit=unit, utc=True, errors="coerce")
    else:
        try:
            ts = pd.to_datetime(s, utc=True, format="mixed")  # pandas >= 2.0
        except TypeError:
            ts = pd.to_datetime(s.astype(str), utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps; first few bad: {s[ts.isna()].head().tolist()}")
    df[out_col] = ts.dt.floor("min")
    if tcol != out_col:
        df = df.drop(columns=[tcol])
    return df

def _read_any_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _force_ts_cols(df, out_col="ts")
    cols = set(df.columns)

    alt = {}
    if "o" in cols and "open"  not in cols: alt["o"] = "open"
    if "h" in cols and "high"  not in cols: alt["h"] = "high"
    if "l" in cols and "low"   not in cols: alt["l"] = "low"
    if "c" in cols and "close" not in cols: alt["c"] = "close"
    if alt:
        df = df.rename(columns=alt); cols = set(df.columns)

    if {"open","high","low","close"}.issubset(cols):
        vol_cands = [c for c in ["volume","vol","v","base_volume","quote_volume"] if c in cols]
        if vol_cands and "volume" not in cols:
            df = df.rename(columns={vol_cands[0]: "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return (df[["ts","open","high","low","close","volume"]]
                  .sort_values("ts")
                  .dropna(subset=["open","high","low","close"]))

    price_cands = [c for c in ["price","p","last","close"] if c in cols]
    if not price_cands:
        raise ValueError(f"{path} lacks OHLCV and no tick price column found.")
    qty_cands = [c for c in ["qty","quantity","size","amount","volume","q"] if c in cols]

    t = df[["ts", price_cands[0]] + ([qty_cands[0]] if qty_cands else [])].copy()
    t = t.rename(columns={price_cands[0]:"price"})
    if qty_cands: t = t.rename(columns={qty_cands[0]:"qty"})

    g = t.groupby(t["ts"].dt.floor("min"))
    o = g["price"].first(); h = g["price"].max(); l = g["price"].min(); c = g["price"].last()
    v = g["qty"].sum() if "qty" in t.columns else pd.Series(0.0, index=o.index, name="qty")
    return (pd.DataFrame({"ts": o.index, "open": o.values, "high": h.values,
                          "low": l.values, "close": c.values, "volume": v.values})
              .dropna(subset=["open","high","low","close"]) 
              .sort_values("ts"))

def load_ohlcv_from_glob(pattern: str) -> pd.DataFrame:
    parts = sorted(glob.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No files match: {pattern}")
    df = (pd.concat([_read_any_ohlcv(p) for p in tqdm(parts, desc="Reading OHLCV")], ignore_index=True)
            .sort_values("ts")
            .drop_duplicates("ts"))
    return df

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vol_bracket.yaml")
    ap.add_argument("--ohlcv", default="", help="Override OHLCV path or glob; otherwise use config.files.ohlcv")
    ap.add_argument("--alerts_csv", default="", help="CSV with 't_alert' to force alert set")
    ap.add_argument("--window_min", type=int, default=180)
    ap.add_argument("--premium_mode", choices=["quantile","fixed"], default="quantile")
    ap.add_argument("--q", type=float, default=0.50)
    ap.add_argument("--premium_mult", type=float, default=0.60)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) if args.config and os.path.exists(args.config) else {"files":{}}
    pattern = args.ohlcv or (cfg.get("files", {}).get("ohlcv") or "")
    if not pattern:
        raise ValueError("Missing OHLCV path/pattern. Provide --ohlcv or files.ohlcv in config.")
    ohlcv = load_ohlcv_from_glob(pattern)

    bars = ohlcv.set_index("ts").sort_index()
    t0, t1 = bars.index.min(), bars.index.max()
    H = pd.Timedelta(minutes=args.window_min)

    # Alerts
    if args.alerts_csv:
        adf = pd.read_csv(args.alerts_csv)
        adf = _force_ts_cols(adf, time_candidates=["t_alert","ts","timestamp","time","date"], out_col="t_alert")
        alerts = pd.DatetimeIndex(adf["t_alert"]).sort_values().drop_duplicates()
    else:
        raise ValueError("--alerts_csv is required to replay canonical alerts")

    # Intersect with coverage + horizon
    alerts = alerts[(alerts >= t0) & (alerts <= (t1 - H))]
    if len(alerts) == 0:
        raise RuntimeError("No usable alerts within OHLCV range and horizon")

    # Precompute abs log returns and realized volatility over horizon H for all timestamps
    close = bars["close"].astype(float)
    abs_lr = (np.log(close).diff().abs()).rename("abs_lr")
    rv_all = abs_lr.shift(-1).rolling(args.window_min, min_periods=args.window_min).sum()

    # Premium baseline
    if args.premium_mode == "quantile":
        base = float(rv_all.dropna().quantile(args.q))
        premium_val = args.premium_mult * base
    else:  # fixed absolute premium
        premium_val = args.premium_mult

    out = []
    for ta in tqdm(alerts, desc="Simulating proxy"):
        rv = rv_all.reindex(bars.index).loc[ta]
        if not np.isfinite(rv):
            continue
        premium = premium_val
        out.append(dict(t_alert=ta, premium=float(premium), rv=float(rv), pnl=float(rv - premium)))

    df = pd.DataFrame(out).sort_values("t_alert")
    pathlib.Path("results").mkdir(exist_ok=True, parents=True)
    df.to_csv("results/straddle_trades.csv", index=False)
    print(json.dumps({"n": int(len(df)), "sum_pnl": float(df["pnl"].sum())}, indent=2, default=float))

if __name__ == "__main__":
    main()
