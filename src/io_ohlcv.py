
import glob
import pandas as pd
from .time_utils import force_ts_cols
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

def read_any_ohlcv(path_pattern: str) -> pd.DataFrame:
    parts = sorted(glob.glob(path_pattern))
    if not parts:
        raise FileNotFoundError(f"No files match: {path_pattern}")
    dfs = []
    for p in tqdm(parts, desc="Reading OHLCV"):
        df = pd.read_csv(p)
        df = force_ts_cols(df, out_col="ts")
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
                df = df.rename(columns={vol_cands[0]:"volume"})
            if "volume" not in df.columns:
                df["volume"] = 0.0
            dfs.append(df[["ts","open","high","low","close","volume"]])
            continue
        price_cands = [c for c in ["price","p","last","close"] if c in cols]
        if not price_cands:
            raise ValueError(f"{p} lacks OHLCV and no tick price col")
        qty_cands = [c for c in ["qty","quantity","size","amount","volume","q"] if c in cols]
        t = df[["ts", price_cands[0]] + ([qty_cands[0]] if qty_cands else [])].copy()
        t = t.rename(columns={price_cands[0]:"price"})
        if qty_cands: t = t.rename(columns={qty_cands[0]:"qty"})
        g = t.groupby(t["ts"].dt.floor("min"))
        o = g["price"].first(); h = g["price"].max(); l = g["price"].min(); c = g["price"].last()
        v = g["qty"].sum() if "qty" in t.columns else pd.Series(0.0, index=o.index, name="qty")
        ohlcv = (pd.DataFrame({"ts": o.index, "open": o.values, "high": h.values,
                               "low": l.values, "close": c.values, "volume": v.values}))
        dfs.append(ohlcv)
    out = (pd.concat(dfs, ignore_index=True)
           .dropna(subset=["open","high","low","close"])
           .sort_values("ts").drop_duplicates("ts"))
    return out
