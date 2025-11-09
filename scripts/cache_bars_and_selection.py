from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, glob, json
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import yaml
except Exception:
    raise SystemExit("pip install pyyaml")

# --- robust timestamp + loader (mirrors your backtester) ---
def _force_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # choose time column
    for c in ["ts","timestamp","open_time","date","datetime","time","open_time_ms"]:
        if c in df.columns:
            tcol = c; break
    else:
        raise ValueError(f"No recognizable time column in {df.columns.tolist()}")
    s = df[tcol]
    if s.dtype.kind in "iuf":
        ser = pd.to_numeric(s, errors="coerce")
        unit = "ms" if float(ser.dropna().median()) > 1e11 else "s"
        ts = pd.to_datetime(ser, unit=unit, utc=True, errors="coerce")
    else:
        try:
            ts = pd.to_datetime(s, utc=True, format="mixed")
        except TypeError:
            ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps; first {df.loc[ts.isna(), tcol].head().tolist()}")
    df["ts"] = ts
    if tcol != "ts": df = df.drop(columns=[tcol])
    return df

def _read_any_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _force_ts(df)
    cols = set(df.columns)
    alt = {}
    if "o" in cols and "open"  not in cols: alt["o"] = "open"
    if "h" in cols and "high"  not in cols: alt["h"] = "high"
    if "l" in cols and "low"   not in cols: alt["l"] = "low"
    if "c" in cols and "close" not in cols: alt["c"] = "close"
    if alt: df = df.rename(columns=alt); cols = set(df.columns)
    if {"open","high","low","close"}.issubset(cols):
        v = "volume" if "volume" in cols else None
        if not v:
            for vc in ["vol","v","base_volume","quote_volume","qty","quantity"]:
                if vc in cols: v = vc; break
        if v and v != "volume": df = df.rename(columns={v:"volume"})
        if "volume" not in df.columns: df["volume"] = 0.0
        out = df[["ts","open","high","low","close","volume"]].sort_values("ts")
        return out.dropna(subset=["open","high","low","close"])
    # tick → 1m
    price_cands = [c for c in ["price","p","last","close"] if c in cols]
    if not price_cands: raise ValueError(f"{path} lacks OHLCV and no tick price col")
    qty_cands = [c for c in ["qty","quantity","size","amount","volume","q"] if c in cols]
    t = df[["ts", price_cands[0]] + ([qty_cands[0]] if qty_cands else [])].copy()
    t = t.rename(columns={price_cands[0]:"price"})
    if qty_cands: t = t.rename(columns={qty_cands[0]:"qty"})
    g = t.groupby(t["ts"].dt.floor("min"))
    o = g["price"].first(); h = g["price"].max(); l = g["price"].min(); c = g["price"].last()
    v = g["qty"].sum() if "qty" in t.columns else pd.Series(0.0, index=o.index, name="qty")
    out = pd.DataFrame({"ts": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values, "volume": v.values})
    return out.dropna(subset=["open","high","low","close"]).sort_values("ts")

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def compute_atr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.sort_values("ts").copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = _ema(tr, span=window)
    return df

# --- main ---
root = Path(".").resolve()
cfg = yaml.safe_load(open("configs/vol_bracket.yaml", "r", encoding="utf-8"))
files = cfg.get("files", {})
atr_window = int(cfg.get("strategy", {}).get("atr_window", 60))

# --- expand all possible file inputs (string, list, dir, glob) ---
def _expand_any(x):
    import os, glob
    out = []
    if not x:
        return out
    if isinstance(x, list):
        for y in x:
            out.extend(_expand_any(y))
        return out
    if isinstance(x, str):
        # if contains wildcards → expand
        if any(ch in x for ch in "*?[]"):
            out.extend(glob.glob(x))
        # if it's a directory → take all csvs inside
        elif os.path.isdir(x):
            out.extend(glob.glob(os.path.join(x, "*.csv")))
        else:
            out.append(x)
        return out
    return out  # unknown type → ignore

pats = []
pats += _expand_any(files.get("ohlcv"))
pats += _expand_any(files.get("ohlcv_pattern"))
pats += _expand_any(files.get("ohlcv_parts"))

# Fallbacks if config didn’t specify
if not pats:
    import glob
    pats = glob.glob("data/ohlcv/*.csv")
if not pats:
    import glob
    pats = glob.glob("data/ticks/**/*.csv", recursive=True)

# Dedup + sort, and WARN if nothing found
pats = sorted(set(pats))
if not pats:
    raise SystemExit("No OHLCV/tick CSVs found. Populate files.* in configs/vol_bracket.yaml or put data into data/ohlcv/.")

print(f"[Files] resolved {len(pats)} CSVs")
for p in pats[:5]:
    print("[Files] sample:", p)

bars = pd.concat([_read_any_ohlcv(p) for p in sorted(set(pats))], ignore_index=True)
bars = compute_atr(bars, atr_window)
bars.to_csv("results/ohlcv_1m_cache.csv", index=False)
print(f"[Cache] wrote results/ohlcv_1m_cache.csv (n={len(bars)})")

# alerts
## --- build selection_log.csv with robust fallback ---
sel_rows = []
alerts_path = Path("results/alerts_used.csv")
if not alerts_path.exists():
    raise SystemExit("results/alerts_used.csv not found. Run run_vol_bracket.py once to save alerts_used.csv.")

alerts = pd.read_csv(alerts_path)
if "t_alert" not in alerts.columns:
    alerts = alerts.rename(columns={"ts": "t_alert"})
alerts["t_alert"] = pd.to_datetime(alerts["t_alert"], utc=True).dt.floor("min")

bars_i = bars.set_index(pd.to_datetime(bars["ts"], utc=True)).sort_index()

try:
    # primary path: use your selector
    from src.engine.direction import DirCfg, DriftCfg, CvdCfg, select_direction

    cfg = yaml.safe_load(open("configs/vol_bracket.yaml", "r", encoding="utf-8"))
    d_cfg = cfg.get("strategy", {}).get("direction", {})

    dir_cfg = DirCfg(
        mode=d_cfg.get("mode", "hybrid_and"),
        drift=DriftCfg(
            M_min=int(d_cfg.get("drift", {}).get("M_min", 3)),
            eps_atr=float(d_cfg.get("drift", {}).get("eps_atr", 0.08)),
        ),
        cvd=CvdCfg(
            source=d_cfg.get("cvd", {}).get("source", "bar_proxy"),
            baseline_min=int(d_cfg.get("cvd", {}).get("baseline_min", 240)),
            z_thr=float(d_cfg.get("cvd", {}).get("z_thr", 0.6)),
            z_strong=float(d_cfg.get("cvd", {}).get("z_strong", 1.2)),
        ),
        anchor_price=d_cfg.get("anchor_price", "select"),
    )

    for ts in alerts["t_alert"]:
        if ts < bars_i.index.min() or ts > bars_i.index.max():
            continue
        atr_ref = float(bars_i.loc[ts, "ATR"])
        bias, info = select_direction(bars_i, ts, atr_ref, dir_cfg)
        sel_rows.append({
            "t_alert": ts,
            "bias": int(bias) if bias else 0,
            "t_sel": info.get("t_sel"),
            "reason": info.get("reason"),
            "drift": info.get("drift"),
            "drift_dir": info.get("drift_dir"),
            "drift_sig": info.get("drift_sig"),
            "cvd_z": info.get("cvd_z"),
            "cvd_dir": info.get("cvd_dir"),
            "cvd_sig": info.get("cvd_sig"),
        })

    print("[Cache] selection built via src.engine.direction")

except Exception as e:
    print(f"[WARN] Direction import/selection failed ({e}). Falling back to trades.")
    # fallback: derive bias from your latest trades (if present); else bias=0
    tr_p = Path("results/vol_bracket_trades.csv")
    if tr_p.exists():
        tr = pd.read_csv(tr_p)
        tr["t_alert"] = pd.to_datetime(tr["t_alert"], utc=True).dt.floor("min")
        tr = tr.sort_values("t_alert").drop_duplicates(subset=["t_alert"], keep="last")
        bias_map = dict(zip(tr["t_alert"], tr["bias"]))
    else:
        bias_map = {}

    for ts in alerts["t_alert"]:
        sel_rows.append({
            "t_alert": ts,
            "bias": int(bias_map.get(ts, 0)),
            "t_sel": ts,          # anchor at alert if we don’t know t_sel
            "reason": "fallback",
            "drift": None, "drift_dir": None, "drift_sig": None,
            "cvd_z": None, "cvd_dir": None, "cvd_sig": None,
        })

sel = pd.DataFrame(sel_rows)
sel["t_sel"] = pd.to_datetime(sel["t_sel"], utc=True, errors="coerce").dt.floor("min")
Path("results").mkdir(parents=True, exist_ok=True)
sel.to_csv("results/selection_log.csv", index=False)
print(f"[Cache] wrote results/selection_log.csv (n={len(sel)})")
