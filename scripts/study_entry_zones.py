import pandas as pd, numpy as np
from pathlib import Path

bars_p = Path("results/ohlcv_1m_cache.csv")
sel_p  = Path("results/selection_log.csv")
if not bars_p.exists() or not sel_p.exists():
    raise SystemExit("Pre-cache first: python scripts/cache_bars_and_selection.py")

bars = pd.read_csv(bars_p)
bars["ts"] = pd.to_datetime(bars["ts"], utc=True); bars = bars.set_index("ts").sort_index()
bars["ATR"] = pd.to_numeric(bars["ATR"], errors="coerce")

sel = pd.read_csv(sel_p)
sel["t_alert"] = pd.to_datetime(sel["t_alert"], utc=True).dt.floor("min")
sel["t_sel"]   = pd.to_datetime(sel["t_sel"], utc=True, errors="coerce").dt.floor("min")

H_MIN = 180
def window(ts): return bars.loc[ts : ts + pd.Timedelta(minutes=H_MIN)]
def anchor_price(ts):
    row = bars.loc[bars.index[bars.index.searchsorted(ts)]]
    return float(row["close"]), float(row["ATR"])

records = []
MIN_WAIT = 5
RANGE_MIN = 6
GUARD_MIN = 8

for _, r in sel.iterrows():
    bias = int(r.get("bias",0))
    if bias == 0: continue
    t0 = r["t_sel"] if pd.notna(r["t_sel"]) else r["t_alert"]
    if pd.isna(t0): t0 = r["t_alert"]
    p0, atr0 = anchor_price(t0)
    w = window(t0)
    if w.empty: continue

    w_form = w.iloc[:RANGE_MIN] if len(w) >= RANGE_MIN else w
    hi_sel = float(w_form["high"].max())
    lo_sel = float(w_form["low"].min())

    w_guard = w.iloc[:GUARD_MIN] if len(w) >= GUARD_MIN else w
    if bias == 1:
        adverse = (p0 - w_guard["low"].cummin()).max() / atr0
    else:
        adverse = (w_guard["high"].cummax() - p0).max() / atr0

    first_break_ts = None
    for ts, row in w.iloc[MIN_WAIT:].iterrows():
        if bias == 1 and row["close"] > hi_sel:
            first_break_ts = ts; break
        if bias == -1 and row["close"] < lo_sel:
            first_break_ts = ts; break

    retest_depth_atr = np.nan; rebreak_ts = None
    if first_break_ts is not None:
        level = hi_sel if bias == 1 else lo_sel
        touched = False
        for ts, row in w.loc[first_break_ts:].iterrows():
            a = float(row["ATR"]); 
            if bias == 1:
                near = abs(row["low"] - level) / a
                if near <= 2.0: retest_depth_atr = near if np.isnan(retest_depth_atr) else min(retest_depth_atr, near)
                if touched and row["close"] >= level + 0.05*a:
                    rebreak_ts = ts; break
                if near <= 1.0: touched = True
            else:
                near = abs(row["high"] - level) / a
                if near <= 2.0: retest_depth_atr = near if np.isnan(retest_depth_atr) else min(retest_depth_atr, near)
                if touched and row["close"] <= level - 0.05*a:
                    rebreak_ts = ts; break
                if near <= 1.0: touched = True

    up = float((w["high"].cummax() - p0).max() / atr0)
    dn = float((p0 - w["low"].cummin()).max() / atr0)
    max_side = up if bias==1 else dn
    opp_side = dn if bias==1 else up

    records.append({
        "t_alert": r["t_alert"], "t_sel": t0, "bias": bias,
        "adverse_guard_8m_atr": float(adverse),
        "first_break_ts": first_break_ts,
        "retest_depth_atr": np.nan if np.isnan(retest_depth_atr) else float(retest_depth_atr),
        "rebreak_ts": rebreak_ts,
        "donchian_hi": hi_sel, "donchian_lo": lo_sel,
        "max_side_atr": max_side, "opp_side_atr": opp_side
    })

df = pd.DataFrame.from_records(records)
outdir = Path("results/study"); outdir.mkdir(parents=True, exist_ok=True)
(df).to_csv(outdir / "entry_path_events.csv", index=False)

summary = {
    "n": int(len(df)),
    "adverse_guard_q50": float(df["adverse_guard_8m_atr"].quantile(0.50)) if len(df) else np.nan,
    "adverse_guard_q75": float(df["adverse_guard_8m_atr"].quantile(0.75)) if len(df) else np.nan,
    "retest_depth_q50": float(df["retest_depth_atr"].quantile(0.50)) if df["retest_depth_atr"].notna().any() else np.nan,
    "retest_depth_q75": float(df["retest_depth_atr"].quantile(0.75)) if df["retest_depth_atr"].notna().any() else np.nan,
    "first_break_rate": float(df["first_break_ts"].notna().mean()) if len(df) else np.nan,
    "rebreak_rate": float(df["rebreak_ts"].notna().mean()) if len(df) else np.nan,
    "max_side_q50": float(df["max_side_atr"].quantile(0.50)) if len(df) else np.nan,
    "max_side_q75": float(df["max_side_atr"].quantile(0.75)) if len(df) else np.nan,
    "clean_run_rate_ge6": float((df["max_side_atr"] >= 6.0).mean()) if len(df) else np.nan,
    "clean_run_rate_ge10": float((df["max_side_atr"] >= 10.0).mean()) if len(df) else np.nan,
}
pd.Series(summary).to_json(outdir / "summary.json", indent=2)

grid = []
for x in [0.6,0.8,1.0,1.2]:
    for rmin,rmax in [(0.2,0.8),(0.4,1.0),(0.6,1.2)]:
        m = df.copy()
        m = m[m["adverse_guard_8m_atr"] <= x]
        m = m[(m["retest_depth_atr"].between(rmin, rmax))]
        if len(m)==0:
            grid.append({"adverse_guard":x, "retest_band": f"{rmin}-{rmax}", "n":0, "p_clean":np.nan})
        else:
            p_clean = float(((m["max_side_atr"] >= 8.0) & (m["opp_side_atr"] < 2.0)).mean())
            grid.append({"adverse_guard":x, "retest_band": f"{rmin}-{rmax}", "n": int(len(m)), "p_clean": p_clean})
pd.DataFrame(grid).to_csv(outdir / "entry_zone_grid.csv", index=False)

print(summary)
print("[OK] Wrote results/study/entry_path_events.csv & entry_zone_grid.csv & summary.json")
