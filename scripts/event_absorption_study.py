import json, glob
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repo root is importable
import sys, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_ohlcv import read_any_ohlcv
from src.atr_basic import atr
from src.engine.footprint import ticks_to_microbars, attach_minute_ATR60_to_micro
from src.engine.absorption import AbsorptionCfg, absorption_events
from src.utils import donchian_high, donchian_low

# -------------- Tick discovery + loader --------------
import os, glob as _glob
import hashlib

# Progress helper
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, total=None, desc=None):
        it = list(it)
        n = len(it) if total is None else total
        print(f"{desc or 'Progress'}: 0/{n}", end="\r")
        for i, x in enumerate(it, 1):
            if i % max(1, n // 10 or 1) == 0:
                print(f"{desc or 'Progress'}: {i}/{n}", end="\r")
            yield x
        print(f"{desc or 'Progress'}: {n}/{n}")

TICK_COL_OPTS = [
    ("ts","timestamp","time","date","open_time","open_time_ms","ts_ms"),
    ("price","p","last","close","trade_price"),
    ("qty","size","amount","volume","q","quantity","base_volume","quote_volume","vol","v"),
    ("is_buyer_maker","is_buyer_maker_flag","isBuyerMaker","is_buy_maker","buyer_maker","is_maker","maker_flag","side")
]

def _normalize_tick_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    def pick(cands):
        for c in cands:
            lc = c.lower()
            if lc in cols: return cols[lc]
        return None
    ts = pick(TICK_COL_OPTS[0])
    px = pick(TICK_COL_OPTS[1])
    q  = pick(TICK_COL_OPTS[2])
    ib = pick(TICK_COL_OPTS[3])
    if ts is None or px is None:
        raise ValueError("Tick CSV missing required columns (ts/price).")
    mapping = {ts:"ts", px:"price"}
    if q is not None:
        mapping[q] = "qty"
    out = df.rename(columns=mapping)
    # derive qty if missing
    if "qty" not in out.columns:
        # try quote amount / price
        q_quote = None
        for c in ("quote_qty","quote_volume","amount_q","notional"):
            if c in df.columns:
                q_quote = c; break
        if q_quote is not None:
            out["qty"] = pd.to_numeric(df[q_quote], errors="coerce") / pd.to_numeric(out["price"], errors="coerce")
        else:
            out["qty"] = 1.0
    # normalize side/is_buyer_maker
    if ib is not None:
        if ib.lower() == "side" and ib in df.columns:
            side = df[ib].astype(str).str.upper()
            out["is_buyer_maker"] = (side == "SELL").astype(int)
        else:
            out = out.rename(columns={ib:"is_buyer_maker"})
    # ts to UTC with numeric unit detection if needed
    if not np.issubdtype(out["ts"].dtype, np.datetime64):
        ser = pd.to_numeric(out["ts"], errors="coerce")
        unit = "ms" if float(ser.dropna().median()) > 1e11 else "s"
        out["ts"] = pd.to_datetime(ser, utc=True, errors="coerce", unit=unit)
    return out[["ts","price","qty"] + (["is_buyer_maker"] if "is_buyer_maker" in out.columns else [])]

def find_tick_paths(cfg: dict, cli_ticks_glob: str | None) -> list:
    if cli_ticks_glob:
        paths = sorted(_glob.glob(cli_ticks_glob))
        if paths:
            return paths
    ticks_glob = (cfg.get("files", {}) or {}).get("ticks_glob")
    if ticks_glob:
        paths = sorted(_glob.glob(ticks_glob))
        if paths:
            return paths
    # Fallback auto-detect near ohlcv_glob root
    ohlcv_glob = (cfg.get("files", {}) or {}).get("ohlcv_glob", "")
    ohlcv_paths = _glob.glob(ohlcv_glob)
    root = os.path.commonpath(ohlcv_paths) if ohlcv_paths else "data"
    guess = sorted(_glob.glob(os.path.join(root, "**", "*ticks*.csv"), recursive=True))
    return guess

def read_ticks(paths: list, chunksize: int = 1_000_000) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["ts","price","qty","is_buyer_maker"]) 
    parts = []
    used, skipped = 0, 0
    for p in tqdm(paths, desc="Reading ticks", total=len(paths)):
        try:
            df = pd.read_csv(p, low_memory=False)
            df = _normalize_tick_cols(df)
            # enforce lean dtypes
            df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce").astype("float32")
            if "is_buyer_maker" in df.columns:
                df["is_buyer_maker"] = df["is_buyer_maker"].astype("int8")
            parts.append(df)
            used += 1
        except Exception as e:
            print(f"[Warn] skip non-tick file: {p} ({e})")
            skipped += 1
    t = pd.concat(parts, ignore_index=True).sort_values("ts")
    if "is_buyer_maker" not in t.columns:
        sgn = np.sign(t["price"].diff().fillna(0.0))
        t["is_buyer_maker"] = (sgn <= 0).astype(int)
    if skipped:
        print(f"[Load] used {used} tick files, skipped {skipped}")
    return t


def load_config(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _expand(path_or_glob: str) -> list:
    if not path_or_glob:
        return []
    if any(ch in path_or_glob for ch in "*?[]"):
        return glob.glob(path_or_glob)
    return [path_or_glob]


def main(cfg_path: str = "configs/event_study.yaml", *, ohlcv_glob: str | None = None, ticks_glob: str | None = None):
    cfg = load_config(cfg_path) if Path(cfg_path).exists() else {}
    study = cfg.get("study", {})
    horizon_min = int(study.get("horizon_min", 180))
    k_list = study.get("k_list", [2.0,3.0,4.0])
    m_list = study.get("m_list", [1.0,1.5])
    don_win = int(study.get("donchian_window", 60))

    files = cfg.get("files", {})
    src_glob = ohlcv_glob or files.get("ohlcv_glob") or files.get("ohlcv") or "data/ticks/BTCUSDT/BTCUSDT-1m-2025-*.csv"
    alerts_csv = files.get("alerts_csv", "results/alerts_used.csv")

    # Minute OHLCV and ATR60
    paths = _expand(src_glob)
    if not paths:
        print(f"[Error] No OHLCV files found for glob: {src_glob}")
        print("Hint: set files.ohlcv_glob in configs/event_study.yaml or pass --ohlcv_glob")
        return
    parts = []
    print(f"[Load] OHLCV: {len(paths)} files matched; reading...")
    for p in tqdm(sorted(set(paths)), desc="Reading OHLCV", total=len(set(paths))):
        try:
            parts.append(read_any_ohlcv(p))
        except Exception as e:
            print(f"[Warn] failed to read OHLCV: {p} ({e})")
    bars = pd.concat(parts, ignore_index=True).sort_values("ts").drop_duplicates(subset=["ts"], keep="last").set_index("ts")
    bars["ATR"] = atr(bars[["open","high","low","close"]], window=60)
    print(f"[Load] OHLCV minutes: {len(bars):,}")

    # Alerts and gate windows (for gating tick load)
    if not Path(alerts_csv).exists():
        print(f"[Error] alerts CSV not found at: {alerts_csv}")
        print("Hint: run your bracket to materialize alerts_used.csv or point files.alerts_csv to a CSV with column t_alert")
        return
    alerts = pd.read_csv(alerts_csv)
    if "t_alert" not in alerts.columns:
        print("[Error] alerts CSV must contain column t_alert")
        return
    alerts["t_alert"] = pd.to_datetime(alerts["t_alert"], utc=True)

    # Gate-first tick processing
    def _coalesce(intervals):
        intervals = sorted(intervals)
        out = []
        for s, e in intervals:
            if not out or s > out[-1][1]:
                out.append([s, e])
            else:
                out[-1][1] = max(out[-1][1], e)
        return [(a, b) for a, b in out]

    def build_gate_windows(alerts_df, horizon_min: int, pad_s: int = 15):
        pad = pd.Timedelta(seconds=pad_s)
        raw = [(t, t + pd.Timedelta(minutes=horizon_min)) for t in alerts_df["t_alert"]]
        raw = [(s - pad, e + pad) for s, e in raw]
        return _coalesce(raw)

    def mask_ticks_to_windows(ticks_df: pd.DataFrame, windows):
        if ticks_df.empty:
            return pd.Series(False, index=ticks_df.index)
        m = pd.Series(False, index=ticks_df.index)
        for s, e in windows:
            m |= (ticks_df["ts"] >= s) & (ticks_df["ts"] <= e)
        return m

    windows = build_gate_windows(alerts, horizon_min=int(horizon_min), pad_s=int(study.get("gate_pad_s", 15)))

    # Prefer parquet micro/minute inputs if present to skip tick building
    import os
    min_pq = (cfg.get("files", {}) or {}).get("minutes_parquet")
    mic_pq = (cfg.get("files", {}) or {}).get("micro_parquet")
    micro_loaded_from_parquet = False
    if mic_pq:
        assert os.path.exists(mic_pq), f"Missing micro parquet: {mic_pq}"
        micro = pd.read_parquet(mic_pq)
        if not isinstance(micro.index, pd.DatetimeIndex):
            micro.index = pd.to_datetime(micro.index, utc=True)
        print(f"[Load] microbars parquet: {len(micro):,} rows @ {mic_pq}")

        # If minutes parquet provided, load (for Donchian or any minute analytics)
        if min_pq:
            assert os.path.exists(min_pq), f"Missing minutes parquet: {min_pq}"
            bars = pd.read_parquet(min_pq)
            if not isinstance(bars.index, pd.DatetimeIndex):
                bars.index = pd.to_datetime(bars.index, utc=True)
            print(f"[Load] minutes parquet: {len(bars):,} rows @ {min_pq}")

        # Assert ATR60 exists on micro; if not, attach from minutes ATR
        if ("ATR60" not in micro.columns) or micro["ATR60"].isna().all():
            if (bars is None) or ("ATR" not in bars.columns):
                raise RuntimeError("ATR60 missing on micro and minutes ATR not available to attach.")
            micro = attach_minute_ATR60_to_micro(micro, bars, bars["ATR"])
            print("[Fix] Attached ATR60 to micro from minutes ATR.")

        micro_loaded_from_parquet = True

    # Ticks → microbars via discovery (gated)
    if not micro_loaded_from_parquet:
        tick_paths = find_tick_paths(cfg, ticks_glob)
        if not tick_paths:
            print("[Error] No tick CSVs found (files.ticks_glob unset and autodetect empty).")
            print("Set files.ticks_glob to e.g. data/ticks/BTCUSDT/BTCUSDT-ticks-2025-*.csv")
            return
        ticks = read_ticks(tick_paths)
        ticks = ticks.sort_values("ts")
        print(f"[Load] ticks: {len(ticks):,} rows from {len(tick_paths)} files")
        gate_mask = mask_ticks_to_windows(ticks, windows)
        ticks_g = ticks.loc[gate_mask].copy()
        print(f"[Gate] ticks inside windows: {len(ticks_g):,} / {len(ticks):,} ({100*len(ticks_g)/max(1,len(ticks)):.1f}%)")
        micro_tf = str(cfg.get("absorption", {}).get("micro_tf", "1s"))

        # Caching microbars per gate set (hash) + tf
        cache_cfg = cfg.get("cache", {})
        cache_dir = Path(cache_cfg.get("microbars_dir", "results/cache/microbars"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        def _gate_hash(win):
            items = [(int(s.value//10**9), int(e.value//10**9)) for s, e in win]
            h = hashlib.md5(str((micro_tf, items)).encode()).hexdigest()[:12]
            return h
        key = _gate_hash(windows)
        cache_path = cache_dir / f"micro_{micro_tf}_{key}.parquet"
        micro = None
        if bool(cache_cfg.get("enable", True)) and cache_path.exists() and not bool(cache_cfg.get("force_rebuild", False)):
            try:
                micro = pd.read_parquet(cache_path)
                micro.index = pd.to_datetime(micro.index, utc=True)
                print(f"[Cache] loaded microbars: {cache_path}")
            except Exception as e:
                print(f"[Cache] failed to load ({e}); rebuilding...")
                micro = None
        if micro is None:
            micro = ticks_to_microbars(ticks_g, micro_tf=micro_tf)
            try:
                micro.to_parquet(cache_path, index=True)
                print(f"[Cache] wrote microbars: {cache_path}")
            except Exception as e:
                print(f"[Cache] write failed: {e}")
        micro = attach_minute_ATR60_to_micro(micro, bars, bars["ATR"])  # ATR60 proxy

    # Donchian on minute; forward-fill to micro
    d_hi = donchian_high(bars["close"], window=don_win).reindex(micro.index, method="ffill")
    d_lo = donchian_low(bars["close"], window=don_win).reindex(micro.index, method="ffill")

    # Absorption events
    ac = cfg.get("absorption", {})
    # Calibration readout to help pick thresholds
    def calibrate_print(micro_df: pd.DataFrame, lookback: int, alpha_val: float):
        try:
            from src.engine.footprint import rolling_z as _rz
            z_abs = _rz(micro_df["delta"].abs(), lookback)
            z_tr  = _rz((micro_df["tr"] / micro_df["ATR60"]).replace([np.inf,-np.inf], np.nan), lookback)
            scr   = z_abs - alpha_val * z_tr
            def q(s):
                s = pd.to_numeric(s, errors="coerce").dropna()
                return s.quantile([0.5,0.8,0.9,0.95,0.99]).round(2).to_dict() if len(s) else {}
            print("[Calib] z_abs quantiles:", q(z_abs))
            print("[Calib] z_tratr quantiles:", q(z_tr))
            print("[Calib] score quantiles:", q(scr))
            print("[Calib] max score:", round(float(pd.to_numeric(scr, errors='coerce').max() or 0.0), 2))
        except Exception as e:
            print(f"[Warn] calibration print failed: {e}")

    calibrate_print(micro, lookback=int(ac.get("lookback_s", 60)), alpha_val=float(ac.get("alpha", 1.0)))
    a_cfg = AbsorptionCfg(
        lookback_s=int(ac.get("lookback_s", 60)),
        z_delta_thr=float(ac.get("z_delta_thr", 2.5)),
        wick_bias_min=float(ac.get("wick_bias_min", 0.65)),
        donchian_prox_atr=ac.get("donchian_prox_atr", 0.35),
        lambda_drop=bool(ac.get("lambda_drop", True)),
        alpha=float(ac.get("alpha", 1.0)),
    )
    # Only compute lambda (expensive) if toggled on; absorption_events honors cfg.lambda_drop
    ev = absorption_events(micro, a_cfg, d_hi, d_lo)
    print(f"[Absorption] raw events: {len(ev)}")
    if ev.empty:
        print("[Info] No absorption events fired at current thresholds. Try lowering absorption.z_delta_thr or wick_bias_min, or disable lambda_drop.")
        return
    ev = ev.rename(columns={"t_abs":"ts"})
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)

    # Gates
    # alerts already loaded above

    # Filter events inside gates
    def in_gates(ts):
        return ((alerts["t_alert"] <= ts) & (alerts["t_alert"] + pd.Timedelta(minutes=horizon_min) >= ts)).any()
    ev_g = ev[ev["ts"].map(in_gates)].copy()
    print(f"[Absorption] events inside gates: {len(ev_g)} / {len(ev)}")
    if ev_g.empty:
        print("[Info] No events inside alert gates. Verify files.alerts_csv and horizon, or relax donchian_prox_atr/lambda_drop.")
        return
    # Direction counts
    dir_counts = ev_g["direction"].value_counts().to_dict()
    print(f"[Direction] counts: {dir_counts}")

    # Barrier outcomes (minute-based)
    results = []
    print(f"[Eval] events in gates: {len(ev_g)}; evaluating barriers...")
    for _, r in tqdm(ev_g.iterrows(), total=len(ev_g), desc="Evaluating"):
        start = bars.index[bars.index.searchsorted(r["ts"])]
        if start not in bars.index:
            continue
        dirn = int(r["direction"])  # +1 or -1
        atr0 = float(bars.loc[start, "ATR"]) if np.isfinite(bars.loc[start, "ATR"]) else np.nan
        if not np.isfinite(atr0) or atr0 <= 0:
            continue
        w = bars.loc[start : start + pd.Timedelta(minutes=horizon_min)]
        hi = w["high"].to_numpy()
        lo = w["low"].to_numpy()
        ts_arr = w.index.to_numpy()
        px0 = float(bars.loc[start, "close"])
        for K in k_list:
            for M in m_list:
                tp_level = px0 + dirn * float(K) * atr0
                sl_level = px0 - dirn * float(M) * atr0
                if dirn == +1:
                    tp_mask = (hi >= tp_level)
                    sl_mask = (lo <= sl_level)
                else:
                    tp_mask = (lo <= tp_level)
                    sl_mask = (hi >= sl_level)
                tp_any = bool(tp_mask.any())
                sl_any = bool(sl_mask.any())
                hit_tp_idx = int(np.argmax(tp_mask)) if tp_any else -1
                hit_sl_idx = int(np.argmax(sl_mask)) if sl_any else -1
                t_hit_tp = ts_arr[hit_tp_idx] if tp_any else None
                t_hit_sl = ts_arr[hit_sl_idx] if sl_any else None
                win = bool(tp_any and (not sl_any or (hit_tp_idx >= 0 and hit_tp_idx < hit_sl_idx)))
                tth = (t_hit_tp - start).astype("timedelta64[m]")/np.timedelta64(1, 'm') if tp_any else np.nan
                results.append({"ts": r["ts"], "dir": dirn, "K": float(K), "M": float(M), "win": win, "tth_min": float(tth) if tp_any else np.nan})

    res = pd.DataFrame(results)
    outdir = Path("results/study"); outdir.mkdir(parents=True, exist_ok=True)
    ev_g.to_csv(outdir / "absorption_events_in_gates.csv", index=False)
    res.to_csv(outdir / "absorption_barrier_outcomes.csv", index=False)
    summary = res.groupby(["K","M"]).agg(n=("win","size"), p_win=("win","mean"), med_tth=("tth_min","median")).reset_index()
    summary.to_csv(outdir / "absorption_summary.csv", index=False)
    print(summary.to_string(index=False))
    # PASS line
    pass_mask = ((summary["K"]==3.0) & (summary["M"].isin([1.0,1.5])) & (summary["p_win"]>=0.40))
    print("PASS" if pass_mask.any() else "NO-PASS")

    # Extra slices
    try:
        # Merge results onto events to get score, lambda, donch proximity
        ev_g_idx = ev_g.set_index("ts")
        res_idx = res.set_index("ts")
        merged = res_idx.join(ev_g_idx[["score","near_donch","lambda_below_med"]], how="left")
        # Score bins
        merged["score_bin"] = pd.qcut(merged["score"], q=4, duplicates="drop")
        by_score = merged.groupby(["K","M","score_bin"])['win'].mean().reset_index().rename(columns={'win':'p_win'})
        by_score.to_csv(outdir / "absorption_by_score.csv", index=False)
        # Donchian proximity
        by_donch = merged.groupby(["K","M","near_donch"])['win'].mean().reset_index().rename(columns={'win':'p_win'})
        by_donch.to_csv(outdir / "absorption_by_donch.csv", index=False)
        # Lambda drop
        by_lam = merged.groupby(["K","M","lambda_below_med"])['win'].mean().reset_index().rename(columns={'win':'p_win'})
        by_lam.to_csv(outdir / "absorption_by_lambda.csv", index=False)
    except Exception as e:
        print(f"[Warn] extra slices failed: {e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/event_study.yaml")
    ap.add_argument("--ticks_glob", type=str, default=None)
    ap.add_argument("--ohlcv_glob", type=str, default=None)
    args = ap.parse_args()
    main(args.config, ohlcv_glob=args.ohlcv_glob, ticks_glob=args.ticks_glob)
