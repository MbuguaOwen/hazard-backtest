import json, pathlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.io_ohlcv import read_any_ohlcv
from src.atr_basic import atr
from src.engine.direction import DirCfg, DriftCfg, CvdCfg, select_direction
from src.engine.footprint import ticks_to_microbars, attach_minute_ATR60_to_micro
from src.engine.absorption import AbsorptionCfg, absorption_events
from src.utils import donchian_high, donchian_low

# --- micro flow helpers (bar-proxy CVD) ---
def _bar_proxy_signed_volume(bars: pd.DataFrame) -> pd.Series:
    ret = bars["close"].diff()
    sgn = np.sign(ret).fillna(0.0)
    sv = (bars["volume"].fillna(0.0) * sgn).rename("sv")
    return sv

def micro_cvd_slope(bars: pd.DataFrame, t: pd.Timestamp, lookback: int = 5) -> float:
    if t not in bars.index:
        t = bars.index[bars.index.searchsorted(t)] if len(bars) else t
    end_idx = bars.index.get_loc(t)
    start_idx = max(0, end_idx - (lookback - 1))
    idx = bars.index[start_idx:end_idx+1]
    if len(idx) == 0:
        return 0.0
    sv = _bar_proxy_signed_volume(bars.loc[idx])
    return float(sv.sum())

def micro_cvd_delta_z(bars: pd.DataFrame, t: pd.Timestamp, lookback: int = 30, window:int = 5) -> float:
    if t not in bars.index:
        t = bars.index[bars.index.searchsorted(t)] if len(bars) else t
    # use rolling sum over small window, z-scored over last `lookback` bars
    upto = bars.loc[:t]
    if len(upto) < max(10, lookback//2):
        return 0.0
    sv = _bar_proxy_signed_volume(upto)
    roll = sv.rolling(window=window, min_periods=max(2, window//2)).sum()
    series = roll.dropna()
    if len(series) < lookback + 2:
        base = series
    else:
        base = series.iloc[-(lookback+1):]
    if len(base) < 3:
        return 0.0
    cur = float(base.iloc[-1])
    prev = float(base.iloc[-2])
    mu = float(base.iloc[:-1].mean()) if len(base) > 2 else float(base.mean())
    sd = float(base.iloc[:-1].std(ddof=1)) if len(base) > 3 else float(base.std(ddof=1) or 0.0)
    if sd == 0.0:
        return 0.0
    z_cur = (cur - mu) / sd
    z_prev = (prev - mu) / sd
    return float(z_cur - z_prev)

try:
    from tqdm.auto import tqdm  # progress bar
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


def load_config(path: str) -> dict:
    import yaml
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def _apply_costs(px: float, direction: int, fee_bps: float, slip_bps: float) -> Tuple[float, float]:
    slip = px * (slip_bps * 1e-4)
    px_adj = px + (-direction) * slip
    fee = px * (fee_bps * 1e-4)
    return px_adj, fee


def enter_long_retest_second_break(
    bars: pd.DataFrame,
    t0: pd.Timestamp,
    U: float,
    eps_atr: float,
    retest_atr: float,
    max_wait_min: int,
    atr_series: pd.Series,
    min_wait_min: int = 3,
) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[float], Dict]:
    """Require first close >= U + eps*ATR, then a retest within +/- retest_atr*ATR around U, then a second close >= U + eps*ATR.
    Returns (t_entry, entry_px, retest_depth_atr, diag) where diag contains first/retest/rebreak timestamps.
    """
    t_start = t0 + pd.Timedelta(minutes=min_wait_min)
    t_end = t0 + pd.Timedelta(minutes=max_wait_min)
    w = bars.loc[t_start:t_end]
    first_ts = None
    for ts, r in w.iterrows():
        a = float(atr_series.loc[ts])
        if r["close"] >= U + eps_atr * a:
            first_ts = ts
            break
    if first_ts is None:
        return None, None, None, {"first_break_ts": None, "retest_ts": None, "rebreak_ts": None}
    touched = False
    depth = None
    retest_ts = None
    retest_low = None
    for ts, r in bars.loc[first_ts:t_end].iterrows():
        a = float(atr_series.loc[ts])
        near_hi = U + retest_atr * a
        near_lo = U - retest_atr * a
        if r["low"] <= near_hi and r["high"] >= near_lo:
            touched = True
            depth = max(0.0, (U - float(r["low"])) / a)
            if retest_ts is None:
                retest_ts = ts
            retest_low = float(r["low"]) if retest_low is None else min(retest_low, float(r["low"]))
        if touched and r["close"] >= U + eps_atr * a:
            return ts, float(r["close"]), depth, {"first_break_ts": first_ts, "retest_ts": retest_ts, "rebreak_ts": ts, "retest_low": retest_low}
    return None, None, None, {"first_break_ts": first_ts, "retest_ts": retest_ts, "rebreak_ts": None}

def enter_short_retest_second_break(
    bars: pd.DataFrame,
    t0: pd.Timestamp,
    L: float,
    eps_atr: float,
    retest_atr: float,
    max_wait_min: int,
    atr_series: pd.Series,
    min_wait_min: int = 3,
) -> Tuple[Optional[pd.Timestamp], Optional[float], Optional[float], Dict]:
    t_start = t0 + pd.Timedelta(minutes=min_wait_min)
    t_end = t0 + pd.Timedelta(minutes=max_wait_min)
    w = bars.loc[t_start:t_end]
    first_ts = None
    for ts, r in w.iterrows():
        a = float(atr_series.loc[ts])
        if r["close"] <= L - eps_atr * a:
            first_ts = ts
            break
    if first_ts is None:
        return None, None, None, {"first_break_ts": None, "retest_ts": None, "rebreak_ts": None}
    touched = False
    depth = None
    retest_ts = None
    retest_high = None
    for ts, r in bars.loc[first_ts:t_end].iterrows():
        a = float(atr_series.loc[ts])
        near_hi = L + retest_atr * a
        near_lo = L - retest_atr * a
        if r["low"] <= near_hi and r["high"] >= near_lo:
            touched = True
            depth = max(0.0, (float(r["high"]) - L) / a)
            if retest_ts is None:
                retest_ts = ts
            retest_high = float(r["high"]) if retest_high is None else max(retest_high, float(r["high"]))
        if touched and r["close"] <= L - eps_atr * a:
            return ts, float(r["close"]), depth, {"first_break_ts": first_ts, "retest_ts": retest_ts, "rebreak_ts": ts, "retest_high": retest_high}
    return None, None, None, {"first_break_ts": first_ts, "retest_ts": retest_ts, "rebreak_ts": None}


def trade_metrics(trades: pd.DataFrame) -> Dict:
    if len(trades) == 0:
        return dict(n_trades=0, sum_pnl=0.0, mean_pnl=0.0, p_win=0.0, max_dd=0.0, sharpe=0.0, mean_r=0.0, sharpe_r=0.0)
    pnl = trades["pnl"].fillna(0.0).values
    eq = pnl.cumsum()
    max_dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(365 * 24 * 60)) if np.std(pnl) > 1e-12 else 0.0
    out = dict(
        n_trades=int(len(trades)),
        sum_pnl=float(np.sum(pnl)),
        mean_pnl=float(np.mean(pnl)),
        p_win=float((pnl > 0).mean()),
        max_dd=max_dd,
        sharpe=sharpe,
    )
    if "pnl_r" in trades.columns:
        r = trades["pnl_r"].fillna(0.0).values
        sharpe_r = float(np.mean(r) / np.std(r) * np.sqrt(365 * 24 * 60)) if np.std(r) > 1e-12 else 0.0
        out.update(dict(mean_r=float(np.mean(r)), sharpe_r=sharpe_r))
    return out


def run_from_config(cfg_path: str):
    cfg = load_config(cfg_path)
    files = cfg.get("files", {})
    strat = cfg.get("strategy", {})
    out = cfg.get("out", {"dir": "results"})

    # Load bars and ATR
    bars = read_any_ohlcv(files["ohlcv"]).sort_values("ts").set_index("ts")
    bars["ATR"] = atr(bars[["open", "high", "low", "close"]], window=int(strat.get("atr_window", 60)))

    # Alerts
    alerts = pd.read_csv(files["alerts_csv"]) if files.get("alerts_csv") else pd.DataFrame(columns=["t_alert"])
    if "t_alert" not in alerts.columns:
        raise ValueError("files.alerts_csv must contain 't_alert'")
    alerts = pd.to_datetime(alerts["t_alert"], utc=True, errors="coerce").dt.floor("min")

    # Direction config
    d_cfg = strat.get("direction", {})
    dir_cfg = DirCfg(
        mode=d_cfg.get("mode", "none"),
        drift=DriftCfg(
            M_min=int(d_cfg.get("drift", {}).get("M_min", 3)),
            eps_atr=float(d_cfg.get("drift", {}).get("eps_atr", 0.10)),
        ),
        cvd=CvdCfg(
            source=d_cfg.get("cvd", {}).get("source", "bar_proxy"),
            baseline_min=int(d_cfg.get("cvd", {}).get("baseline_min", 240)),
            z_thr=float(d_cfg.get("cvd", {}).get("z_thr", 0.8)),
        ),
        anchor_price=d_cfg.get("anchor_price", "select"),
    )

    print("[Cfg] direction.mode =", dir_cfg.mode)
    k_breakout = float(strat["k_breakout"])
    # allow sl_mult to be null (use structure stop)
    _sl_raw = strat.get("sl_mult", None)
    sl_mult = float(_sl_raw) if _sl_raw is not None else None
    tp_mult = float(strat["tp_mult"])
    be_frac = float(strat.get("be_frac", 0.60))
    trail_mult = strat.get("trail_mult")
    horizon_min = int(strat.get("horizon_min", 180))
    entry_recalc_atr = bool(strat.get("entry_recalc_atr", True))
    fee_bps = float(strat.get("fee_bps", 1.0))
    slip_bps = float(strat.get("slip_bps", 0.5))

    ec = strat.get("entry_confirm", {})
    assert ec.get("mode", "retest_second_break") in ("retest_second_break","absorption_reversal"), "Wrong entry mode active."
    assert dir_cfg.mode in ("hybrid_and", "hybrid_adaptive", "drift", "cvd", "absorption"), "Direction selector not active."
    epsilon_atr = float(ec.get("epsilon_atr", 0.05))
    retest_atr = float(ec.get("retest_atr", 0.10))
    # impulse and flow filters (optional)
    imp = ec.get("impulse", {}) if isinstance(ec.get("impulse", {}), dict) else {}
    min_body_atr = imp.get("min_body_atr", None)
    min_tr_atr = imp.get("min_true_range_atr", None)
    flow = ec.get("flow", {}) if isinstance(ec.get("flow", {}), dict) else {}
    cvd_slope_min = flow.get("cvd_slope_min", None)
    cvd_delta_z_min = flow.get("cvd_delta_z_min", None)
    min_wait_min = int(ec.get("min_wait_min", 3))
    range_min = int(ec.get("range_min", min_wait_min))
    adverse_guard_atr = float(ec.get("adverse_guard_atr", 0.8))
    guard_window_min = int(ec.get("guard_window_min", 8))
    max_wait_min = int(ec.get("max_wait_min", 60))

    # Optional absorption precompute
    use_absorption = (ec.get("mode") == "absorption_reversal") or (d_cfg.get("mode") == "absorption")
    micro = None
    events = None
    if use_absorption:
        import glob
        src_path = files.get("ohlcv")
        paths = []
        if isinstance(src_path, str) and any(ch in src_path for ch in "*?[]"):
            paths = glob.glob(src_path)
        elif isinstance(src_path, str):
            paths = [src_path]
        ticks = []
        for p in sorted(paths):
            try:
                df = pd.read_csv(p)
                lowcols = {c: c.strip().lower() for c in df.columns}
                df = df.rename(columns=lowcols)
                if {"ts","price"}.issubset(set(df.columns)):
                    if "qty" not in df.columns:
                        for qc in ("quantity","size","amount","volume"):
                            if qc in df.columns:
                                df = df.rename(columns={qc:"qty"}); break
                    if "qty" not in df.columns:
                        df["qty"] = 1.0
                    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                    ticks.append(df[["ts","price","qty"] + (["is_buyer_maker"] if "is_buyer_maker" in df.columns else [])])
            except Exception:
                continue
        if len(ticks):
            t = pd.concat(ticks, ignore_index=True).dropna(subset=["ts","price"]).sort_values("ts")
            micro = ticks_to_microbars(t, micro_tf=str(strat.get("absorption",{}).get("micro_tf","1s")))
        else:
            tmp = bars[["open","high","low","close","volume"]].copy()
            tmp = tmp.rename(columns={"volume":"buy_vol"})
            tmp["sell_vol"] = 0.0
            tmp["delta"] = tmp["buy_vol"] - tmp["sell_vol"]
            tmp["tr"] = (tmp["high"] - tmp["low"]).fillna(0.0)
            micro = tmp.copy()
        micro = attach_minute_ATR60_to_micro(micro, bars, bars["ATR"])  # ATR as ATR60 proxy
        don_win = int(strat.get("horizon_min", 180))
        d_hi = donchian_high(bars["close"], window=don_win).reindex(micro.index, method="ffill")
        d_lo = donchian_low(bars["close"], window=don_win).reindex(micro.index, method="ffill")
        ac = strat.get("absorption", {})
        a_cfg = AbsorptionCfg(
            lookback_s=int(ac.get("lookback_s", 60)),
            z_delta_thr=float(ac.get("z_delta_thr", 2.5)),
            tr_max_atr=float(ac.get("tr_max_atr", 0.30)),
            wick_bias_min=float(ac.get("wick_bias_min", 0.65)),
            donchian_prox_atr=ac.get("donchian_prox_atr", 0.35),
            lambda_drop=bool(ac.get("lambda_drop", True)),
            alpha=float(ac.get("alpha", 1.0)),
        )
        ev = absorption_events(micro, a_cfg, d_hi, d_lo)
        events = ev.set_index("t_abs").sort_index()
        print(f"[Absorption] events: {len(events)}")

    # Intersect alerts with available bars and ensure full horizon coverage
    bars_idx = bars.index
    t0, t1 = bars_idx.min(), bars_idx.max()
    h_exit = pd.Timedelta(minutes=horizon_min)
    alerts = alerts[(alerts >= t0) & (alerts <= (t1 - h_exit))]
    alerts = pd.DatetimeIndex(alerts.sort_values().unique())

    # Save alerts used
    pathlib.Path(out["dir"]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_alert": alerts}).to_csv(f"{out['dir']}/alerts_used.csv", index=False)

    # Simulation
    trades = []
    trade_log = []
    n_adverse_skips = 0
    n_postsel_breakouts = 0
    for t_gate in tqdm(alerts, desc="Simulating alerts"):
        if t_gate not in bars.index:
            idx = bars.index.searchsorted(t_gate)
            if idx >= len(bars):
                continue
            t_gate = bars.index[idx]

        row_gate = bars.loc[t_gate]
        atr_ref = float(row_gate["ATR"]) if pd.notna(row_gate["ATR"]) else np.nan
        if not np.isfinite(atr_ref) or atr_ref <= 0:
            continue

        # Absorption direction: find event in gate window if enabled
        abs_event = None
        if use_absorption and d_cfg.get("mode") == "absorption" and events is not None:
            # window: from selection time to selection+max_wait
            t_sel_probe = t_gate + pd.Timedelta(minutes=int(dir_cfg.drift.M_min))
            w_start = t_sel_probe
            w_end = t_sel_probe + pd.Timedelta(minutes=int(ec.get("max_wait_min", 45)))
            evw = events.loc[(events.index >= w_start) & (events.index <= w_end)]
            if len(evw):
                first = evw.iloc[0]
                abs_event = {"t_abs": first.name, "direction": int(first["direction"]),
                             "abs_low": float(first["abs_low"]), "abs_high": float(first["abs_high"]) }
        bias, dinfo = select_direction(bars, t_gate, atr_ref, dir_cfg, absorption_event=abs_event)
        if dir_cfg.mode != "none" and bias is None:
            trade_log.append({
                "t_alert": t_gate, "skip": "direction",
                **{k: dinfo.get(k) for k in ["t_sel", "drift", "drift_dir", "drift_sig", "cvd_z", "cvd_dir", "cvd_sig", "reason"]}
            })
            continue

        horizon_ts = t_gate + pd.Timedelta(minutes=horizon_min)
        t_sel = pd.to_datetime(dinfo.get("t_sel")) if dinfo and dinfo.get("t_sel") else t_gate
        idx_sel = bars.index[bars.index.searchsorted(t_sel)]
        p0 = float(bars.loc[idx_sel]["close"]) if dir_cfg.anchor_price == "select" else float(row_gate["close"])
        atr_anchor = float(bars.loc[bars.index[bars.index.searchsorted(t_sel)]]["ATR"])
        atr_levels = atr_anchor if entry_recalc_atr else float(row_gate["ATR"])  
        U = p0 + k_breakout * atr_levels
        L = p0 - k_breakout * atr_levels

        # Adverse guard over max wait window
        guard_end = t_sel + pd.Timedelta(minutes=guard_window_min)
        w_guard = bars.loc[t_sel:guard_end]
        if bias == +1:
            adverse = (p0 - float(w_guard["low"].cummin().min())) / atr_anchor if len(w_guard) else 0.0
        else:
            adverse = (float(w_guard["high"].cummax().max()) - p0) / atr_anchor if len(w_guard) else 0.0
        if adverse >= adverse_guard_atr:
            trade_log.append({"t_alert": t_gate, "skip": "adverse_guard", "adverse_atr": float(adverse), "t_sel": t_sel})
            n_adverse_skips += 1
            continue

        if ec.get("mode") == "absorption_reversal":
            if events is None:
                trade_log.append({"t_alert": t_gate, "skip": "no_absorption_event", "t_sel": t_sel})
                continue
            w_start = t_sel
            w_end = t_sel + pd.Timedelta(minutes=int(ec.get("max_wait_min", 60)))
            evw = events.loc[(events.index >= w_start) & (events.index <= w_end)]
            if not len(evw):
                trade_log.append({"t_alert": t_gate, "skip": "no_absorption_event", "t_sel": t_sel})
                continue
            ev_first = evw.iloc[0]
            bias = int(ev_first["direction"])  # direction opposite aggressor
            anchor = float(ev_first["abs_low"]) if bias == +1 else float(ev_first["abs_high"])
            mseg = micro.loc[ev_first.name: w_end]
            atr_micro = mseg["ATR60"]
            ret_ts = None
            for ts, r in mseg.iterrows():
                a = float(atr_micro.loc[ts]) if ts in atr_micro.index else float("nan")
                if not np.isfinite(a) or a <= 0: continue
                dist = (anchor - float(r["low"])) / a if bias==+1 else (float(r["high"]) - anchor) / a
                if dist >= 0 and dist <= retest_atr:
                    ret_ts = ts
                    break
            if ret_ts is None:
                trade_log.append({"t_alert": t_gate, "skip": "no_retest", "t_sel": t_sel, "abs_ts": str(ev_first.name)})
                continue
            t_rebreak = None
            body_atr = tr_atr = breach = None
            cvd_slope_val = cvd_dz_val = None
            for ts, r in mseg.loc[ret_ts:].iterrows():
                a = float(atr_micro.loc[ts])
                if a <= 0: continue
                close = float(r["close"])
                body_atr = abs(close - float(r["open"])) / a
                tr_atr = (float(r["high"]) - float(r["low"])) / a
                breach = bias * (close - anchor) / a
                if min_body_atr is not None and body_atr < float(min_body_atr):
                    continue
                if min_tr_atr is not None and tr_atr < float(min_tr_atr):
                    continue
                if cvd_slope_min is not None:
                    window = mseg.loc[:ts].tail(5)
                    cvd_slope_val = float(window["delta"].sum())
                    if bias * cvd_slope_val < float(cvd_slope_min):
                        continue
                if cvd_delta_z_min is not None:
                    look = mseg.loc[:ts].tail(30)
                    z = (look["delta"] - look["delta"].rolling(30, min_periods=10).mean()) / look["delta"].rolling(30, min_periods=10).std(ddof=1)
                    cvd_dz_val = float((z.iloc[-1] - z.iloc[-2])) if len(z.dropna())>=2 else 0.0
                    if bias * cvd_dz_val < float(cvd_delta_z_min):
                        continue
                if breach >= float(epsilon_atr):
                    t_rebreak = ts
                    break
            if t_rebreak is None:
                trade_log.append({"t_alert": t_gate, "skip": "no_rebreak", "t_sel": t_sel, "abs_ts": str(ev_first.name), "retest_ts": str(ret_ts)})
                continue
            t_entry = bars.index[bars.index.searchsorted(pd.to_datetime(t_rebreak))]
            entry = float(bars.loc[t_entry, "close"]) if t_entry in bars.index else float(mseg.loc[t_rebreak, "close"]) 
            ret_depth = None
            diag = {"first_break_ts": ev_first.name, "retest_ts": ret_ts, "rebreak_ts": t_rebreak}
        else:
            # Legacy post-selection Donchian breakout
            rng_end = t_sel + pd.Timedelta(minutes=range_min)
            rng = bars.loc[t_sel:rng_end]
            high_sel = float(rng["high"].max()) if len(rng) else p0
            low_sel  = float(rng["low"].min()) if len(rng) else p0
            U2, L2 = high_sel, low_sel
            if bias == +1:
                t_entry, entry, ret_depth, diag = enter_long_retest_second_break(bars, t_sel, U2, epsilon_atr, retest_atr, max_wait_min, bars["ATR"], min_wait_min=min_wait_min)
            else:
                t_entry, entry, ret_depth, diag = enter_short_retest_second_break(bars, t_sel, L2, epsilon_atr, retest_atr, max_wait_min, bars["ATR"], min_wait_min=min_wait_min)
            if diag.get("first_break_ts") is not None:
                n_postsel_breakouts += 1
            if t_entry is None or entry is None:
                if diag.get("first_break_ts") is None:
                    trade_log.append({"t_alert": t_gate, "skip": "no_first_break", "t_sel": t_sel})
                elif diag.get("retest_ts") is None:
                    trade_log.append({"t_alert": t_gate, "skip": "no_retest", "t_sel": t_sel, "first_break_ts": diag.get("first_break_ts")})
                else:
                    trade_log.append({"t_alert": t_gate, "skip": "no_rebreak", "t_sel": t_sel, "first_break_ts": diag.get("first_break_ts"), "retest_ts": diag.get("retest_ts")})
                continue

        # Entry ATR base
        atr_base = float(bars.loc[t_entry, "ATR"]) if entry_recalc_atr else atr_ref
        if not np.isfinite(atr_base) or atr_base <= 0:
            continue

        # Compute retest depth strictly between first_break and rebreak, then impulse/flow filters at rebreak candle
        body_atr = None
        tr_atr = None
        cvd_slope_val = None
        cvd_dz_val = None
        breach = None
        # Level at which rebreak happens (post-selection Donchian)
        level = U2 if bias == +1 else L2
        # Retest depth on segment [first_break_ts, t_entry]
        retest_depth_atr_obs = None
        try:
            fb = pd.to_datetime(diag.get("first_break_ts")) if diag.get("first_break_ts") is not None else None
            if fb is not None:
                seg = bars.loc[fb:t_entry]
                if bias == +1:
                    ret_low = float(seg["low"].min())
                    retest_depth_atr_seg = (level - ret_low) / float(atr_anchor)
                else:
                    ret_high = float(seg["high"].max())
                    retest_depth_atr_seg = (ret_high - level) / float(atr_anchor)
                retest_depth_atr_obs = float(retest_depth_atr_seg)
                if retest_depth_atr_seg > retest_atr:
                    trade_log.append({"t_alert": t_gate, "skip": "retest_depth", "t_sel": t_sel, "rebreak_ts": t_entry, "retest_depth_atr": float(retest_depth_atr_seg)})
                    continue
        except Exception:
            pass
        try:
            candle = bars.loc[t_entry]
            atr_now = float(candle["ATR"]) if np.isfinite(candle.get("ATR", np.nan)) else atr_base
            body_atr = abs(float(candle["close"]) - float(candle["open"])) / atr_now if atr_now > 0 else 0.0
            tr_atr = (float(candle["high"]) - float(candle["low"])) / atr_now if atr_now > 0 else 0.0
            breach = (bias * (float(candle["close"]) - float(level)) / atr_now) if atr_now > 0 else 0.0
            if min_body_atr is not None and body_atr < float(min_body_atr):
                trade_log.append({"t_alert": t_gate, "skip": "impulse_body", "t_sel": t_sel, "rebreak_ts": t_entry, "body_atr": float(body_atr)})
                continue
            if min_tr_atr is not None and tr_atr < float(min_tr_atr):
                trade_log.append({"t_alert": t_gate, "skip": "impulse_tr", "t_sel": t_sel, "rebreak_ts": t_entry, "tr_atr": float(tr_atr)})
                continue
            # require directional breach beyond level
            if epsilon_atr is not None and float(breach) < float(epsilon_atr):
                trade_log.append({"t_alert": t_gate, "skip": "epsilon", "t_sel": t_sel, "rebreak_ts": t_entry, "breach": float(breach)})
                continue
            if cvd_slope_min is not None:
                cvd_slope_val = micro_cvd_slope(bars, t_entry, lookback=5)
                signed_slope = float(bias) * float(cvd_slope_val)
                if signed_slope < float(cvd_slope_min):
                    trade_log.append({"t_alert": t_gate, "skip": "flow_slope", "t_sel": t_sel, "rebreak_ts": t_entry, "cvd_slope": float(cvd_slope_val), "signed_slope": float(signed_slope)})
                    continue
            if cvd_delta_z_min is not None:
                cvd_dz_val = micro_cvd_delta_z(bars, t_entry, lookback=30)
                signed_dz = float(bias) * float(cvd_dz_val)
                if signed_dz < float(cvd_delta_z_min):
                    trade_log.append({"t_alert": t_gate, "skip": "flow_dz", "t_sel": t_sel, "rebreak_ts": t_entry, "cvd_dz": float(cvd_dz_val), "signed_dz": float(signed_dz)})
                    continue
        except Exception:
            pass

        # Stop/Target
        stop_cfg = strat.get("stop", {}) if isinstance(strat.get("stop", {}), dict) else {}
        use_structure = stop_cfg.get("mode") == "structure"
        if use_structure:
            buffer_atr = float(stop_cfg.get("buffer_atr", 0.15))
            cap_atr = float(stop_cfg.get("cap_atr", 2.0))
            # retest extremes captured in diag
            atr_now = float(bars.loc[t_entry, "ATR"]) if np.isfinite(bars.loc[t_entry].get("ATR", np.nan)) else atr_base
            if bias == +1:
                # recompute retest low on segment for structure anchoring
                fb = pd.to_datetime(diag.get("first_break_ts")) if diag.get("first_break_ts") is not None else None
                if fb is not None:
                    seg = bars.loc[fb:t_entry]
                    ret_low = float(seg["low"].min())
                else:
                    ret_low = float(bars.loc[t_entry, "low"]) if t_entry in bars.index else entry - atr_now
                sl_level = ret_low - buffer_atr * atr_now
                sl_dist_atr = max(0.0, (entry - sl_level) / atr_now)
            else:
                fb = pd.to_datetime(diag.get("first_break_ts")) if diag.get("first_break_ts") is not None else None
                if fb is not None:
                    seg = bars.loc[fb:t_entry]
                    ret_high = float(seg["high"].max())
                else:
                    ret_high = float(bars.loc[t_entry, "high"]) if t_entry in bars.index else entry + atr_now
                sl_level = ret_high + buffer_atr * atr_now
                sl_dist_atr = max(0.0, (sl_level - entry) / atr_now)
            sl_dist_atr = min(sl_dist_atr, cap_atr)
            sl = entry - bias * sl_dist_atr * atr_now
        else:
            # ATR-multiple stop
            if sl_mult is None:
                # if not specified, default to 2.0
                _sm = 2.0
            else:
                _sm = float(sl_mult)
            sl = entry - bias * _sm * atr_base
            sl_dist_atr = float(_sm)

        tp = entry + bias * tp_mult * atr_base
        qty = 1.0  # unit size; PnL in price units
        sl_dist_price = abs(entry - sl)
        tp_dist_atr = float(tp_mult)

        entry_adj, fee_e = _apply_costs(entry, bias, fee_bps, slip_bps)
        best = entry
        sl_working = sl
        be_price = entry + bias * be_frac * (tp - entry)
        moved_to_be = False

        # Walk forward for exit
        exit_reason = "TIME"
        touched_sl = False
        mfe_price = 0.0
        mae_price = 0.0
        for ts, r in bars.loc[t_entry:horizon_ts].iterrows():
            hi, lo, cl = float(r["high"]), float(r["low"]), float(r["close"])
            hit_tp = (hi >= tp) if bias == +1 else (lo <= tp)
            hit_sl = (lo <= sl_working) if bias == +1 else (hi >= sl_working)

            # track excursions
            if bias == +1:
                mfe_price = max(mfe_price, hi - entry)
                mae_price = max(mae_price, entry - lo)
            else:
                mfe_price = max(mfe_price, entry - lo)
                mae_price = max(mae_price, hi - entry)

            if hit_tp and hit_sl:
                exit_px = tp
                t_exit = ts
                exit_reason = "TP"
                exit_adj, fee_x = _apply_costs(exit_px, -bias, fee_bps, slip_bps)
                pnl = (exit_adj - entry_adj) * bias * qty - (fee_e + fee_x) * qty
                pnl_r = pnl / sl_dist_price if sl_dist_price > 0 else 0.0
                trades.append(dict(
                    t_alert=t_gate, t_entry=t_entry, t_exit=t_exit, dir=bias,
                    bias=bias, entry=entry, exit=exit_px, pnl=float(pnl), pnl_r=float(pnl_r), atr_anchor=float(atr_levels),
                    exit_reason=exit_reason, touched_sl=touched_sl,
                    entry_mode="retest_second_break", retest_depth_atr=retest_depth_atr_obs,
                    t_sel=t_sel, U2=U2, L2=L2,
                    first_break_ts=diag.get("first_break_ts"), retest_ts=diag.get("retest_ts"), rebreak_ts=diag.get("rebreak_ts"),
                    adverse_atr_at_entry=float(adverse), breach=float(breach) if breach is not None else None,
                    body_atr=float(body_atr) if body_atr is not None else None,
                    tr_atr=float(tr_atr) if tr_atr is not None else None,
                    sl_dist_atr=float(sl_dist_atr), tp_dist_atr=float(tp_dist_atr),
                    cvd_slope=float(cvd_slope_val) if cvd_slope_val is not None else None,
                    cvd_dz=float(cvd_dz_val) if cvd_dz_val is not None else None,
                    mfe_atr=float(mfe_price/atr_base) if atr_base>0 else None,
                    mae_atr=float(mae_price/atr_base) if atr_base>0 else None,
                    drift=dinfo.get("drift") if dinfo else None,
                    drift_sig=dinfo.get("drift_sig") if dinfo else None,
                    cvd_z=dinfo.get("cvd_z") if dinfo else None,
                    cvd_sig=dinfo.get("cvd_sig") if dinfo else None,
                    direction_reason=dinfo.get("reason") if dinfo else None,
                ))
                break

            if hit_tp:
                exit_px = tp
                t_exit = ts
                exit_reason = "TP"
                exit_adj, fee_x = _apply_costs(exit_px, -bias, fee_bps, slip_bps)
                pnl = (exit_adj - entry_adj) * bias * qty - (fee_e + fee_x) * qty
                pnl_r = pnl / sl_dist_price if sl_dist_price > 0 else 0.0
                trades.append(dict(
                    t_alert=t_gate, t_entry=t_entry, t_exit=t_exit, dir=bias,
                    bias=bias, entry=entry, exit=exit_px, pnl=float(pnl), pnl_r=float(pnl_r), atr_anchor=float(atr_levels),
                    exit_reason=exit_reason, touched_sl=touched_sl,
                    entry_mode="retest_second_break", retest_depth_atr=retest_depth_atr_obs,
                    t_sel=t_sel, U2=U2, L2=L2,
                    first_break_ts=diag.get("first_break_ts"), retest_ts=diag.get("retest_ts"), rebreak_ts=diag.get("rebreak_ts"),
                    adverse_atr_at_entry=float(adverse), breach=float(breach) if breach is not None else None,
                    body_atr=float(body_atr) if body_atr is not None else None,
                    tr_atr=float(tr_atr) if tr_atr is not None else None,
                    sl_dist_atr=float(sl_dist_atr), tp_dist_atr=float(tp_dist_atr),
                    cvd_slope=float(cvd_slope_val) if cvd_slope_val is not None else None,
                    cvd_dz=float(cvd_dz_val) if cvd_dz_val is not None else None,
                    mfe_atr=float(mfe_price/atr_base) if atr_base>0 else None,
                    mae_atr=float(mae_price/atr_base) if atr_base>0 else None,
                    drift=dinfo.get("drift") if dinfo else None,
                    drift_sig=dinfo.get("drift_sig") if dinfo else None,
                    cvd_z=dinfo.get("cvd_z") if dinfo else None,
                    cvd_sig=dinfo.get("cvd_sig") if dinfo else None,
                    direction_reason=dinfo.get("reason") if dinfo else None,
                ))
                break

            if hit_sl:
                exit_px = sl_working
                t_exit = ts
                exit_reason = "SL"
                touched_sl = True
                exit_adj, fee_x = _apply_costs(exit_px, -bias, fee_bps, slip_bps)
                pnl = (exit_adj - entry_adj) * bias * qty - (fee_e + fee_x) * qty
                pnl_r = pnl / sl_dist_price if sl_dist_price > 0 else 0.0
                trades.append(dict(
                    t_alert=t_gate, t_entry=t_entry, t_exit=t_exit, dir=bias,
                    bias=bias, entry=entry, exit=exit_px, pnl=float(pnl), pnl_r=float(pnl_r), atr_anchor=float(atr_levels),
                    exit_reason=exit_reason, touched_sl=touched_sl,
                    entry_mode="retest_second_break", retest_depth_atr=retest_depth_atr_obs,
                    t_sel=t_sel, U2=U2, L2=L2,
                    first_break_ts=diag.get("first_break_ts"), retest_ts=diag.get("retest_ts"), rebreak_ts=diag.get("rebreak_ts"),
                    adverse_atr_at_entry=float(adverse), breach=float(breach) if breach is not None else None,
                    body_atr=float(body_atr) if body_atr is not None else None,
                    tr_atr=float(tr_atr) if tr_atr is not None else None,
                    sl_dist_atr=float(sl_dist_atr), tp_dist_atr=float(tp_dist_atr),
                    cvd_slope=float(cvd_slope_val) if cvd_slope_val is not None else None,
                    cvd_dz=float(cvd_dz_val) if cvd_dz_val is not None else None,
                    mfe_atr=float(mfe_price/atr_base) if atr_base>0 else None,
                    mae_atr=float(mae_price/atr_base) if atr_base>0 else None,
                    drift=dinfo.get("drift") if dinfo else None,
                    drift_sig=dinfo.get("drift_sig") if dinfo else None,
                    cvd_z=dinfo.get("cvd_z") if dinfo else None,
                    cvd_sig=dinfo.get("cvd_sig") if dinfo else None,
                    direction_reason=dinfo.get("reason") if dinfo else None,
                ))
                break

            best = max(best, hi) if bias == +1 else min(best, lo)
            if not moved_to_be and ((bias == +1 and cl >= be_price) or (bias == -1 and cl <= be_price)):
                sl_working = entry
                moved_to_be = True
            if moved_to_be and trail_mult is not None:
                trail_dist = float(trail_mult) * atr_base
                tgt = best - bias * trail_dist
                if (bias == +1 and tgt > sl_working) or (bias == -1 and tgt < sl_working):
                    sl_working = tgt
        else:
            # TIME exit
            exit_px = float(bars.loc[horizon_ts, "close"]) if horizon_ts in bars.index else float(bars.iloc[-1]["close"]) 
            t_exit = horizon_ts if horizon_ts in bars.index else bars.index[-1]
            exit_adj, fee_x = _apply_costs(exit_px, -bias, fee_bps, slip_bps)
            pnl = (exit_adj - entry_adj) * bias * qty - (fee_e + fee_x) * qty
            pnl_r = pnl / sl_dist_price if sl_dist_price > 0 else 0.0
            trades.append(dict(
                t_alert=t_gate, t_entry=t_entry, t_exit=t_exit, dir=bias,
                bias=bias, entry=entry, exit=exit_px, pnl=float(pnl), pnl_r=float(pnl_r), atr_anchor=float(atr_levels),
                exit_reason="TIME", touched_sl=touched_sl,
                entry_mode="retest_second_break", retest_depth_atr=retest_depth_atr_obs,
                t_sel=t_sel, U2=U2, L2=L2,
                first_break_ts=diag.get("first_break_ts"), retest_ts=diag.get("retest_ts"), rebreak_ts=diag.get("rebreak_ts"),
                adverse_atr_at_entry=float(adverse),
                drift=dinfo.get("drift") if dinfo else None,
                drift_sig=dinfo.get("drift_sig") if dinfo else None,
                cvd_z=dinfo.get("cvd_z") if dinfo else None,
                cvd_sig=dinfo.get("cvd_sig") if dinfo else None,
                direction_reason=dinfo.get("reason") if dinfo else None,
            ))

    trades_df = pd.DataFrame(trades)
    pathlib.Path(out["dir"]).mkdir(parents=True, exist_ok=True)
    if len(trades_df):
        trades_df.to_csv(f"{out['dir']}/vol_bracket_trades.csv", index=False)
    # Metrics
    metrics = trade_metrics(trades_df)
    metrics["alerts"] = int(len(alerts))

    # Direction summary
    from collections import Counter
    reasons = Counter([t.get("direction_reason") for t in trades if t.get("direction_reason")])
    bias_counts = {
        "+1": sum(1 for t in trades if t.get("dir") == +1),
        "-1": sum(1 for t in trades if t.get("dir") == -1),
        "None": 0,
    }
    print("[Direction] reasons:", dict(reasons))
    print("[Direction] bias counts:", bias_counts)
    modes = Counter([t.get("entry_mode") for t in trades])
    print("[Entry] modes:", dict(modes))
    skips = Counter([t.get("skip") for t in trade_log if t.get("skip")])
    print("[Skips]", dict(skips))

    # Debug diagnostics (truth serum)
    wanted = ["direction","adverse_guard","retest_depth","epsilon","impulse_body","impulse_tr","flow_slope","flow_dz","no_first_break"]
    counts = {k: int(skips.get(k, 0)) for k in wanted}
    print("counts:", counts)

    # Per-trade medians for key entry metrics
    def _med(s):
        try:
            return float(pd.to_numeric(s, errors="coerce").median()) if len(s) else float("nan")
        except Exception:
            return float("nan")
    if len(trades_df):
        meds = {
            "retest_depth_atr": _med(trades_df.get("retest_depth_atr")),
            "sl_dist_atr": _med(trades_df.get("sl_dist_atr")),
            "body_atr": _med(trades_df.get("body_atr")),
            "tr_atr": _med(trades_df.get("tr_atr")),
            "breach": _med(trades_df.get("breach")),
        }
        print("per-trade medians:", meds)

        # MFE/MAE medians (ATR) for winners vs losers
        winners = trades_df[trades_df["pnl"] > 0]
        losers = trades_df[trades_df["pnl"] <= 0]
        mfe_mae = {
            "winners": {"mfe_atr": _med(winners.get("mfe_atr")), "mae_atr": _med(winners.get("mae_atr"))},
            "losers": {"mfe_atr": _med(losers.get("mfe_atr")), "mae_atr": _med(losers.get("mae_atr"))},
        }
        print("MFE/MAE medians (ATR):", mfe_mae)

    json.dump(metrics, open(f"{out['dir']}/vol_bracket_metrics.json", "w"), indent=2, default=float)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vol_bracket.yaml")
    args = ap.parse_args()
    run_from_config(args.config)


