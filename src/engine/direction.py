from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Dict
import pandas as pd
import numpy as np

Mode = Literal["none","drift","cvd","hybrid_and","hybrid_adaptive","absorption"]

@dataclass
class DriftCfg:
    M_min: int = 3              # minutes to wait after gate
    eps_atr: float = 0.05       # require |drift| >= eps*ATR_ref (loosened)

@dataclass
class CvdCfg:
    source: Literal["tick","bar_proxy"] = "bar_proxy"
    baseline_min: int = 240     # lookback for z-score baseline
    z_thr: float = 0.4          # require |z| >= z_thr (loosened)
    z_strong: float = 1.2       # strong CVD threshold for adaptive mode

@dataclass
class DirCfg:
    mode: Mode = "hybrid_and"
    drift: DriftCfg = field(default_factory=DriftCfg)
    cvd: CvdCfg = field(default_factory=CvdCfg)
    anchor_price: Literal["gate","select"] = "select"  # use gate close vs post-M close as P0

def _bar_proxy_signed_volume(bars: pd.DataFrame) -> pd.Series:
    # bars index: ts (UTC minute); columns: open, high, low, close, volume
    # sign by bar return
    ret = bars["close"].diff()
    sgn = np.sign(ret).fillna(0.0)
    sv = (bars["volume"].fillna(0.0) * sgn).rename("sv")
    return sv

def _cvd_over(bars: pd.DataFrame, start: pd.Timestamp, M: int) -> float:
    # bar-proxy CVD over [start, start+M)
    w = bars.loc[start : start + pd.Timedelta(minutes=M)]
    if len(w) == 0: return np.nan
    return float(_bar_proxy_signed_volume(w).sum())

def _cvd_zscore(bars: pd.DataFrame, gate_t: pd.Timestamp, M: int, baseline_min: int) -> Tuple[float,float,float]:
    # compute bar-proxy CVD over rolling M-min windows in [gate_t - baseline_min, gate_t)
    start = gate_t - pd.Timedelta(minutes=baseline_min + M)
    base = bars.loc[start: gate_t]
    if len(base) < (baseline_min // 2):  # not enough history
        return np.nan, np.nan, np.nan
    sv = _bar_proxy_signed_volume(base)
    # rolling M-min sum aligned to end
    cvd_M = sv.rolling(window=M, min_periods=max(1, M//2)).sum()
    base_vals = cvd_M.loc[:gate_t - pd.Timedelta(minutes=1)].dropna()
    if len(base_vals) < 30:
        return np.nan, np.nan, np.nan
    mu, sd = float(base_vals.mean()), float(base_vals.std(ddof=1) or 0.0)
    cur = _cvd_over(bars, gate_t, M)
    if not np.isfinite(cur) or sd == 0.0:
        return np.nan, mu, sd
    z = (cur - mu) / sd
    return float(z), mu, sd

def select_direction(
    bars: pd.DataFrame,                  # 1m OHLCV, index=ts (UTC), cols [open,high,low,close,volume, ATR(optional)]
    gate_t: pd.Timestamp,
    atr_ref: float,                      # ATR at reference (gate)
    cfg: DirCfg,
    absorption_event: Optional[Dict] = None
) -> Tuple[Optional[int], Dict]:
    """
    Returns (dir, info) where dir in {+1 (long), -1 (short), None}, and info has diagnostics.
    We wait M minutes from gate for selection, then decide bias. Entry happens at/after that.
    """
    M = cfg.drift.M_min
    t_sel = gate_t + pd.Timedelta(minutes=M)
    if t_sel > bars.index.max():
        return None, {"reason":"t_sel_out_of_range"}

    # Drift over [gate_t, t_sel]
    p_gate = float(bars.loc[bars.index[bars.index.searchsorted(gate_t)]] ["close"])
    p_sel  = float(bars.loc[bars.index[bars.index.searchsorted(t_sel)]]  ["close"])
    drift = p_sel - p_gate
    drift_sig = (atr_ref > 0) and (abs(drift) >= cfg.drift.eps_atr * atr_ref)
    drift_dir = +1 if drift > 0 else (-1 if drift < 0 else 0)

    # CVD z-score over first M minutes post-gate (bar proxy)
    z, mu, sd = _cvd_zscore(bars, gate_t, M, cfg.cvd.baseline_min)
    cvd_sig = np.isfinite(z) and (abs(z) >= cfg.cvd.z_thr)
    cvd_dir = +1 if (np.isfinite(z) and z > 0) else (-1 if (np.isfinite(z) and z < 0) else 0)

    info = dict(
        gate=str(gate_t), t_sel=str(t_sel),
        drift=float(drift), drift_dir=int(drift_dir), drift_sig=bool(drift_sig),
        cvd_z=float(z) if np.isfinite(z) else None, cvd_dir=int(cvd_dir), cvd_sig=bool(cvd_sig),
        cvd_mu=float(mu) if np.isfinite(mu) else None, cvd_sd=float(sd) if np.isfinite(sd) else None,
        atr_ref=float(atr_ref)
    )

    if cfg.mode == "absorption":
        # Bias supplied externally via absorption event
        if absorption_event and isinstance(absorption_event, dict):
            d = int(absorption_event.get("direction", 0))
            info = {"reason":"absorption_event","t_sel":str(absorption_event.get("t_abs"))}
            return (d if d in (-1,0,1) else None), info
        return None, {**info, "reason":"absorption_needed"}

    if cfg.mode == "none":
        return None, {**info, "reason":"mode_none"}

    if cfg.mode == "drift":
        if drift_sig and drift_dir != 0:
            return drift_dir, {**info, "reason":"drift_only_pass"}
        return None, {**info, "reason":"drift_only_fail"}

    if cfg.mode == "cvd":
        if cvd_sig and cvd_dir != 0:
            return cvd_dir, {**info, "reason":"cvd_only_pass"}
        return None, {**info, "reason":"cvd_only_fail"}

    if cfg.mode == "hybrid_adaptive":
        # Strong CVD can pass solo
        if cvd_sig and np.isfinite(z) and abs(float(z)) >= float(cfg.cvd.z_strong) and cvd_dir != 0:
            return cvd_dir, {**info, "reason":"cvd_strong_pass"}
        # Otherwise require both to agree
        if drift_sig and cvd_sig and (drift_dir == cvd_dir) and drift_dir != 0:
            return drift_dir, {**info, "reason":"hybrid_pass"}
        return None, {**info, "reason":"hybrid_adaptive_fail"}

    # hybrid AND: both present and agree
    if drift_sig and cvd_sig and (drift_dir == cvd_dir) and drift_dir != 0:
        return drift_dir, {**info, "reason":"hybrid_pass"}
    return None, {**info, "reason":"hybrid_fail"}

    # (unreachable in current flow)

    
    
