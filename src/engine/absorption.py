from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from .footprint import rolling_z, rolling_lambda

@dataclass
class AbsorptionCfg:
    lookback_s: int = 60
    z_delta_thr: float = 2.5
    tr_max_atr: float = 0.30
    wick_bias_min: float = 0.65
    donchian_prox_atr: Optional[float] = 0.35
    lambda_drop: bool = True
    alpha: float = 1.0  # score = z(|delta|) - alpha * z(TR/ATR60)

def _wick_bias(row) -> float:
    tr = float(row["tr"]) if np.isfinite(row.get("tr", np.nan)) else 0.0
    if tr <= 0:
        return 0.5
    return max(0.0, min(1.0, (row["close"] - row["low"]) / tr))

def absorption_events(micro: pd.DataFrame, cfg: AbsorptionCfg, donch_hi: pd.Series = None, donch_lo: pd.Series = None) -> pd.DataFrame:
    """
    micro: must include columns [delta, tr, close, low, high, ATR60]; index is micro ts.
    Returns events dataframe with:
      t_abs, direction (+1 long bias, -1 short bias), abs_low, abs_high, score, wick, z_abs, z_tratr, lambda, lambda_below_med, near_donch
    """
    cols = ["t_abs","direction","abs_low","abs_high","score","wick","z_abs","z_tratr","lambda","lambda_below_med","near_donch","delta","tr","close","ATR60"]
    if micro is None or len(micro) == 0:
        return pd.DataFrame(columns=cols)
    z_abs = rolling_z(micro["delta"].abs(), cfg.lookback_s)
    z_tratr = rolling_z((micro["tr"] / micro["ATR60"]).replace([np.inf, -np.inf], np.nan), cfg.lookback_s)
    score = z_abs - cfg.alpha * z_tratr

    wick = micro.apply(_wick_bias, axis=1)
    if cfg.lambda_drop:
        lam = rolling_lambda(micro, lookback=cfg.lookback_s)
        lam_med = lam.rolling(30, min_periods=10).median()
        lam_drop = lam < lam_med
    else:
        lam = pd.Series(index=micro.index, dtype="float64")
        lam_drop = pd.Series(False, index=micro.index)

    near = pd.Series(True, index=micro.index, dtype="bool")
    if cfg.donchian_prox_atr is not None and donch_hi is not None and donch_lo is not None:
        # Align Donchian to micro index
        dhi = donch_hi.reindex(micro.index, method="ffill")
        dlo = donch_lo.reindex(micro.index, method="ffill")
        dist_hi = (dhi - micro["close"]).abs() / micro["ATR60"]
        dist_lo = (micro["close"] - dlo).abs() / micro["ATR60"]
        near = (dist_hi <= cfg.donchian_prox_atr) | (dist_lo <= cfg.donchian_prox_atr)

    cond_core = (score >= cfg.z_delta_thr) & (micro["ATR60"] > 0)
    sells_abs = (micro["delta"] < 0) & (wick >= cfg.wick_bias_min)
    buys_abs  = (micro["delta"] > 0) & ((1.0 - wick) >= cfg.wick_bias_min)

    cond = cond_core & near
    if cfg.lambda_drop:
        cond = cond & lam_drop.fillna(False)

    fire = cond & (sells_abs | buys_abs)
    if not fire.any():
        return pd.DataFrame(columns=cols)
    out = micro.loc[fire].copy()
    out["score"] = score.loc[fire]
    out["wick"] = wick.loc[fire]
    out["z_abs"] = z_abs.loc[fire]
    out["z_tratr"] = z_tratr.loc[fire]
    out["lambda"] = lam.loc[fire]
    out["lambda_below_med"] = lam_drop.loc[fire].fillna(False)
    out["near_donch"] = near.loc[fire].fillna(True)

    # direction opposite the aggressor
    out["direction"] = np.where(out["delta"] < 0, +1, -1)
    out["abs_low"]  = micro.loc[fire, "low"]
    out["abs_high"] = micro.loc[fire, "high"]
    out = out[["direction","abs_low","abs_high","score","wick","z_abs","z_tratr","lambda","lambda_below_med","near_donch","delta","tr","close","ATR60"]]
    out = out.rename_axis("t_abs").reset_index()
    return out
