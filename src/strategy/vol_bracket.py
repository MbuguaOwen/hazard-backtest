
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
import pandas as pd
import numpy as np

from ..atr import atr
from ..utils import ensure_datetime_index, tie_break_same_bar

@dataclass
class VolBracketParams:
    k_breakout: float = 1.25
    atr_window: int = 60
    sl_mult: float = 10.50
    tp_mult: float = 40.0
    be_frac: float = 0.40
    trail_mult: Optional[float] = None
    horizon_min: int = 180
    reference: str = "last"
    entry_recalc_atr: bool = False
    oco: bool = True
    fee_bps: float = 1.0
    slip_bps: float = 0.5
    entry_confirm: Optional[Dict[str, Any]] = None
    entry_filters: Optional[Dict[str, Any]] = None

@dataclass
class RiskParams:
    per_alert_r: float = 0.0025
    max_total_r: float = 0.01
    equity: float = 100.0

class VolBracketStrategy:
    def __init__(self, ohlcv: pd.DataFrame, params: VolBracketParams, risk: RiskParams):
        self.bars = ensure_datetime_index(ohlcv.copy(), col="ts")
        self.params = params
        self.risk = risk
        self.bars["ATR"] = atr(self.bars[["open","high","low","close"]], window=params.atr_window)

    def _qty_for_risk(self, entry: float, sl: float) -> float:
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            return 0.0
        cash_risk = self.risk.per_alert_r * self.risk.equity
        return cash_risk / risk_per_unit

    def _simulate_one(self, t_alert: pd.Timestamp):
        bars = self.bars
        if t_alert not in bars.index:
            idx = bars.index.searchsorted(t_alert)
            if idx >= len(bars):
                return None
            t_alert = bars.index[idx]

        row0 = bars.loc[t_alert]
        atr_ref = float(row0["ATR"])
        if not np.isfinite(atr_ref) or atr_ref <= 0:
            return None

        p0 = float(row0["close"])
        U = p0 + self.params.k_breakout * atr_ref
        L = p0 - self.params.k_breakout * atr_ref

        horizon_ts = t_alert + pd.Timedelta(minutes=self.params.horizon_min)
        df = bars.loc[t_alert:horizon_ts]

        trig_ts = None
        direction = None
        entry = None
        entry_mode_hit = None  # "touch" | "first_close" | "retest_second_break"
        retest_depth_atr = None

        ec = self.params.entry_confirm or {}
        mode = str(ec.get("mode", "next_bar_hold")).lower()
        eps = float(ec.get("epsilon_atr", 0.05)) * atr_ref
        max_wait = int(ec.get("max_wait_min", 60))
        filt = self.params.entry_filters or {}
        range_mult = float(filt.get("range_expansion_mult", 0.0) or 0.0)
        prev_close = p0

        if mode == "next_bar_hold":
            armed_up = True
            armed_dn = True
            t_touched = None
            touched_dir = None
            t_start = df.index[0]
            for ts, r in df.iloc[1:].iterrows():
                hi, lo, op, cl = float(r["high"]), float(r["low"]), float(r["open"]), float(r["close"])
                # detect first touch beyond levels
                touched_up = armed_up and (hi >= U)
                touched_dn = armed_dn and (lo <= L)

                if t_touched is None and (touched_up or touched_dn):
                    t_touched = ts
                    touched_dir = +1 if touched_up else -1
                    prev_close = cl
                    continue

                # after a touch, require next bar hold + close beyond level
                if t_touched is not None:
                    if touched_dir == +1:
                        hold = (lo >= U - eps) and (cl >= U)
                        if hold:
                            # range-expansion filter (true range vs ATR at entry bar)
                            if range_mult > 0:
                                tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
                                atr_here = float(r.get("ATR", atr_ref))
                                if not (tr >= range_mult * atr_here):
                                    # fail filter, re-arm and continue
                                    t_touched, touched_dir = None, None
                                    armed_up = True
                                    armed_dn = True
                                    prev_close = cl
                                    continue
                            direction, entry, trig_ts = +1, cl, ts
                            entry_mode_hit = "touch"
                            break
                        else:
                            # re-arm if failed hold
                            t_touched, touched_dir = None, None
                            armed_up = True
                            armed_dn = True
                    else:
                        hold = (hi <= L + eps) and (cl <= L)
                        if hold:
                            if range_mult > 0:
                                tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
                                atr_here = float(r.get("ATR", atr_ref))
                                if not (tr >= range_mult * atr_here):
                                    t_touched, touched_dir = None, None
                                    armed_up = True
                                    armed_dn = True
                                    prev_close = cl
                                    continue
                            direction, entry, trig_ts = -1, cl, ts
                            entry_mode_hit = "touch"
                            break
                        else:
                            t_touched, touched_dir = None, None
                            armed_up = True
                            armed_dn = True

                # give up waiting after max_wait minutes since touch
                if t_touched is not None and (ts - t_touched) >= pd.Timedelta(minutes=max_wait):
                    t_touched, touched_dir = None, None
                prev_close = cl

        elif mode == "retest_second_break":
            # First get a close beyond U/L, then retest near level within retest_atr and no deep close inside,
            # then enter on second close beyond.
            retest = float(ec.get("retest_atr", 0.10)) * atr_ref
            first_dir = None
            t_first = None
            for ts, r in df.iloc[1:].iterrows():
                hi, lo, op, cl = float(r["high"]), float(r["low"]), float(r["open"]), float(r["close"])
                if first_dir is None:
                    if cl >= U + eps:
                        first_dir = +1
                        t_first = ts
                        prev_close = cl
                        continue
                    if cl <= L - eps:
                        first_dir = -1
                        t_first = ts
                        prev_close = cl
                        continue
                else:
                    # enforce max wait from first close
                    if (ts - t_first) >= pd.Timedelta(minutes=max_wait):
                        break
                    if first_dir == +1:
                        # retest toward U: bar low near U but close not far inside
                        near = (lo >= U - retest)
                        not_inside = (cl >= U - retest)
                        if near and not_inside and cl >= U + eps:
                            if range_mult > 0:
                                tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
                                atr_here = float(r.get("ATR", atr_ref))
                                if not (tr >= range_mult * atr_here):
                                    prev_close = cl
                                    continue
                            # measure retest depth in ATRs
                            atr_here = float(r.get("ATR", atr_ref))
                            retest_depth_atr = max(0.0, (U - lo)) / atr_here if atr_here > 0 else None
                            direction, entry, trig_ts = +1, cl, ts
                            entry_mode_hit = "retest_second_break"
                            break
                    else:
                        near = (hi <= L + retest)
                        not_inside = (cl <= L + retest)
                        if near and not_inside and cl <= L - eps:
                            if range_mult > 0:
                                tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
                                atr_here = float(r.get("ATR", atr_ref))
                                if not (tr >= range_mult * atr_here):
                                    prev_close = cl
                                    continue
                            atr_here = float(r.get("ATR", atr_ref))
                            retest_depth_atr = max(0.0, (hi - L)) / atr_here if atr_here > 0 else None
                            direction, entry, trig_ts = -1, cl, ts
                            entry_mode_hit = "retest_second_break"
                            break
                prev_close = cl
        else:
            # Fallback to simple second-break close with epsilon
            for ts, r in df.iloc[1:].iterrows():
                hi, lo, op, cl = float(r["high"]), float(r["low"]), float(r["open"]), float(r["close"])
                if cl >= U + eps:
                    if range_mult > 0:
                        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
                        atr_here = float(r.get("ATR", atr_ref))
                        if not (tr >= range_mult * atr_here):
                            prev_close = cl
                            continue
                    direction, entry, trig_ts = +1, cl, ts
                    entry_mode_hit = "first_close"
                    break
                if cl <= L - eps:
                    if range_mult > 0:
                        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
                        atr_here = float(r.get("ATR", atr_ref))
                        if not (tr >= range_mult * atr_here):
                            prev_close = cl
                            continue
                    direction, entry, trig_ts = -1, cl, ts
                    entry_mode_hit = "first_close"
                    break

        if trig_ts is None:
            return dict(t_alert=t_alert, t_entry=None, t_exit=None, dir=0, entry=np.nan, exit=np.nan, pnl=0.0)

        atr_base = float(bars.loc[trig_ts]["ATR"]) if self.params.entry_recalc_atr else atr_ref
        sl = entry - direction * self.params.sl_mult * atr_base
        tp = entry + direction * self.params.tp_mult * atr_base
        qty = self._qty_for_risk(entry, sl)
        if qty <= 0:
            return None

        best = entry
        sl_working = sl
        be_price = entry + direction * self.params.be_frac * (tp - entry)
        moved_to_be = False

        def _apply_costs(px: float, direction: int, fee_bps: float, slip_bps: float):
            slip = px * (slip_bps * 1e-4)
            px_adj = px + (-direction) * slip
            fee = px * (fee_bps * 1e-4)
            return px_adj, fee

        entry_adj, fee_e = _apply_costs(entry, direction, self.params.fee_bps, self.params.slip_bps)

        # outcome tracking
        touched_sl = False
        exit_reason = "TIME"  # default unless we hit TP/SL earlier

        for ts, r in bars.loc[trig_ts:horizon_ts].iterrows():
            hi, lo, op, cl = float(r["high"]), float(r["low"]), float(r["open"]), float(r["close"])
            hit_tp = (hi >= tp) if direction == +1 else (lo <= tp)
            hit_sl = (lo <= sl_working) if direction == +1 else (hi >= sl_working)

            if hit_tp and hit_sl:
                exit_px = tp
                t_exit = ts
                exit_reason = "TP"
                exit_adj, fee_x = _apply_costs(exit_px, -direction, self.params.fee_bps, self.params.slip_bps)
                pnl = (exit_adj - entry_adj)*direction*qty - (fee_e + fee_x)*qty
                return dict(t_alert=t_alert, t_entry=trig_ts, t_exit=t_exit, dir=direction,
                            entry=entry, exit=exit_px, pnl=float(pnl),
                            exit_reason=exit_reason, touched_sl=touched_sl,
                            entry_mode_hit=entry_mode_hit, retest_depth_atr=retest_depth_atr)

            if hit_tp:
                exit_px = tp
                t_exit = ts
                exit_reason = "TP"
                exit_adj, fee_x = _apply_costs(exit_px, -direction, self.params.fee_bps, self.params.slip_bps)
                pnl = (exit_adj - entry_adj)*direction*qty - (fee_e + fee_x)*qty
                return dict(t_alert=t_alert, t_entry=trig_ts, t_exit=t_exit, dir=direction,
                            entry=entry, exit=exit_px, pnl=float(pnl),
                            exit_reason=exit_reason, touched_sl=touched_sl,
                            entry_mode_hit=entry_mode_hit, retest_depth_atr=retest_depth_atr)

            if hit_sl:
                exit_px = sl_working
                t_exit = ts
                exit_reason = "SL"
                touched_sl = True
                exit_adj, fee_x = _apply_costs(exit_px, -direction, self.params.fee_bps, self.params.slip_bps)
                pnl = (exit_adj - entry_adj)*direction*qty - (fee_e + fee_x)*qty
                return dict(t_alert=t_alert, t_entry=trig_ts, t_exit=t_exit, dir=direction,
                            entry=entry, exit=exit_px, pnl=float(pnl),
                            exit_reason=exit_reason, touched_sl=touched_sl,
                            entry_mode_hit=entry_mode_hit, retest_depth_atr=retest_depth_atr)

            best = max(best, hi) if direction == +1 else min(best, lo)
            # Move to BE when threshold reached
            if not moved_to_be and ((direction == +1 and cl >= be_price) or (direction == -1 and cl <= be_price)):
                sl_working = entry
                moved_to_be = True

            # Trail after BE: lifting SL is not an SL touch
            if moved_to_be and self.params.trail_mult is not None:
                trail_dist = self.params.trail_mult * atr_base
                tgt = best - direction*trail_dist
                if (direction == +1 and tgt > sl_working) or (direction == -1 and tgt < sl_working):
                    sl_working = tgt

        exit_px = float(bars.loc[horizon_ts]["close"]) if horizon_ts in bars.index else float(bars.iloc[-1]["close"])
        t_exit = horizon_ts if horizon_ts in bars.index else bars.index[-1]
        exit_adj, fee_x = _apply_costs(exit_px, -direction, self.params.fee_bps, self.params.slip_bps)
        pnl = (exit_adj - entry_adj)*direction*qty - (fee_e + fee_x)*qty
        exit_reason = "TIME"
        return dict(t_alert=t_alert, t_entry=trig_ts, t_exit=t_exit, dir=direction,
                    entry=entry, exit=exit_px, pnl=float(pnl),
                    exit_reason=exit_reason, touched_sl=touched_sl,
                    entry_mode_hit=entry_mode_hit, retest_depth_atr=retest_depth_atr)

    def run(self, alerts: pd.Series) -> pd.DataFrame:
        alerts = pd.DatetimeIndex(pd.to_datetime(alerts.dropna().unique(), utc=True)).sort_values()
        rows = []
        open_exits = []  # list of exit timestamps for active trades
        for t in tqdm(alerts, desc="Simulating alerts"):
            # release finished slots
            open_exits = [te for te in open_exits if te > t]
            active_risk = self.risk.per_alert_r * len(open_exits)

            # can we spawn?
            if active_risk + self.risk.per_alert_r > self.risk.max_total_r:
                continue

            res = self._simulate_one(t)
            if res is None:
                continue
            rows.append(res)
            if res["t_exit"] is not None:
                open_exits.append(res["t_exit"])
        return pd.DataFrame(rows)

