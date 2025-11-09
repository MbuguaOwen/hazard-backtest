import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import yaml
from src.io_ohlcv import read_any_ohlcv
from src.atr_basic import atr

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

def load_cfg(p):
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    files = cfg.get("files", {})
    study = cfg.get("study", {})
    out = cfg.get("out", {"dir":"results/event_study"})

    ohlcv_pat = files.get("ohlcv")
    alerts_csv = files.get("alerts_csv")
    if not ohlcv_pat or not alerts_csv:
        raise ValueError("Config must have files.ohlcv and files.alerts_csv")

    bars = read_any_ohlcv(ohlcv_pat).sort_values("ts").set_index("ts")
    bars["ATR"] = atr(bars[["open","high","low","close"]], window=int(study.get("atr_window", 60)))
    close = bars["close"].astype(float)
    abs_lr = (np.log(close).diff().abs()).rename("abs_lr")

    alerts = pd.read_csv(alerts_csv)
    if "t_alert" not in alerts.columns:
        raise ValueError("alerts_csv must contain 't_alert'")
    alerts = pd.to_datetime(alerts["t_alert"], utc=True).dt.floor("min")
    alerts = pd.DatetimeIndex(alerts[(alerts >= bars.index.min()) & (alerts <= bars.index.max())].sort_values().unique())

    H = int(study.get("horizon_min", 180))
    rv_all = abs_lr.shift(-1).rolling(H, min_periods=H).sum()

    prem = cfg.get("premium", {"mode":"quantile","q":0.50,"premium_mult":0.60})
    if prem.get("mode","quantile") == "quantile":
        base = float(rv_all.dropna().quantile(float(prem.get("q", 0.50))))
        premium_val = float(prem.get("premium_mult", 0.60)) * base
    else:
        premium_val = float(prem.get("value", 0.01))

    rows = []
    for ta in tqdm(alerts, desc="Event study"):
        if ta not in bars.index:
            ix = bars.index.searchsorted(ta)
            if ix >= len(bars):
                continue
            ta = bars.index[ix]
        rv = rv_all.reindex(bars.index).loc[ta]
        atr_ref = float(bars.loc[ta, "ATR"]) if pd.notna(bars.loc[ta, "ATR"]) else np.nan
        rows.append(dict(t_alert=ta, rv=float(rv) if np.isfinite(rv) else np.nan,
                         atr_ref=atr_ref, pnl=float(rv - premium_val) if np.isfinite(rv) else np.nan))

    df = pd.DataFrame(rows).dropna(subset=["rv"]).sort_values("t_alert")
    pathlib.Path(out["dir"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out['dir']}/event_level.csv", index=False)

    # Summary
    res = dict(
        n=int(len(df)),
        mean_rv=float(df["rv"].mean()) if len(df) else 0.0,
        mean_pnl=float(df["pnl"].mean()) if len(df) else 0.0,
        premium=float(premium_val),
    )
    json.dump(res, open(f"{out['dir']}/summary.json","w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/event_study.yaml")
    args = ap.parse_args()
    main(args.config)

