
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
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
    files = cfg["files"]
    study = cfg["study"]
    out = cfg["out"]

    bars = read_any_ohlcv(files["ohlcv"]).sort_values("ts").set_index("ts")
    bars["ATR"] = atr(bars[["open","high","low","close"]], window=study.get("atr_window", 60))

    alerts = pd.read_csv(files["alerts_csv"])
    if "t_alert" not in alerts.columns:
        raise ValueError("alerts_csv must contain 't_alert'")
    alerts = pd.to_datetime(alerts["t_alert"], utc=True).dt.floor("min")
    alerts = alerts[(alerts >= bars.index.min()) & (alerts <= bars.index.max())]
    alerts = pd.DatetimeIndex(alerts.sort_values().unique())

    H = pd.Timedelta(minutes=study["horizon_min"])
    thr_list = study.get("thr_atr", [2,3,4,6,8,10,15,20,30,50,75,100])

    rows = []
    for t in tqdm(alerts, desc="Measuring excursions"):
        if t not in bars.index:
            ix = bars.index.searchsorted(t)
            if ix >= len(bars): continue
            t = bars.index[ix]
        p0 = float(bars.loc[t, "close"])
        a0 = float(bars.loc[t, "ATR"])
        if not pd.notna(a0) or a0 <= 0: continue
        w = bars.loc[t:t+H]
        up = (w["high"].max() - p0) / a0
        dn = (p0 - w["low"].min()) / a0
        max_exc = max(up, dn)
        net = (float(w.iloc[-1]["close"]) - p0) / a0
        rows.append(dict(t_alert=t, atr=a0, up_atr=up, dn_atr=dn, max_exc_atr=max_exc, net_move_atr=net))

    df = pd.DataFrame(rows).sort_values("t_alert")
    pathlib.Path(out["dir"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out['dir']}/excursions.csv", index=False)

    # exceedance table
    ex_rows = []
    for thr in thr_list:
        ex_rows.append(dict(thr_atr=thr, frac=(df["max_exc_atr"] >= thr).mean()))
    ex = pd.DataFrame(ex_rows)
    ex.to_csv(f"{out['dir']}/exceedance.csv", index=False)

    # summary
    q = df["max_exc_atr"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    summary = dict(
        n_alerts=int(len(df)),
        median=float(q.get(0.5, float("nan"))),
        p75=float(q.get(0.75, float("nan"))),
        p90=float(q.get(0.9, float("nan"))),
        p95=float(q.get(0.95, float("nan"))),
        p99=float(q.get(0.99, float("nan"))),
        exceedance={str(r["thr_atr"]): float(r["frac"]) for _, r in ex.iterrows()},
    )
    open(f"{out['dir']}/summary.json","w").write(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/excursions.yaml")
    args = ap.parse_args()
    main(args.config)
