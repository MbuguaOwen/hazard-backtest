
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from src.metrics import trade_metrics

def _norm(df, col):
    s = pd.to_datetime(df[col], utc=True, errors="coerce").dt.floor("min")
    df = df.copy()
    df[col] = s
    return df.dropna(subset=[col])

def load_trades(path):
    df = pd.read_csv(path)
    if "t_alert" not in df.columns:
        raise ValueError(f"{path} missing t_alert")
    df = _norm(df, "t_alert")
    if "pnl" not in df.columns and {"premium","rv"}.issubset(df.columns):
        df["pnl"] = df["rv"] - df["premium"]
    if "pnl" not in df.columns:
        raise ValueError(f"{path} missing pnl or (premium, rv)")
    return df[["t_alert","pnl"]].sort_values("t_alert")

def main(bcsv, pcsv, out_json):
    b = load_trades(bcsv)
    p = load_trades(pcsv)
    inter = sorted(set(b["t_alert"]) & set(p["t_alert"]))
    b_i = b[b["t_alert"].isin(inter)].set_index("t_alert").sort_index()
    p_i = p[p["t_alert"].isin(inter)].set_index("t_alert").sort_index()
    df = pd.DataFrame({"pnl_bracket": b_i["pnl"], "pnl_proxy": p_i["pnl"]})

    corr = df.corr().iloc[0,1] if len(df) >= 2 else float("nan")

    out = dict(
        counts=dict(
            bracket_all=len(b),
            proxy_all=len(p),
            intersection=len(df),
            bracket_only=len(set(b["t_alert"]) - set(p["t_alert"])),
            proxy_only=len(set(p["t_alert"]) - set(b["t_alert"]))
        ),
        corr=float(corr) if pd.notna(corr) else None,
        bracket_intersection=trade_metrics(pd.DataFrame({"pnl": df["pnl_bracket"]})),
        proxy_intersection=trade_metrics(pd.DataFrame({"pnl": df["pnl_proxy"]})),
        bracket_all=trade_metrics(pd.DataFrame({"pnl": b["pnl"]})),
        proxy_all=trade_metrics(pd.DataFrame({"pnl": p["pnl"]})),
    )
    pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json,"w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracket_csv", default="results/vol_bracket_trades.csv")
    ap.add_argument("--proxy_csv",   default="results/straddle_trades.csv")
    ap.add_argument("--out_json",    default="results/compare_metrics.json")
    args = ap.parse_args()
    main(args.bracket_csv, args.proxy_csv, args.out_json)
