import json, os
import pandas as pd

def neighbor_avg(rows: pd.DataFrame, idx: int, key_cols) -> float:
    """Average sum_pnl over neighbors differing by at most 1 step in any continuous grid column; exact match on others."""
    target = rows.iloc[idx]
    # Simple: average over params that match exactly (no detailed distance calc available without grid metadata)
    same = rows.copy()
    for k in key_cols:
        same = same[same[k] == target[k]]
    return float(same["sum_pnl"].mean()) if len(same) else float("nan")

def main():
    path = "results/sweeps/summary.json"
    if not os.path.exists(path):
        raise SystemExit("results/sweeps/summary.json not found; run scripts/sweep_vol_bracket.py first")
    data = json.load(open(path, "r"))
    # Flatten
    rows = []
    for rec in data:
        p = rec.get("params", {})
        tr = rec.get("train", {})
        rows.append({**p, **tr})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows in summary.json")
    # Sort by train sum_pnl
    df = df.sort_values("sum_pnl", ascending=False)
    # Compute crude neighbor average on a subset of keys
    keys = [k for k in ["k_breakout","sl_mult","tp_mult","be_frac","trail_mult"] if k in df.columns]
    neigh = []
    for i in range(min(len(df), 50)):
        neigh.append(neighbor_avg(df, i, [k for k in keys if k != "tp_mult"]))
    df_top = df.head(min(len(df), 50)).copy()
    df_top["neighbor_avg_sum_pnl"] = neigh
    os.makedirs("results/sweeps", exist_ok=True)
    outp = "results/sweeps/plateau_report.csv"
    df_top.to_csv(outp, index=False)
    print(f"Wrote {outp}")

if __name__ == "__main__":
    main()

