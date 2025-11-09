import pandas as pd
from pathlib import Path

def main():
    trades_path = Path("results/vol_bracket_trades.csv")
    exc_path = Path("results/excursions/excursions.csv")
    if not trades_path.exists() or not exc_path.exists():
        raise SystemExit("Missing inputs: ensure trades and excursions are generated first.")

    tr = pd.read_csv(trades_path)
    ex = pd.read_csv(exc_path)

    tr["t_alert"] = pd.to_datetime(tr["t_alert"], utc=True).dt.floor("min")
    ex["t_alert"] = pd.to_datetime(ex["t_alert"], utc=True).dt.floor("min")

    df = pd.merge(tr, ex, on="t_alert", how="inner")

    # Bias correctness by horizon close direction
    # net_move_atr: (close_at_horizon - P0) / ATR_ref
    df["bias_correct_net"] = (df["dir"] * df["net_move_atr"]) > 0

    # Bias correctness by max excursion side
    df["max_dir"] = (df["up_atr"] >= df["dn_atr"]).astype(int)  # 1 if up side bigger
    df["bias_is_up"] = (df["dir"] == 1).astype(int)
    df["bias_correct_max"] = df["max_dir"] == df["bias_is_up"]
    df["max_margin_atr"] = (df["up_atr"] - df["dn_atr"]).abs()

    out = {
        "n": int(len(df)),
        "bias_correct_net_rate": float(df["bias_correct_net"].mean()) if len(df) else 0.0,
        "bias_correct_max_rate": float(df["bias_correct_max"].mean()) if len(df) else 0.0,
        "max_margin_q50": float(df["max_margin_atr"].quantile(0.50)) if len(df) else 0.0,
        "max_margin_q75": float(df["max_margin_atr"].quantile(0.75)) if len(df) else 0.0,
    }
    print(out)

if __name__ == "__main__":
    main()

