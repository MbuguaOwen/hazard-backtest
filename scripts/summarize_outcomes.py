import pandas as pd, json

def main():
    df = pd.read_csv("results/vol_bracket_trades.csv", parse_dates=["t_alert","t_entry","t_exit"]) if True else pd.DataFrame()
    n = len(df)
    if "exit_reason" in df.columns:
        df["exit_reason"] = df["exit_reason"].astype(str).str.upper().str.strip()
    by_reason = df["exit_reason"].value_counts().to_dict() if "exit_reason" in df.columns else {}
    clean = int((df["exit_reason"] != "SL").sum()) if "exit_reason" in df.columns else 0
    tp = int((df["exit_reason"] == "TP").sum()) if "exit_reason" in df.columns else 0
    sl = int((df["exit_reason"] == "SL").sum()) if "exit_reason" in df.columns else 0
    time_no_sl = int(((df["exit_reason"] == "TIME") & (df["pnl"] > 0)).sum()) if "exit_reason" in df.columns else 0

    # retest depth distributions (winners vs losers)
    depth = df["retest_depth_atr"].dropna() if "retest_depth_atr" in df.columns else pd.Series([], dtype=float)
    win_depth = df.loc[(df.get("pnl", 0) > 0) & df["retest_depth_atr"].notna(), "retest_depth_atr"] if "retest_depth_atr" in df.columns else pd.Series([], dtype=float)
    loss_depth = df.loc[(df.get("pnl", 0) <= 0) & df["retest_depth_atr"].notna(), "retest_depth_atr"] if "retest_depth_atr" in df.columns else pd.Series([], dtype=float)
    # MFE/MAE medians from precomputed ATR-based columns
    mfe_win = df.loc[df.get("pnl", 0) > 0, "mfe_atr"].dropna() if "mfe_atr" in df.columns else pd.Series([], dtype=float)
    mae_win = df.loc[df.get("pnl", 0) > 0, "mae_atr"].dropna() if "mae_atr" in df.columns else pd.Series([], dtype=float)
    mfe_loss = df.loc[df.get("pnl", 0) <= 0, "mfe_atr"].dropna() if "mfe_atr" in df.columns else pd.Series([], dtype=float)
    mae_loss = df.loc[df.get("pnl", 0) <= 0, "mae_atr"].dropna() if "mae_atr" in df.columns else pd.Series([], dtype=float)
    def stats(s: pd.Series):
        if len(s) == 0:
            return dict(n=0)
        q = s.quantile([0.25, 0.5, 0.75]).to_dict()
        return dict(n=int(len(s)), mean=float(s.mean()), q25=float(q.get(0.25, float('nan'))), q50=float(q.get(0.5, float('nan'))), q75=float(q.get(0.75, float('nan'))))
    out = dict(
        n_trades=n,
        clean_rate = (clean / n) if n else 0.0,
        tp_rate    = (tp / n) if n else 0.0,
        time_pos_rate = (time_no_sl / n) if n else 0.0,
        sl_rate = (sl / n) if n else 0.0,
        exit_reason_counts = by_reason,
        mean_pnl = float(df["pnl"].mean()) if n else 0.0,
        retest_depth_atr_stats = dict(win=stats(win_depth), loss=stats(loss_depth)),
        mfe_mae_atr = dict(
            winners=dict(mfe=stats(mfe_win), mae=stats(mae_win)),
            losers=dict(mfe=stats(mfe_loss), mae=stats(mae_loss)),
        ),
    )
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
