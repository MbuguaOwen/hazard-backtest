
import pandas as pd
import numpy as np

def trade_metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return dict(n_trades=0, sum_pnl=0.0, mean_pnl=0.0, p_win=0.0, max_dd=0.0, sharpe=0.0)
    pnl = trades["pnl"].fillna(0.0).values
    eq = pnl.cumsum()
    max_dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    sharpe = float(np.mean(pnl)/np.std(pnl)*np.sqrt(365*24*60)) if np.std(pnl) > 1e-12 else 0.0
    return dict(
        n_trades=int(len(trades)),
        sum_pnl=float(np.sum(pnl)),
        mean_pnl=float(np.mean(pnl)),
        p_win=float((pnl > 0).mean()),
        max_dd=max_dd,
        sharpe=sharpe,
    )
