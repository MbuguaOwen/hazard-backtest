
import pandas as pd
import numpy as np

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, window: int = 60) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/window, adjust=False).mean()
