
import pandas as pd

def ensure_datetime_index(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    if df.index.name != col:
        df = df.copy()
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col).sort_index()
        else:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
    return df

def tie_break_same_bar(entry_up: float, entry_dn: float, open_px: float, close_px: float) -> int:
    # direction-of-close heuristic
    return +1 if close_px >= open_px else -1

class Cooldown:
    def __init__(self, minutes: int):
        self.minutes = minutes
        self.last_ts = None
    def ready(self, ts: pd.Timestamp) -> bool:
        if self.last_ts is None:
            return True
        return ts >= self.last_ts + pd.Timedelta(minutes=self.minutes)
    def mark(self, ts: pd.Timestamp):
        self.last_ts = ts

def donchian_high(series, window:int=60):
    return series.rolling(window, min_periods=max(5, window//3)).max()

def donchian_low(series, window:int=60):
    return series.rolling(window, min_periods=max(5, window//3)).min()
