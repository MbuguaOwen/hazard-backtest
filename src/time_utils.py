
import pandas as pd

def force_ts_cols(df: pd.DataFrame, time_candidates=None, out_col="ts") -> pd.DataFrame:
    if time_candidates is None:
        time_candidates = ["ts","t_alert","timestamp","open_time","date","datetime","time","open_time_ms"]
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in time_candidates if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"No recognizable time column in {list(df.columns)}")
    s = df[tcol]
    if s.dtype.kind in "iuf":
        ser = pd.to_numeric(s, errors="coerce")
        med = float(ser.dropna().median())
        unit = "ms" if med > 1e11 else "s"
        ts = pd.to_datetime(ser, unit=unit, utc=True, errors="coerce")
    else:
        try:
            ts = pd.to_datetime(s, utc=True, format="mixed")
        except TypeError:
            ts = pd.to_datetime(s.astype(str), utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps; first few bad: {s[ts.isna()].head().tolist()}")
    df[out_col] = ts.dt.floor("min")
    if tcol != out_col:
        df = df.drop(columns=[tcol])
    return df
