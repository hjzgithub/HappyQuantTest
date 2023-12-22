import pandas as pd

def ts_rolling_z_score(df: pd.Series, back_window: int):
    return ((df - df.rolling(back_window).mean()) / df.rolling(back_window).std(ddof=1)).values