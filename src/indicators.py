from typing import Union
import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series: Union[pd.Series, pd.DataFrame],
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    # Si nos pasaron un DataFrame de 1 columna, sacamos esa serie:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line   = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist        = macd_line - signal_line

    return pd.DataFrame({
        "MACD":   macd_line,
        "Signal": signal_line,
        "Hist":   hist,
    })

def compute_bollinger(
    series: Union[pd.Series, pd.DataFrame],
    window: int = 20,
    num_std: int = 2
) -> pd.DataFrame:
    # Si te pasan DataFrame de 1 columna, sácale la Serie
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    ma    = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = ma + num_std*std
    lower = ma - num_std*std

    return pd.DataFrame({
        "BB_MA":    ma,
        "BB_Upper": upper,
        "BB_Lower": lower
    }, index=series.index)

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    df["ATR"] = atr
    return df

def moving_averages(df: pd.DataFrame, windows=(20, 50)) -> pd.DataFrame:
    for w in windows:
        df[f"MA{w}"] = df["Close"].rolling(w).mean()
    return df
    
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI14"] = compute_rsi(df["Close"])
    macd_df = compute_macd(df["Close"])
    bb_df = compute_bollinger(df["Close"])
    df = pd.concat([df, macd_df, bb_df], axis=1)
    return df


