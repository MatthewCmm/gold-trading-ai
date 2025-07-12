# src/data_fetch.py

import yfinance as yf
import pandas as pd

def download_gold(period="2y", interval="1d"):
    df = yf.download("GC=F", period=period, interval=interval)

    # --- NUEVO BLOQUE ---
    # Si las columnas vienen en MultiIndex, quita el segundo nivel
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # ['Open', 'High', ...]
    # Opcional: reordena o filtra
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    # ---------------------

    return df


