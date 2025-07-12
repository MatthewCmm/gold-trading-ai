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


import yfinance as yf
import pandas as pd


def download_gold(*, start=None, end=None, period="2y", interval="1d"):
    """
    Descarga precios de futuros de oro (GC=F) con yfinance.

    - Si se proporciona `start` (y opcionalmente `end`), usa fechas absolutas.
    - De lo contrario, emplea `period` (p. ej. '2y', '6mo').
    """
    if start:
        df = yf.download("GC=F", start=start, end=end, interval=interval)
    else:
        df = yf.download("GC=F", period=period, interval=interval)

    # Aplana posibles columnas MultiIndex: ('Open', 'GC=F') → 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Conserva columnas clave y elimina filas vacías
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    return df
