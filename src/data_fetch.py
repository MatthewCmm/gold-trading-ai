# src/data_fetch.py

import yfinance as yf
import pandas as pd

def download_gold(start: str = None, end: str = None) -> pd.DataFrame:
    """
    Descarga data de GC=F (oro COMEX) y devuelve
    un DataFrame con las columnas Open, High, Low, Close, Volume.
    """
    df = yf.download(
        "GC=F",
        start=start,
        end=end,
        progress=False,
        auto_adjust=False    # Importante: deja tanto Close como Adj Close.
    )
    # Si Yahoo te trae "Adj Close" y no quieres usarla:
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    return df

