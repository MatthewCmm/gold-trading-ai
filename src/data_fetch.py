import yfinance as yf
import pandas as pd


def download_gold(*, start=None, end=None, period="2y", interval="1d"):
    """
    Descarga precios de futuros de oro (GC=F) vía yfinance.
    Si se pasa `start`, usa fechas absolutas; de lo contrario, emplea `period`.
    """
    if start:
        df = yf.download("GC=F", start=start, end=end, interval=interval)
    else:
        df = yf.download("GC=F", period=period, interval=interval)

    # ---- Aplana columnas MultiIndex ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Filtra columnas y elimina filas vacías
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    return df
