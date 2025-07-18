import yfinance as yf
import pandas as pd
import os
from fredapi import Fred

# Mapea los IDs de FRED a sus descripciones
SERIES = {
    "UNRATE":   "Unemployment Rate",
    # Cuando tengas el código correcto de la PMI, añádelo aquí:
    # "PMNFMFG":  "ISM Manufacturing PMI",
    "VIXCLS":   "CBOE VIX Index",
    "DTWEXBGS": "Trade Weighted Dollar Index (Broad)"
}

def download_gold(*, start=None, end=None, period="2y", interval="1d"):
    """
    Descarga precios de futuros de oro (GC=F) vía yfinance.
    Si se pasa `start`, usa fechas absolutas; de lo contrario, emplea `period`.
    """
    if start:
        df = yf.download("GC=F", start=start, end=end, interval=interval)
    else:
        df = yf.download("GC=F", period=period, interval=interval)

    # Aplana columnas MultiIndex si las hay
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Filtra columnas clave y elimina filas vacías
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def fetch_macro_series():
    """
    Descarga series macroeconómicas desde FRED y las guarda en data/raw/*.parquet.
    """
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        raise RuntimeError("Define FRED_API_KEY en el entorno")
    fred = Fred(api_key=fred_key)
    os.makedirs("data/raw", exist_ok=True)

    for series_id in SERIES:
        try:
            print(f"Descargando {series_id} …")
            s = fred.get_series(series_id)
        except ValueError:
            print(f"⚠️ Serie {series_id} no encontrada, omitiendo.")
            continue

        df = s.to_frame(name=series_id)
        path = f"data/raw/{series_id}.parquet"
        df.to_parquet(path)
        print(f"  → guardado en {path}")


if __name__ == "__main__":
    fetch_macro_series()