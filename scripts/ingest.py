import os
import datetime as dt
import requests
import pandas as pd
import yfinance as yf

# Reemplaza Fred() por tu propia lógica REST
API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    raise RuntimeError("FRED_API_KEY no está definida en el entorno")

def fetch_prices(ticker: str = "GC=F", start: str = "2010-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True)
    path = f"data/raw/{ticker.replace('=','')}_prices.parquet"
    df.to_parquet(path)
    print(f"[OK] Prices saved to {path}")
    return df

def fetch_macro(series: str = "T10YIE"):
    """Descarga la serie macro vía REST JSON y la guarda en Parquet."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series,
        "api_key":   API_KEY,
        "file_type": "json",
    }
    print(f"Fetching {series} from FRED…")
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("observations", [])

    # Convertir a DataFrame y sanitizar (convertir "." a NaN)
    df = pd.DataFrame(data)
    # coercer errores convierte "." en NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Indexar por fecha y renombrar columna
    df = df.set_index("date")[["value"]].rename(columns={"value": series})

    path = f"data/raw/{series}.parquet"
    df.to_parquet(path)
    print(f"[OK] Macro series saved to {path}")
    return df

def main():
    os.makedirs("data/raw", exist_ok=True)
    print("Start ingestion:", dt.datetime.now())
 # 1) Precio del oro
    fetch_prices()

    # 2) Series macro adicionales desde FRED
    fetch_macro("T10YIE")
    fetch_macro("CPIAUCSL")  # Inflación (CPI yoy)
    fetch_macro("DEXUSAL")   # Índice dólar
    fetch_macro("DGS10")     # Rendimiento 10 años
    fetch_macro("DGS2")      # Rendimiento 2 años
    fetch_macro("GDPC1")     # PIB real trimestral
    print("Finished ingestion:", dt.datetime.now())

if __name__ == "__main__":
    main()

