#!/usr/bin/env python3
# scripts/download_intraday.py

import yfinance as yf
import pandas as pd
from pathlib import Path

# Parámetros
TICKER     = "GC=F"
PERIOD     = "7d"      # últimos 7 días
INTERVAL   = "5m"      # velas de 5 minutos
OUT_PATH   = Path("data/raw/GCF_intraday_5m.parquet")

def main():
    df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
    df = df.tz_localize(None)             # quitar tz para consistencia
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_parquet(OUT_PATH)
    print(f"📈 Datos intradía guardados en {OUT_PATH}")

if __name__ == "__main__":
    main()
