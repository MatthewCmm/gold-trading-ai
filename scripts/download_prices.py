#!/usr/bin/env python3
import os
from src.data_fetch import download_gold

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# Descarga 2 años de precios diarios de futuros de oro
df_price = download_gold(period="2y", interval="1d")

# Guarda en el path que usa load_raw()
out_path = os.path.join(RAW_DIR, "GCF_prices.parquet")
df_price.to_parquet(out_path)
print(f"▶️  Saved price data to {out_path}")

