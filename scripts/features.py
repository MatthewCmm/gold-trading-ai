#!/usr/bin/env python3
# scripts/features.py

import os
import datetime as dt

import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

def load_raw():
    # Lee precios de oro
    df_px = pd.read_parquet(os.path.join(RAW_DIR, "GCF_prices.parquet"))
    df_px = df_px[["Close", "High", "Low", "Open", "Volume"]]
    df_px.columns = ["PX_Close", "PX_High", "PX_Low", "PX_Open", "PX_Volume"]
    
    # Lee serie macro
    df_mac = pd.read_parquet(os.path.join(RAW_DIR, "T10YIE.parquet"))
    df_mac.columns = ["BE_10YIE"]  # breakeven 10Y
    
    return df_px, df_mac

def engineer_features(df_px: pd.DataFrame, df_mac: pd.DataFrame) -> pd.DataFrame:
    # Une por índice (fecha)
    df = df_px.join(df_mac, how="left")
    
    # Lagged macro (1 mes atras)
    df["BE_10YIE_L1"] = df["BE_10YIE"].shift(1)
    
    # Rolling averages del precio
    df["MA_Close_20"] = df["PX_Close"].rolling(20).mean()
    df["MA_Close_50"] = df["PX_Close"].rolling(50).mean()
    
    # Rendimientos diarios
    df["Ret_Close"] = df["PX_Close"].pct_change()
    
    # Cualquier otra transformación...
    
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    today = dt.datetime.now().strftime("%Y%m%d")
    
    df_px, df_mac = load_raw()
    df_feat = engineer_features(df_px, df_mac)
    
    out_path = os.path.join(OUT_DIR, f"features_{today}.parquet")
    df_feat.to_parquet(out_path)
    print(f"[OK] Features saved to {out_path}")

if __name__ == "__main__":
    main()
