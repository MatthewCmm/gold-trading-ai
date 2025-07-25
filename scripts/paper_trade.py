#!/usr/bin/env python3
# scripts/paper_trade.py

import glob
import pandas as pd
import yfinance as yf
import joblib
from pandas.tseries.offsets import BDay

# Paths y parámetros
MODEL_PATH    = "models/rf_model.pkl"
SCALER_PATH   = "models/scaler.pkl"
FEATURES_GLOB = "data/processed/features_*.parquet"
THRESH_CSV    = "reports/threshold_sweep.csv"
START_EQ      = 10_000
TEST_DAYS     = 21  # días hábiles (~1 mes)

def load_models():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, scaler.n_features_in_

def select_best_threshold(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    best = df.loc[df["Sharpe"].idxmax(), "threshold"]
    print(f"Umbral seleccionado (máximo Sharpe): {best:.6f}")
    return best

def simulate_out_of_sample():
    model, scaler, n_feat = load_models()
    threshold = select_best_threshold(THRESH_CSV)

    # Cargar último parquet de features
    feat_file = sorted(glob.glob(FEATURES_GLOB))[-1]
    df_feat   = pd.read_parquet(feat_file)
    df_feat.index = pd.to_datetime(df_feat.index)
    print(f"Simulando sobre features desde {df_feat.index.min().date()} hasta {df_feat.index.max().date()}")
    
    end_date   = df_feat.index.max()
    start_date = end_date - BDay(TEST_DAYS)
    equity     = START_EQ
    records    = []

    for date in pd.date_range(start_date, end_date, freq="B"):
        # 1) Descargar precios t y t+1
        data = yf.download(
            "GC=F",
            start=date,
            end=date + pd.Timedelta(2, "D"),
            interval="1d",
            progress=False
        )["Close"]
        if len(data) < 2:
            continue
        p0, p1 = data.iloc[0], data.iloc[1]

        # 2) Extraer y preparar features
        if date not in df_feat.index:
            continue
        X_full = df_feat.loc[date].values.reshape(1, -1)
        X_used = X_full[:, :n_feat]  # coger sólo las primeras n_feat columnas

        # 3) Generar señal con umbral
        y_pred = model.predict(scaler.transform(X_used))[0]
        if   y_pred >  threshold: sig =  1
        elif y_pred < -threshold: sig = -1
        else:                     sig =  0

        # 4) Calcular PnL y equity
        pnl     = sig * (p1 / p0 - 1)
        equity *= (1 + pnl)

        records.append({
            "date":    date,
            "signal":  sig,
            "price_t": float(p0),
            "price_t1":float(p1),
            "y_pred":  float(y_pred),
            "pnl":     float(pnl),
            "equity":  float(equity)
        })

    df_out = pd.DataFrame(records)
    counts = df_out["signal"].value_counts().reindex([1,0,-1], fill_value=0)
    print(f"Señales → Long: {counts[1]}, Hold: {counts[0]}, Short: {counts[-1]}")
    df_out.to_csv("reports/paper_trade_simulation.csv", index=False)
    print(f"Simulación completa: {start_date.date()} → {end_date.date()}")

if __name__ == "__main__":
    simulate_out_of_sample()
