#!/usr/bin/env python3
"""
Paper-trading de 1-mes (días hábiles) usando el RF entrenado
y el umbral óptimo guardado en reports/threshold_sweep.csv
"""

import glob, os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

# ───────── paths & const ─────────
PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"
THRESH_CSV    = "reports/threshold_sweep.csv"

START_EQ   = 10_000
TEST_DAYS  = 21          # ~1 mes hábil
TICKER     = "GC=F"
# ─────────────────────────────────

def latest_features() -> tuple[pd.DataFrame, bool]:
    p = sorted(Path(PROCESSED_DIR).glob("features_*.parquet"))[-1]
    intra = "_5m" in p.stem
    df = pd.read_parquet(p).tz_localize(None)
    df = df.drop(columns="target", errors="ignore")
    return df, intra

def load_model(intra: bool):
    suf = "_intraday" if intra else ""
    model  = joblib.load(Path(MODEL_DIR)/f"rf_model{suf}.pkl")
    scaler = joblib.load(Path(MODEL_DIR)/f"scaler{suf}.pkl")
    return model, scaler

def best_threshold(path: str) -> float:
    df = pd.read_csv(path)
    return float(df.loc[df["Sharpe"].idxmax(), "threshold"])

def main() -> None:
    df_feat, intra = latest_features()
    model, scaler  = load_model(intra)
    thresh         = best_threshold(THRESH_CSV)

    n_feat  = scaler.n_features_in_
    end_d   = df_feat.index.max()
    start_d = end_d - BDay(TEST_DAYS)
    equity  = START_EQ
    recs    = []

    for d in pd.date_range(start_d, end_d, freq="B"):
        # ▸ precios t y t+1
        px = yf.download(TICKER, start=d, end=d + BDay(2),
                         interval="1d", progress=False)["Close"]
        if len(px) < 2 or d not in df_feat.index:
            continue
        p0, p1 = px.iloc[0], px.iloc[1]

        # ▸ features alineados y recortados
        feats = df_feat.loc[d].values.astype(float)
        xs    = scaler.transform([feats[:n_feat]])
        yhat  = model.predict(xs)[0]

        sig = 1 if yhat > thresh else -1 if yhat < -thresh else 0
        pnl = sig * (p1 / p0 - 1)
        equity *= (1 + pnl)

        recs.append({
            "date":    d.date(),
            "signal":  sig,
            "price_t": p0,
            "price_t1":p1,
            "y_pred":  yhat,
            "pnl":     pnl,
            "equity":  equity
        })

    df_out = pd.DataFrame(recs)
    long, hold, short = (df_out["signal"] == 1).sum(), (df_out["signal"] == 0).sum(), (df_out["signal"] == -1).sum()
    print(f"Señales → Long: {long}, Hold: {hold}, Short: {short}")
    Path("reports").mkdir(exist_ok=True)
    df_out.to_csv("reports/paper_trade_simulation.csv", index=False)
    print(f"✅  Simulación guardada en reports/paper_trade_simulation.csv")

if __name__ == "__main__":
    main()
