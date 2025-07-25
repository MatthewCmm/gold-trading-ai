#!/usr/bin/env python3
# src/threshold_sweep.py

import os
import glob
import pandas as pd
import numpy as np
import joblib
from backtesting import Strategy
from src.backtester import run_backtest
from src.strategy import MACrossStrategy

# Lista de umbrales a probar
thresholds = np.linspace(0.0002, 0.0020, 10)  # de 0.02% a 0.20%

class MLStrategy(Strategy):
    feature_names: list[str] = []
    threshold: float = 0.001

    def init(self):
        self.model  = joblib.load("models/rf_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")

    def next(self):
        feats = [getattr(self.data, c)[-1] for c in type(self).feature_names]
        Xs    = self.scaler.transform(np.array(feats).reshape(1, -1))
        pred  = self.model.predict(Xs)[0]
        t     = type(self).threshold

        if pred >  t and not self.position.is_long:
            self.position.close(); self.buy()
        elif pred < -t and not self.position.is_short:
            self.position.close(); self.sell()

def load_data():
    df_price = pd.read_parquet("data/raw/GCF_prices.parquet")
    df_price.index = pd.to_datetime(df_price.index).tz_localize(None)

    feats = sorted(glob.glob("data/processed/features_*.parquet"))[-1]
    df_feat = pd.read_parquet(feats)
    df_feat.index = pd.to_datetime(df_feat.index).tz_localize(None)

    MLStrategy.feature_names = df_feat.columns.tolist()
    return df_price.join(df_feat, how="inner").dropna()

def main():
    df     = load_data()
    results = []

    for t in thresholds:
        MLStrategy.threshold = t
        stats, _ = run_backtest(df, MLStrategy, cash=10_000, commission=0.001)
        results.append({
            "threshold": t,
            "Sharpe":    stats["Sharpe Ratio"],
            "Return [%]":stats["Return [%]"],
            "Max DD [%]":stats["Max. Drawdown [%]"],
            "# Trades":  stats["# Trades"],
        })

    df_res = pd.DataFrame(results).set_index("threshold")
    print(df_res)

    # ——— guarda la sweep para el paper-trade ———
    os.makedirs("reports", exist_ok=True)
    df_res.to_csv("reports/threshold_sweep.csv", index=True)
    print("\n✅ Guardado reports/threshold_sweep.csv")

if __name__ == "__main__":
    main()

