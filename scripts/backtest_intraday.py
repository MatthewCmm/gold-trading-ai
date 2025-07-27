#!/usr/bin/env python3
# scripts/backtest_compare.py

import glob
import pandas as pd
import numpy as np
import joblib
from backtesting import Backtest, Strategy

from src.strategy import MACrossStrategy

# 1) Load price data
df_price = pd.read_parquet("data/raw/GCF_prices.parquet")
df_price.index = pd.to_datetime(df_price.index).tz_localize(None)

# 2) Load latest features
feat_file = sorted(glob.glob("data/processed/features_*.parquet"))[-1]
df_feat   = pd.read_parquet(feat_file)
df_feat.index = pd.to_datetime(df_feat.index).tz_localize(None)

# 3) Merge price + features
df = df_price.join(df_feat, how="inner").dropna()

# 4) Define ML-based intraday strategy
class MLStrategy(Strategy):
    def init(self):
        # load trained model & scaler
        self.model  = joblib.load("models/rf_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        # preserve feature order
        self.features = df_feat.columns.tolist()

    def next(self):
        # extract latest feature vector
        x = np.array([getattr(self.data, f)[-1] for f in self.features]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        pred = self.model.predict(x_scaled)[0]
        # go long if pred > 0, short if pred < 0
        if pred > 0 and not self.position.is_long:
            self.position.close()
            self.buy()
        elif pred < 0 and not self.position.is_short:
            self.position.close()
            self.sell()

if __name__ == "__main__":
    # 5) Backtest MA-crossover
    bt_ma = Backtest(
        df,
        MACrossStrategy,
        cash=10_000,
        commission=0.001,
        exclusive=True
    )
    stats_ma = bt_ma.run()
    print("=== MA-Crossover Strategy (con slippage) ===")
    print(stats_ma)

    # 6) Backtest ML-based
    bt_ml = Backtest(
        df,
        MLStrategy,
        cash=10_000,
        commission=0.001,
        exclusive=True
    )
    stats_ml = bt_ml.run()
    print("=== ML-Based Strategy (con slippage) ===")
    print(stats_ml)

    # 7) Optional: plot
    bt_ml.plot()

