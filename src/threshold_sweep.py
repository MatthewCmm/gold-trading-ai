#!/usr/bin/env python3
"""
Barrido de umbrales para la RF.
Genera reports/threshold_sweep.csv con métricas por umbral.
"""

import glob, os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from backtesting import Strategy

from src.backtester import run_backtest

# ───────── parámetros ─────────
RAW_PRICE_FILE = "data/raw/GCF_prices.parquet"
PROCESSED_DIR  = "data/processed"
MODEL_DIR      = "models"

THRESHOLDS     = np.linspace(0.0002, 0.0020, 10)
CAPITAL        = 10_000
COMMISSION     = 0.001
# ──────────────────────────────


def load_price() -> pd.DataFrame:
    df = pd.read_parquet(RAW_PRICE_FILE)
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)
    return df.tz_localize(None).add_prefix("PX_")

def load_features() -> tuple[pd.DataFrame, bool]:
    p = sorted(Path(PROCESSED_DIR).glob("features_*.parquet"))[-1]
    intra = "_5m" in p.stem
    df    = pd.read_parquet(p).tz_localize(None)
    df    = df.drop(columns="target", errors="ignore")
    return df, intra

def select_model(intra: bool):
    s = "_intraday" if intra else ""
    return (joblib.load(Path(MODEL_DIR)/f"rf_model{s}.pkl"),
            joblib.load(Path(MODEL_DIR)/f"scaler{s}.pkl"))


class MLStrategy(Strategy):
    feature_names: list[str] = []
    threshold: float = 0.001
    model = scaler = None

    def init(self):
        self.m, self.sc = type(self).model, type(self).scaler

    def next(self):
        feats = [getattr(self.data, c)[-1] for c in type(self).feature_names]
        xs    = self.sc.transform([feats[: self.sc.n_features_in_]])
        pred  = self.m.predict(xs)[0]
        t     = type(self).threshold

        if pred >  t and not self.position.is_long:
            self.position.close(); self.buy()
        elif pred < -t and not self.position.is_short:
            self.position.close(); self.sell()


def main() -> None:
    price = load_price()
    feats, intra = load_features()
    model, scl   = select_model(intra)

    MLStrategy.model, MLStrategy.scaler = model, scl
    MLStrategy.feature_names = feats.columns.tolist()

    df_all = price.join(feats, how="inner").dropna()

    # restaurar OHLCV sin prefijo para Backtesting
    for c in ["Open","High","Low","Close","Volume"]:
        px = f"PX_{c}"
        if px in df_all.columns:
            df_all[c] = df_all[px]; df_all.drop(columns=px, inplace=True)

    rows = []
    for thr in THRESHOLDS:
        MLStrategy.threshold = thr
        stats, _ = run_backtest(df_all, MLStrategy, cash=CAPITAL, commission=COMMISSION)
        rows.append({
            "threshold": thr,
            "Sharpe":    stats["Sharpe Ratio"],
            "Return [%]":stats["Return [%]"],
            "Max DD [%]":stats["Max. Drawdown [%]"],
            "# Trades":  stats["# Trades"],
        })

    out = pd.DataFrame(rows).set_index("threshold").round(6)
    print(out)

    Path("reports").mkdir(exist_ok=True)
    out.to_csv("reports/threshold_sweep.csv")
    print("✅  Guardado reports/threshold_sweep.csv")


if __name__ == "__main__":
    main()
