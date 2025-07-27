#!/usr/bin/env python3
"""
Walk-forward simple: entrena un RF rolling y calcula equity acumulada.
Si no hay datos suficientes, avisa y sale.
"""

import glob, os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ─────────── parámetros ───────────
TRAIN_YEARS      = 2     # 504 barras
TEST_MONTHS      = 1     # 21 barras
CAPITAL          = 10_000
ML_THRESHOLD     = 0.0

PRICE_FILE       = "data/raw/GCF_prices.parquet"
FEATS_FILE       = sorted(Path("data/processed").glob("features_*.parquet"))[-1]
# ──────────────────────────────────


def sharpe(ser: pd.Series) -> float:
    ser = ser.dropna()
    if ser.empty: return np.nan
    return np.sqrt(252) * ser.mean() / ser.std(ddof=1)


def main():
    if not FEATS_FILE:
        raise FileNotFoundError("No feature files en data/processed")

    # --- load & merge
    px = pd.read_parquet(PRICE_FILE).tz_localize(None)
    if getattr(px.columns, 'nlevels', 1) > 1:
        px.columns = px.columns.get_level_values(0)

    feat = pd.read_parquet(FEATS_FILE).tz_localize(None)
    feat = feat.drop(columns=[c for c in ["Open","High","Low","Close","Volume","target"]
                              if c in feat.columns],
                     errors="ignore").select_dtypes("number")

    df = px.join(feat, how="inner").dropna()
    if len(df) <= (TRAIN_YEARS*252 + TEST_MONTHS*21):
        print("⚠️  Datos insuficientes para walk-forward; se omite.")
        return

    feat_cols: List[str] = feat.columns.tolist()
    train_size = TRAIN_YEARS * 252
    test_size  = TEST_MONTHS * 21
    n_splits   = (len(df) - train_size) // test_size
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    equity_curves, metrics = [], []
    for tr, te in tscv.split(df):
        dtr, dte = df.iloc[tr], df.iloc[te]

        # ----- modelo
        Xtr = dtr[feat_cols];  ytr = dtr["Close"].pct_change().shift(-1).dropna()
        Xtr = Xtr.iloc[:-1]    # alinear
        sc  = StandardScaler().fit(Xtr)
        rf  = RandomForestRegressor(random_state=42).fit(sc.transform(Xtr), ytr)

        # ----- señal en test
        Xte = sc.transform(dte[feat_cols])
        pred= rf.predict(Xte)
        sig = np.where(pred > ML_THRESHOLD, 1,
              np.where(pred < -ML_THRESHOLD, -1, 0))
        ret = sig * dte["Close"].pct_change().shift(-1).to_numpy()

        eq  = (1 + pd.Series(ret, index=dte.index).fillna(0)).cumprod() * CAPITAL
        equity_curves.append(eq)

        metrics.append({
            "train_start": dtr.index[0], "train_end": dtr.index[-1],
            "test_start" : dte.index[0], "test_end" : dte.index[-1],
            "Sharpe"     : sharpe(pd.Series(ret)), 
            "Return"     : eq.iloc[-1]/CAPITAL - 1,
            "MaxDD"      : (eq/eq.cummax() - 1).min(),
            "Trades"     : np.count_nonzero(sig)
        })

    # --- exportar resultados
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv("reports/walk_forward_metrics.csv", index=False)

    equity_all = pd.concat(equity_curves).sort_index()
    plt.figure(figsize=(10,4))
    equity_all.plot(); plt.title("Walk-Forward Equity"); plt.ylabel("$")
    plt.tight_layout(); plt.savefig("reports/walk_forward_equity.png")
    print("✅  Walk-forward completado; resultados en reports/")


if __name__ == "__main__":
    main()
