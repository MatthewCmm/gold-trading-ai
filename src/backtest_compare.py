#!/usr/bin/env python3
# src/backtest_compare.py

"""
Comparación de backtests entre la estrategia de cruce de medias móviles (MA-Crossover)
y una estrategia basada en tu modelo ML entrenado, incluyendo slippage.
"""

import glob
import os
import pandas as pd
import numpy as np
import joblib
from backtesting import Strategy
from bokeh.io import save as bokeh_save, output_file
from src.backtester import run_backtest
from src.strategy import MACrossStrategy

# Parámetros de trading
INITIAL_CAPITAL = 10_000
BASE_COMMISSION = 0.001  # 0.1% por trade
SLIPPAGE_PCT    = 0.0005  # 0.05% slippage on entry/exit

# Umbral para la estrategia ML (por ejemplo 0.08% retorno previsto)
ML_THRESHOLD = 0.0008  # calibrado para maximizar Sharpe

class MLStrategy(Strategy):
    """
    Strategy que utiliza el modelo ML para generar señales LONG/SHORT.
    """
    feature_names: list[str] = []  # será asignado en main()

    def init(self):
        self.model  = joblib.load("models/rf_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")

    def next(self):
        feats = [getattr(self.data, col)[-1] for col in type(self).feature_names]
        X   = np.array(feats).reshape(1, -1)
        Xs  = self.scaler.transform(X)
        pred_ret = self.model.predict(Xs)[0]

        if pred_ret > ML_THRESHOLD:
            if not self.position.is_long:
                self.position.close()
                self.buy()
        elif pred_ret < -ML_THRESHOLD:
            if not self.position.is_short:
                self.position.close()
                self.sell()
        # HOLD: no hace nada


def main():
    # 1) Carga de datos OHLCV
    df_price = pd.read_parquet("data/raw/GCF_prices.parquet")
    df_price.index = pd.to_datetime(df_price.index).tz_localize(None)

    # 2) Carga de features procesados
    feat_files = sorted(glob.glob("data/processed/features_*.parquet"))
    if not feat_files:
        raise FileNotFoundError(
            "No se encontró ningún features_*.parquet en data/processed/"
        )
    latest_feat = feat_files[-1]
    df_feat = pd.read_parquet(latest_feat)
    df_feat.index = pd.to_datetime(df_feat.index).tz_localize(None)

    # Asignar columnas a MLStrategy
    MLStrategy.feature_names = df_feat.columns.tolist()

    # 3) Merge precios + features
    df = df_price.join(df_feat, how="inner").dropna()

    # 4) Backtest MA-Crossover con slippage incluido como comisión adicional
    total_comm = BASE_COMMISSION + SLIPPAGE_PCT
    stats_ma, bt_ma = run_backtest(
        df,
        MACrossStrategy,
        cash=INITIAL_CAPITAL,
        commission=total_comm
    )
    print("=== MA-Crossover Strategy (con slippage) ===")
    print(stats_ma)

    # 5) Backtest ML-Based Strategy
    stats_ml, bt_ml = run_backtest(
        df,
        MLStrategy,
        cash=INITIAL_CAPITAL,
        commission=total_comm
    )
    print("\n=== ML-Based Strategy (con slippage) ===")
    print(stats_ml)

    # 6) Guardar curvas de equity como HTML
    os.makedirs("reports", exist_ok=True)
    plot_ma = bt_ma.plot()
    output_file("reports/ma_crossover_equity.html")
    bokeh_save(plot_ma)
    plot_ml = bt_ml.plot()
    output_file("reports/ml_equity.html")
    bokeh_save(plot_ml)

if __name__ == "__main__":
    main()
