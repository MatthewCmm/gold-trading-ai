#!/usr/bin/env python3
"""
Compara:
  • Cruce de medias móviles (baseline)
  • Estrategia ML Random-Forest
Incluyendo comisión + slippage.
"""

import glob, os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from backtesting import Strategy
from bokeh.io import output_file, save as bokeh_save

from src.backtester import run_backtest
from src.strategy   import MACrossStrategy

# ───────── parámetros ─────────
INITIAL_CAPITAL = 10_000
BASE_COMMISSION = 0.001
SLIPPAGE_PCT    = 0.0005
ML_THRESHOLD    = 0.0008

RAW_PRICE_FILE  = "data/raw/GCF_prices.parquet"
PROCESSED_DIR   = "data/processed"
MODEL_DIR       = "models"
REPORT_DIR      = "reports"
# ──────────────────────────────


class MLStrategy(Strategy):
    """
    Estrategia long/short basada en el Random-Forest.
    """
    feature_names: list[str] = []
    model  = None
    scaler = None

    def init(self):
        self.m, self.sc = type(self).model, type(self).scaler   # atajos

    def next(self):
        feats = [getattr(self.data, c)[-1] for c in type(self).feature_names]
        xs    = self.sc.transform([feats[: self.sc.n_features_in_]])
        pred  = self.m.predict(xs)[0]

        if pred >  ML_THRESHOLD and not self.position.is_long:
            self.position.close(); self.buy()
        elif pred < -ML_THRESHOLD and not self.position.is_short:
            self.position.close(); self.sell()
        # si |pred| ≤ umbral ⇒ hold


# ---------- helpers ----------
def load_price_df() -> pd.DataFrame:
    df = pd.read_parquet(RAW_PRICE_FILE)
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # mantén prefijo para distinguir, luego renombraremos
    return df.add_prefix("PX_")

def load_features_df() -> tuple[pd.DataFrame, bool]:
    path = sorted(Path(PROCESSED_DIR).glob("features_*.parquet"))[-1]
    is_intra = "_5m" in path.stem
    df = pd.read_parquet(path).tz_localize(None)
    # sólo quitamos la columna 'target' si existe
    df = df.drop(columns="target", errors="ignore")
    return df, is_intra

def select_model(is_intraday: bool):
    suf = "_intraday" if is_intraday else ""
    mdl  = joblib.load(Path(MODEL_DIR) / f"rf_model{suf}.pkl")
    scl  = joblib.load(Path(MODEL_DIR) / f"scaler{suf}.pkl")
    return mdl, scl
# ------------------------------


def main() -> None:
    # 1) datos
    df_price        = load_price_df()
    df_feat, intra  = load_features_df()

    # 2) merge y limpieza
    df_all = df_price.join(df_feat, how="inner").dropna()

    # reconstruir columnas OHLCV *sin* prefijo (Backtesting las necesita)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        px_col = f"PX_{col}"
        if px_col in df_all.columns:
            df_all[col] = df_all[px_col]
            df_all.drop(columns=px_col, inplace=True)

    # 3) modelo ML + features
    model, scaler            = select_model(intra)
    MLStrategy.model         = model
    MLStrategy.scaler        = scaler
    MLStrategy.feature_names = df_feat.columns.tolist()

    comm = BASE_COMMISSION + SLIPPAGE_PCT

    # 4) backtests
    stats_ma, bt_ma = run_backtest(df_all, MACrossStrategy,
                                   cash=INITIAL_CAPITAL, commission=comm)
    print("=== MA-Crossover (con slippage) ===")
    print(stats_ma)

    stats_ml, bt_ml = run_backtest(df_all, MLStrategy,
                                   cash=INITIAL_CAPITAL, commission=comm)
    print("\n=== ML-Strategy (con slippage) ===")
    print(stats_ml)

    # 5) gráficas
    Path(REPORT_DIR).mkdir(exist_ok=True)
    output_file(Path(REPORT_DIR) / "ma_crossover_equity.html"); bokeh_save(bt_ma.plot())
    output_file(Path(REPORT_DIR) / "ml_equity.html");          bokeh_save(bt_ml.plot())
    print("✅  HTML guardado en reports/")

if __name__ == "__main__":
    main()
