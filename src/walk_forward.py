import glob
import os
from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from backtesting import Backtest, Strategy
from src.strategy import MACrossStrategy
from src.backtester import run_backtest

try:
    from src.backtest_compare import MLStrategy  # optional baseline strategy
except Exception:  # pragma: no cover - module might not exist
    MLStrategy = None


# ----------------------------- Parameters ----------------------------------
TRAIN_YEARS   = 2   # training window in years
TEST_MONTHS   = 1   # holdout length in months
INITIAL_CAPITAL = 10_000
COMMISSION      = 0.001
SLIPPAGE_PCT    = 0.0
ML_THRESHOLD    = 0.0  # threshold for taking long/short positions

PRICE_PATH     = "data/raw/GCF_prices.parquet"
FEATURES_PATHS = sorted(glob.glob("data/processed/features_*.parquet"))


# ----------------------------- Helpers -------------------------------------
def sharpe_ratio(returns: pd.Series) -> float:
    """Compute annualized Sharpe ratio."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    excess = returns - returns.mean()
    denom = returns.std(ddof=1)
    if denom == 0:
        return np.nan
    return np.sqrt(252) * excess.mean() / denom


# ----------------------------- Main logic ----------------------------------
def main() -> None:
    if not FEATURES_PATHS:
        raise FileNotFoundError("No feature files found in data/processed")

    # 1) Load price and feature data
    df_price = pd.read_parquet(PRICE_PATH)
    df_price.index = pd.to_datetime(df_price.index)

    df_feat = pd.read_parquet(FEATURES_PATHS[-1])
    df_feat.index = pd.to_datetime(df_feat.index)

    df_all = df_price.join(df_feat, how="inner").dropna()
    feat_cols: List[str] = df_feat.columns.tolist()

    train_size = TRAIN_YEARS * 252
    test_size  = TEST_MONTHS * 21

    # Garantizar al menos 2 splits para TimeSeriesSplit
    computed_splits = (len(df_all) - train_size) // test_size
    n_splits = max(2, computed_splits)

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    metrics = []
    equity_curves = []

    for train_index, test_index in tscv.split(df_all):
        df_train = df_all.iloc[train_index].copy()
        df_test  = df_all.iloc[test_index].copy()

        # --- train model
        X_train  = df_train[feat_cols]
        y_train  = df_train["PX_Close"].pct_change().shift(-1).dropna()
        X_train  = X_train.iloc[:-1]

        scaler   = StandardScaler().fit(X_train)
        Xs_train = scaler.transform(X_train)
        model    = RandomForestRegressor(random_state=42)
        model.fit(Xs_train, y_train)

        # --- predict on test
        X_test  = df_test[feat_cols]
        Xs_test = scaler.transform(X_test)
        y_pred  = model.predict(Xs_test)

        sig        = np.where(y_pred > ML_THRESHOLD, 1,
                              np.where(y_pred < -ML_THRESHOLD, -1, 0))
        ret        = sig * df_test["PX_Close"].pct_change().shift(-1).to_numpy()
        ret_series = pd.Series(ret, index=df_test.index)

        equity = (1 + ret_series.fillna(0)).cumprod() * INITIAL_CAPITAL
        equity_curves.append(equity)

        sharpe       = sharpe_ratio(ret_series)
        total_return = equity.iloc[-1] / INITIAL_CAPITAL - 1
        max_dd       = (equity / equity.cummax() - 1).min()
        trades       = np.count_nonzero(sig)

        metrics.append({
            "train_start": df_train.index[0],
            "train_end"  : df_train.index[-1],
            "test_start" : df_test.index[0],
            "test_end"   : df_test.index[-1],
            "Sharpe"     : sharpe,
            "Return"     : total_return,
            "MaxDD"      : max_dd,
            "Trades"     : trades,
        })

    # Export metrics
    df_metrics = pd.DataFrame(metrics)
    os.makedirs("reports", exist_ok=True)
    df_metrics.to_csv("reports/walk_forward_metrics.csv", index=False)
    print(df_metrics.describe())

    # Curva de equity agregada
    equity_all = pd.concat(equity_curves).sort_index()
    plt.figure(figsize=(10, 4))
    equity_all.plot()
    plt.title("Walk-forward Equity Curve")
    plt.ylabel("Equity ($)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("reports/walk_forward_equity.png")


if __name__ == "__main__":
    main()


