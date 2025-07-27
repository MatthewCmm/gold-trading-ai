#!/usr/bin/env python3
import glob, pandas as pd, numpy as np, joblib
from backtesting import Backtest, Strategy

# Carga datos
feat = sorted(glob.glob("data/processed/features_5m_*.parquet"))[-1]
df_f = pd.read_parquet(feat); df_f.index = pd.to_datetime(df_f.index)
df_p = pd.read_parquet("data/raw/GCF_intraday_5m.parquet"); df_p.index = pd.to_datetime(df_p.index)
df   = df_p.join(df_f, how="inner").dropna()

# Estrategia parametrizada
class MLIntraday(Strategy):
    threshold = 0.0
    def init(self):
        self.model  = joblib.load("models/rf_model_intraday.pkl")
        self.scaler = joblib.load("models/scaler_intraday.pkl")
    def next(self):
        X = self.scaler.transform([[ getattr(self.data, c)[-1] for c in df_f.columns ]])
        pred = self.model.predict(X)[0]
        t    = type(self).threshold
        if pred> t and not self.position.is_long:
            self.position.close(); self.buy()
        elif pred< -t and not self.position.is_short:
            self.position.close(); self.sell()

# Sweep
thresholds = np.linspace(0.0002, 0.0020, 10)
results = []
for t in thresholds:
    MLIntraday.threshold = t
    bt = Backtest(df, MLIntraday, cash=10_000, commission=0.0005, exclusive=True)
    stats = bt.run()
    results.append({
      "threshold": t,
      "Sharpe":    stats["Sharpe Ratio"],
      "Return [%]":stats["Return [%]"],
      "Max DD [%]":stats["Max. Drawdown [%]"],
      "# Trades":  stats["# Trades"],
    })

pd.DataFrame(results).set_index("threshold").to_csv("reports/threshold_sweep_intraday.csv")
print("✅ Sweep intradía guardado en reports/threshold_sweep_intraday.csv")
