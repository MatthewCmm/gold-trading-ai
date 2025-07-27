#!/usr/bin/env python3
# scripts/train_model.py

import os
import glob
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error
import joblib

# ──────────────────────────
PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"
# ──────────────────────────


# 1) Locate latest features parquet
feature_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "features_*.parquet")))
if not feature_files:
    sys.exit("❌  No feature files found in data/processed/")
feat_path = feature_files[-1]
print(f"🔍  Using features file → {feat_path}")

# 2) Intraday or daily?
is_intraday = "_5m" in os.path.basename(feat_path)
horizon     = "5m" if is_intraday else "1d"

# 3) Load dataframe
df = pd.read_parquet(feat_path)
df.index = pd.to_datetime(df.index)
print(f"📊  Loaded {df.shape[0]} rows × {df.shape[1]} cols")

# 4) Build target (next-bar %-return)
price_col = "Close" if is_intraday else "PX_Close"
if price_col not in df.columns:
    sys.exit(f"❌  Expected column '{price_col}' not found in features parquet")

df["target"] = df[price_col].pct_change().shift(-1)
df.dropna(subset=["target"], inplace=True)

if df.empty:
    sys.exit("⚠️  No data left after target shift – check the features parquet.")

# 5) Separate X / y  (keep only numeric columns)
feature_cols = [c for c in df.columns if c not in {"target"} and np.issubdtype(df[c].dtype, np.number)]
X = df[feature_cols].values
y = df["target"].values
print(f"⚙️  Training on {len(feature_cols)} numeric features, target {horizon} ahead")

# 6) Scale
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# 7) Time-series CV
tscv = TimeSeriesSplit(n_splits=5)
r2_scores = []
for fold, (tr, te) in enumerate(tscv.split(X_scaled), 1):
    model_cv = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model_cv.fit(X_scaled[tr], y[tr])
    r2 = model_cv.score(X_scaled[te], y[te])
    r2_scores.append(r2)
    print(f"   Fold {fold} ▸ R² = {r2:.4f}")

print("🔗  R² per fold:", np.round(r2_scores, 4).tolist())

# 8) Final fit on full set (reuse last model_cv for speed, optional full re-fit)
model = model_cv
y_hat = model.predict(X_scaled)

mse      = mean_squared_error(y, y_hat)
mae      = mean_absolute_error(y, y_hat)
dir_acc  = np.mean(np.sign(y_hat) == np.sign(y))
print(f"📈  MSE={mse:.6f} | MAE={mae:.6f} | DirAcc={dir_acc:.4f}")

# 9) Feature importances
imp = pd.Series(model.feature_importances_, index=feature_cols)
print("🎯  Top-10 features:")
print(imp.sort_values(ascending=False).head(10))

# 10) Save artefacts
os.makedirs(MODEL_DIR, exist_ok=True)
suffix      = "_intraday" if is_intraday else ""
model_path  = os.path.join(MODEL_DIR, f"rf_model{suffix}.pkl")
scaler_path = os.path.join(MODEL_DIR, f"scaler{suffix}.pkl")
joblib.dump(model,  model_path)
joblib.dump(scaler, scaler_path)
print(f"✅  Saved model → {model_path}")
print(f"✅  Saved scaler → {scaler_path}")
