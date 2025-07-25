#!/usr/bin/env python3
# scripts/train_model.py

import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# 1) Localizar el último archivo de features
feature_files = sorted(glob.glob("data/processed/features_*.parquet"))
if not feature_files:
    print("No se encontró ningún archivo de features en data/processed/")
    exit(1)
latest_feat_path = feature_files[-1]

# 2) Cargar features procesados
try:
    df = pd.read_parquet(latest_feat_path)
    df.index = pd.to_datetime(df.index)
    print(f"Cargado features desde {latest_feat_path} con {df.shape[1]} columnas")
except Exception as e:
    print(f"Error al leer features: {e}")
    exit(1)

# 3) Definir target y eliminar filas sin target
df["target"] = df["PX_Close"].pct_change().shift(-1)
df = df.dropna(subset=["target"])
if df.empty:
    print("No hay datos suficientes para entrenar. Verifica el archivo de features.")
    exit(1)

# 4) Separar X e y usando todas las columnas menos 'target'
feature_cols = [c for c in df.columns if c != "target"]
print(f"Entrenando con {len(feature_cols)} features")
X = df[feature_cols].values
y = df["target"].values

# 5) Escalado de features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6) Validación cruzada temporal
tscv = TimeSeriesSplit(n_splits=5)
r2_scores = []
for i, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    model_cv = RandomForestRegressor(n_estimators=100, random_state=42)
    model_cv.fit(X_tr, y_tr)
    r2 = model_cv.score(X_te, y_te)
    r2_scores.append(r2)
    print(f"  Fold {i} R²: {r2:.4f}")
print("R² por fold:", r2_scores)

# 7) Métricas en todo el dataset con el último modelo entrenado
model = model_cv
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
directional_acc = np.mean(np.sign(y_pred) == np.sign(y))
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, Directional Accuracy: {directional_acc:.4f}")

# 8) Importancia de features
importances = pd.Series(model.feature_importances_, index=feature_cols)
print("Top 10 features:")
print(importances.sort_values(ascending=False).head(10))

# 9) Guardar modelo y scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print(f"Modelo y scaler guardados en 'models/' (features: {len(feature_cols)})")

