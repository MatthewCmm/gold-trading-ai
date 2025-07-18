#!/usr/bin/env python3
# scripts/train_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# 1) Cargar features procesados
try:
    df = pd.read_parquet("data/processed/features_20250717.parquet")
except Exception as e:
    print(f"Error al leer features: {e}")
    exit(1)

# 2) Definir target y eliminar sólo filas sin target
df["target"] = df["PX_Close"].pct_change().shift(-1)
df = df.dropna(subset=["target"])
if df.empty:
    print("No hay datos suficientes para entrenar. Verifica el archivo de features.")
    exit(1)

# 3) Separar X e y
feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values
y = df["target"].values

# 4) Escalado de features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5) Validación cruzada temporal
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(X_scaled):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    scores.append(model.score(X_te, y_te))

print("R² por fold:", scores)

# 6) Métricas en todo el dataset
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
directional_acc = np.mean(np.sign(y_pred) == np.sign(y))
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, Directional Accuracy: {directional_acc:.4f}")

# 7) Importancia de features
importances = pd.Series(model.feature_importances_, index=feature_cols)
print("Top 10 features:")
print(importances.sort_values(ascending=False).head(10))

# 8) Guardar modelo y scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Modelo y scaler guardados en carpeta 'models/'")

