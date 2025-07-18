#!/usr/bin/env python3
"""
scripts/predict_signal.py

Carga el modelo entrenado y el scaler, extrae las últimas features procesadas,
estima el retorno esperado y genera una señal de trading simple (LONG / SHORT / HOLD).
"""
import os
import joblib
import pandas as pd

# Parámetro de umbral de señal (ej. 0.1% retorno previsto)
THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD", 0.001))

MODEL_PATH  = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "data/processed/features_latest.parquet"


def load_resources():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def get_latest_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Nos quedamos solo con la última fila
    return df.iloc[[-1]]


def generate_signal(pred_ret: float, threshold: float) -> str:
    if pred_ret > threshold:
        return "LONG"
    elif pred_ret < -threshold:
        return "SHORT"
    else:
        return "HOLD"


def main():
    # 1) Cargar modelo y scaler
    model, scaler = load_resources()

    # 2) Extraer y escalar features
    df_feat = get_latest_features(FEATURES_PATH)
    X = df_feat.values
    X_s = scaler.transform(X)

    # 3) Predicción de retorno
    pred_ret = model.predict(X_s)[0]

    # 4) Generar señal
    signal = generate_signal(pred_ret, THRESHOLD)

    # 5) Output
    print(f"Predicted Return: {pred_ret:.5f}")
    print(f"Signal: {signal}")

if __name__ == "__main__":
    main()
