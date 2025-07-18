#!/usr/bin/env python3
# scripts/evaluate_model.py

import os
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from scripts.models.core_lstm import LSTMAttention, SEQ_LEN, DEVICE, HIDDEN_DIM

MODEL_DIR = "models"
FEATURES_PATH = sorted(glob.glob("data/processed/features_*.parquet"))[-1]

def load_test_data(scaler_path, seq_len=SEQ_LEN):
    # Carga dataframe de features ya sin NaNs
    df = pd.read_parquet(FEATURES_PATH).dropna()
    X = df.drop(columns=["PX_Close"]).values
    y = (df["PX_Close"].shift(-1) > df["PX_Close"]).astype(int).values
    X, y = X[:-1], y[:-1]

    # Carga scaler y estandariza
    scaler: StandardScaler = pd.read_pickle(scaler_path)
    X_scaled = scaler.transform(X)

    # Construye secuencias
    seqs, labels = [], []
    for i in range(len(X_scaled) - seq_len):
        seqs.append(X_scaled[i : i + seq_len])
        labels.append(y[i + seq_len])
    X_test = torch.tensor(np.array(seqs), dtype=torch.float32).to(DEVICE)
    y_test = np.array(labels)
    return X_test, y_test

def main():
    # ——— Selecciona modelo y scaler más recientes ———
    models = sorted(glob.glob(f"{MODEL_DIR}/lstm_attn_*.pt"))
    scalers = sorted(glob.glob(f"{MODEL_DIR}/scaler_*.pkl"))
    model_file, scaler_file = models[-1], scalers[-1]
    print("Usando modelo:", model_file)
    print("Usando scaler:", scaler_file)

    # ——— Carga datos de test ———
    X_test, y_test = load_test_data(scaler_file)

    # ——— Inicializa modelo y carga pesos ———
    model = LSTMAttention(input_dim=X_test.shape[2], hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.to(DEVICE).eval()

    # ——— Predicciones ———
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    # ——— Métricas ———
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC : {auc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()
