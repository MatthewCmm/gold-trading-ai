#!/usr/bin/env python3
# scripts/models/core_lstm.py

import os
import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1) Hiperparámetros
SEQ_LEN    = 30      # secuencia de 30 días
BATCH_SIZE = 64
HIDDEN_DIM = 32
EPOCHS     = 20
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR  = "models"
FEATURES_FILE = sorted(os.listdir("data/processed"))[-1]
FEATURES_PATH = os.path.join("data/processed", FEATURES_FILE)


class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.out  = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        h, _ = self.lstm(x)              # (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(h), dim=1)  # (batch, seq_len, 1)
        context = (weights * h).sum(dim=1)            # (batch, hidden_dim)
        return self.out(context)


def load_data(seq_len=SEQ_LEN):
    df = pd.read_parquet(FEATURES_PATH).dropna()
    # Objetivo: predecir dirección del precio al día siguiente
    X = df.drop(columns=["PX_Close"]).values
    y = (df["PX_Close"].shift(-1) > df["PX_Close"]).astype(int).values
    X, y = X[:-1], y[:-1]

    # Estandarizar features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Crear secuencias
    seqs, labels = [], []
    for i in range(len(X) - seq_len):
        seqs.append(X[i : i + seq_len])
        labels.append(y[i + seq_len])
    X_tensor = torch.tensor(seqs, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    return dl, scaler


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    dataloader, scaler = load_data()

    model = LSTMAttention(input_dim=dataloader.dataset.tensors[0].shape[2], hidden_dim=HIDDEN_DIM)
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {total_loss/len(dataloader):.4f}")

    # Guardar modelo y scaler
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_attn_{ts}.pt")
    pd.to_pickle(scaler, f"{MODEL_DIR}/scaler_{ts}.pkl")
    print(f"[OK] Model and scaler saved to {MODEL_DIR}/")

if __name__ == "__main__":
    train()

