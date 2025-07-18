import glob, pickle

import torch
import numpy as np
import pandas as pd
from backtesting import Strategy
from scripts.models.core_lstm import LSTMAttention, SEQ_LEN, DEVICE, HIDDEN_DIM

class MLStrategy(Strategy):
    """
    Usa la probabilidad de tu LSTM-Attention para abrir/cerrar
    posiciones largas en oro.
    """

    def init(self):
        # 1) Carga scaler y modelo
        scaler_path = sorted(glob.glob("models/scaler_*.pkl"))[-1]
        model_path  = sorted(glob.glob("models/lstm_attn_*.pt"))[-1]
        scaler = pickle.load(open(scaler_path, "rb"))

        model = LSTMAttention(
            input_dim=len(self.data.df.columns) - 1, 
            hidden_dim=HIDDEN_DIM
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE).eval()

        # 2) Prepara el array completo de features
        #    Rellena NaNs para poder generar secuencias desde el primer día
        df_full = pd.DataFrame(self.data.df).fillna(method="ffill").fillna(0)
        X_full = df_full.drop(columns=["Close"]).values
        X_scaled = scaler.transform(X_full)

        # 3) Pre-computa la probabilidad de subida para cada barra
        probs = [None] * len(X_scaled)
        for i in range(SEQ_LEN, len(X_scaled)):
            seq = torch.tensor(
                X_scaled[i-SEQ_LEN : i],
                dtype=torch.float32,
            ).unsqueeze(0).to(DEVICE)  # (1, SEQ_LEN, feat_dim)
            with torch.no_grad():
                prob = torch.sigmoid(model(seq)).item()
            probs[i] = prob

        # Guarda en atributos
        self.probs = probs
        self.model = model  # opcional, por si quieres volver a usarlo

    def next(self):
        idx = len(self.data.Close) - 1  # índice de la barra actual
        prob = self.probs[idx]
        if prob is None:
            return  # aún no hay secuencia completa

        # Señal de trading
        if prob > 0.5 and not self.position.is_long:
            self.buy()
        elif prob < 0.5 and self.position.is_long:
            self.position.close()

