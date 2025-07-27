#!/usr/bin/env python3
# scripts/features_intraday.py

import os
import pandas as pd
import yfinance as yf
from ta.momentum  import RSIIndicator
from ta.trend     import MACD
from ta.volatility import AverageTrueRange

# ── Parámetros generales ──────────────────────────────────────────────
SYMBOL        = "GC=F"
INTERVAL      = "5m"
LOOKBACK_DAYS = 30
OUTPUT_DIR    = "data/processed"

WINDOWS = dict(
    rsi       = 14,
    macd_slow = 26,
    macd_fast = 12,
    macd_sign = 9,
    atr       = 14,
)

# ── 1) descarga de velas intradía ─────────────────────────────────────
def download_intraday(symbol: str, days: int, interval: str) -> pd.DataFrame:
    """
    Descarga las velas intradía de *symbol* y devuelve un DataFrame OHLCV.
    Aplana posibles MultiIndex de columnas que yfinance devuelve.
    """
    df = yf.download(
        symbol,
        period   = f"{days}d",
        interval = interval,
        progress = False
    ).loc[:, ["Open", "High", "Low", "Close", "Volume"]].dropna()

    # ▶ aplanar MultiIndex (‘GC=F’) si aparece
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"[download] {len(df)} filas intradía obtenidas ({interval}, últimos {days}d)")
    return df

# ── 2) indicadores técnicos ───────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade RSI, MACD, ATR y retorno de vela."""
    df = df.copy()

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    # RSI
    df["RSI"] = RSIIndicator(close, window=WINDOWS["rsi"]).rsi()

    # MACD + señal
    macd = MACD(
        close,
        window_slow = WINDOWS["macd_slow"],
        window_fast = WINDOWS["macd_fast"],
        window_sign = WINDOWS["macd_sign"],
    )
    df["MACD"]     = macd.macd()
    df["MACD_sig"] = macd.macd_signal()

    # ATR
    df["ATR"] = AverageTrueRange(
        high, low, close, window=WINDOWS["atr"]
    ).average_true_range()

    # Retorno simple de la vela anterior
    df["ret_1"] = close.pct_change()

    print("[indicators] head:\n", df[["RSI", "MACD", "ATR", "ret_1"]].head())
    return df

# ── 3) rutina principal ───────────────────────────────────────────────
def main():
    # descarga
    df_raw = download_intraday(SYMBOL, LOOKBACK_DAYS, INTERVAL)

    # indicadores
    df_ind = compute_indicators(df_raw)

    # target = retorno de la siguiente vela
    df_ind["target"] = df_ind["Close"].pct_change().shift(-1)

    # limpiar NaNs iniciales
    df_out = df_ind.dropna()
    print(f"[output] {len(df_out)} filas tras dropna()")

    # guardar parquet
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_tag = df_out.index[-1].strftime("%Y%m%d_%H%M")
    out_path = os.path.join(OUTPUT_DIR, f"features_5m_{date_tag}.parquet")
    df_out.to_parquet(out_path)
    print(f"[OK] Features intradía guardadas en {out_path}")

if __name__ == "__main__":
    main()
