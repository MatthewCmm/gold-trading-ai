#!/usr/bin/env python3
# scripts/features.py

print("▶️  Loaded scripts/features.py")

import os
import datetime as dt
import glob
from pathlib import Path

import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline

from src.indicators   import compute_rsi, compute_macd, compute_atr
from src.news         import fetch_gold_news, enrich_news
from src.news_store   import update_news_history

# ────────────────────────────────
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
# ────────────────────────────────


# ──────────────────────────────── helpers
def load_raw() -> pd.DataFrame:
    """Load daily gold price parquet and flatten columns if needed."""
    px_path = os.path.join(RAW_DIR, "GCF_prices.parquet")
    df_px   = pd.read_parquet(px_path)
    if hasattr(df_px.columns, "nlevels") and df_px.columns.nlevels > 1:
        df_px.columns = df_px.columns.get_level_values(0)
    return df_px


def build_macro_features() -> pd.DataFrame:
    """
    Load and transform every macro series in data/raw,
    **excluding** price or intraday parquet dumps.
    """
    files = glob.glob(f"{RAW_DIR}/*.parquet")

    # Keep only genuine macro series
    macro_files = [
        f for f in files
        if not f.endswith("_prices.parquet")            # skip GCF_prices.parquet
           and "intraday" not in Path(f).stem.lower()   # skip GCF_intraday_5m.parquet
    ]

    df_list = []
    for f in macro_files:
        name = Path(f).stem
        series_df = pd.read_parquet(f)[[name]]
        df_list.append(series_df)

    df_mac = pd.concat(df_list, axis=1).ffill().dropna()

    # --- key extra series (ensure present) ---
    unrate = pd.read_parquet(f"{RAW_DIR}/UNRATE.parquet")[['UNRATE']]
    vix    = pd.read_parquet(f"{RAW_DIR}/VIXCLS.parquet")[['VIXCLS']]
    dxy    = pd.read_parquet(f"{RAW_DIR}/DTWEXBGS.parquet")[['DTWEXBGS']]
    df_mac = pd.concat([df_mac, unrate, vix, dxy], axis=1).ffill()

    df_mac.index = pd.to_datetime(df_mac.index).tz_localize(None)
    df_mac = df_mac.loc[:, ~df_mac.columns.duplicated()]

    # Base transforms
    df_mac['Inflation_YoY'] = df_mac['CPIAUCSL'] / df_mac['CPIAUCSL'].shift(12) - 1
    df_mac['USD_Return']    = df_mac['DEXUSAL'].pct_change()
    df_mac['YieldSpread']   = df_mac['DGS10'] - df_mac['DGS2']
    df_mac['GDP_Growth']    = df_mac['GDPC1'] / df_mac['GDPC1'].shift(4) - 1

    for col in ['Inflation_YoY', 'USD_Return', 'YieldSpread', 'GDP_Growth']:
        df_mac[f"{col}_L1"]  = df_mac[col].shift(1)
        df_mac[f"{col}_MA3"] = df_mac[col].rolling(3).mean()

    return df_mac


def engineer_features(df_px: pd.DataFrame, df_mac: pd.DataFrame) -> pd.DataFrame:
    """Merge price, macro & news and compute technical indicators."""
    df = df_px.rename(columns={
        "Open":   "PX_Open",
        "High":   "PX_High",
        "Low":    "PX_Low",
        "Close":  "PX_Close",
        "Volume": "PX_Volume",
    })

    df = df.join(df_mac, how="left")

    for c in ["sentiment", "news_count", "sentiment_diff"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df = df.ffill()

    # ── Technicals ─────────────────
    df["RSI_14"] = compute_rsi(df["PX_Close"], period=14)

    atr_df = pd.DataFrame(
        {"High": df["PX_High"], "Low": df["PX_Low"], "Close": df["PX_Close"]}
    )
    df["ATR_14"] = compute_atr(atr_df, period=14)["ATR"]

    macd_df = compute_macd(df["PX_Close"], fast=12, slow=26, signal=9)
    df["MACD"]     = macd_df["MACD"]
    df["MACD_sig"] = macd_df["Signal"]

    df["EMA_20"] = df["PX_Close"].ewm(span=20).mean()
    df["EMA_50"] = df["PX_Close"].ewm(span=50).mean()

    return df
# ────────────────────────────────


def main():
    print("▶️  Starting feature-build pipeline…")
    os.makedirs(OUT_DIR, exist_ok=True)
    today = dt.datetime.now().strftime("%Y%m%d")

    df_px  = load_raw()
    df_mac = build_macro_features()

    # ── News sentiment (fail-safe) ─────────────────────────────
    news_key = os.getenv("NEWSAPI_KEY")
    try:
        if not news_key:
            raise RuntimeError("NEWSAPI_KEY env var not set")
        df_new      = fetch_gold_news(api_key=news_key, query="gold", limit=50)
        df_enriched = enrich_news(df_new)

        hist_path   = os.path.join(RAW_DIR, "news_history.csv")
        df_history  = update_news_history(df_enriched, hist_path)
        df_history["publishedAt"] = pd.to_datetime(df_history["publishedAt"])

        df_news = (
            df_history
            .set_index("publishedAt")["sentiment"]
            .resample("D").mean().ffill()
        )
        news_count     = df_history.set_index("publishedAt").resample("D").size().rename("news_count")
        sentiment_diff = df_news.diff().rename("sentiment_diff")

        df_news_feat = pd.concat([df_news, news_count, sentiment_diff], axis=1).ffill()

    except Exception as e:
        # ⚠️  Any error → create neutral (zeros) news features
        print(f"⚠️  News sentiment unavailable ({e}); filling zeros.")
        idx_full      = pd.to_datetime(df_px.index).tz_localize(None)
        df_news_feat  = pd.DataFrame(
            index=idx_full,
            data={
                "sentiment":       0.0,
                "news_count":      0,
                "sentiment_diff":  0.0,
            },
        )

    # align & merge
    df_news_feat.index = df_news_feat.index.tz_localize(None)
    df_mac = df_mac.join(df_news_feat, how="left").ffill()

    # ── Final feature set ─────────────────────────────────────
    df_feat  = engineer_features(df_px, df_mac)
    out_path = os.path.join(OUT_DIR, f"features_{today}.parquet")
    df_feat.to_parquet(out_path)
    print(f"[OK] Features saved to {out_path}")


if __name__ == "__main__":
    main()
