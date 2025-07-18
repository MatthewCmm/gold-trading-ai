#!/usr/bin/env python3
# scripts/features.py

print("▶️  Loaded scripts/features.py")

import os
import datetime as dt
import glob
import pandas as pd
from src.indicators import compute_rsi, compute_macd, compute_atr
from pathlib import Path
from newsapi import NewsApiClient
from transformers import pipeline
from src.news import fetch_gold_news, enrich_news
from src.news_store import update_news_history, load_news_history

# Directories
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

def load_raw() -> pd.DataFrame:
    """
    Load raw price data from GCF_prices.parquet and flatten columns if needed.
    Returns:
        df_px: DataFrame of price data with single-level columns.
    """
    px_path = os.path.join(RAW_DIR, "GCF_prices.parquet")
    df_px = pd.read_parquet(px_path)
    if hasattr(df_px.columns, "nlevels") and df_px.columns.nlevels > 1:
        df_px.columns = df_px.columns.get_level_values(0)
    return df_px


def build_macro_features() -> pd.DataFrame:
    """
    Load and transform all macro series from data/raw (excluding price parquet).
    Returns:
        df_mac: DataFrame of macro indicators and derived features.
    """
    files = glob.glob(f"{RAW_DIR}/*.parquet")
    macro_files = [f for f in files if not f.endswith("_prices.parquet")]

    df_list = []
    for f in macro_files:
        name = Path(f).stem
        series_df = pd.read_parquet(f)[[name]]
        df_list.append(series_df)

    df_mac = pd.concat(df_list, axis=1).ffill().dropna()

    # —— Nuevas series macro ——
    unrate = pd.read_parquet(f"{RAW_DIR}/UNRATE.parquet")[['UNRATE']]
    vix    = pd.read_parquet(f"{RAW_DIR}/VIXCLS.parquet")[['VIXCLS']]
    dxy    = pd.read_parquet(f"{RAW_DIR}/DTWEXBGS.parquet")[['DTWEXBGS']]
    df_mac = pd.concat([df_mac, unrate, vix, dxy], axis=1).ffill()
    # Ensure DatetimeIndex is naive
    df_mac.index = pd.to_datetime(df_mac.index).tz_localize(None)

    # Eliminar columnas duplicadas
    df_mac = df_mac.loc[:, ~df_mac.columns.duplicated()]

    # Transformaciones base
    df_mac['Inflation_YoY'] = df_mac['CPIAUCSL'] / df_mac['CPIAUCSL'].shift(12) - 1
    df_mac['USD_Return']    = df_mac['DEXUSAL'].pct_change()
    df_mac['YieldSpread']   = df_mac['DGS10'] - df_mac['DGS2']
    df_mac['GDP_Growth']    = df_mac['GDPC1'] / df_mac['GDPC1'].shift(4) - 1

    for col in ['Inflation_YoY', 'USD_Return', 'YieldSpread', 'GDP_Growth']:
        df_mac[f"{col}_L1"]  = df_mac[col].shift(1)
        df_mac[f"{col}_MA3"] = df_mac[col].rolling(3).mean()

    return df_mac


def engineer_features(df_px: pd.DataFrame, df_mac: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price and macro data, compute technical indicators and return features DataFrame.
    """
    # 1) Rename price columns
    df = df_px.rename(columns={
        "Open":   "PX_Open",
        "High":   "PX_High",
        "Low":    "PX_Low",
        "Close":  "PX_Close",
        "Volume": "PX_Volume",
    })

    # 2) Merge macro and news features without dropping rows
    df = df.join(df_mac, how="left")

    # 3) Fill missing news-derived features with 0
    for col in ["sentiment", "news_count", "sentiment_diff"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 4) Forward-fill the rest of features
    df = df.ffill()

    # —— Technical indicators ——
    # RSI 14
    df["RSI_14"] = compute_rsi(df["PX_Close"], period=14)
    # ATR 14
    atr_df = pd.DataFrame({
        "High":  df["PX_High"],
        "Low":   df["PX_Low"],
        "Close": df["PX_Close"]
    })
    atr_res = compute_atr(atr_df, period=14)
    df["ATR_14"] = atr_res["ATR"]
    # MACD
    macd_df = compute_macd(df["PX_Close"], fast=12, slow=26, signal=9)
    df["MACD"]     = macd_df["MACD"]
    df["MACD_sig"] = macd_df["Signal"]
    # EMAs
    df["EMA_20"] = df["PX_Close"].ewm(span=20).mean()
    df["EMA_50"] = df["PX_Close"].ewm(span=50).mean()

    return df


def fetch_news_sentiment(api_key: str) -> pd.DataFrame:
    """
    Fetch up to 100 "gold" headlines via NewsAPI, compute average sentiment
    using a HF pipeline, return a 1-row DataFrame indexed by today.
    """
    client = NewsApiClient(api_key=api_key)
    articles = client.get_everything(q='gold', language='en', page_size=100)

    texts = [art['title'] + '. ' + (art.get('description') or '') for art in articles['articles']]
    clf = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    scores = [res['score'] if res['label'] == 'POSITIVE' else -res['score'] for res in clf(texts)]
    avg_score = sum(scores) / len(scores)

    idx = pd.Timestamp.today().normalize()
    return pd.DataFrame({'NewsSentiment': [avg_score]}, index=[idx])


def main():
    print('▶️  Starting feature-build pipeline…')
    os.makedirs(OUT_DIR, exist_ok=True)
    today = dt.datetime.now().strftime('%Y%m%d')

    df_px = load_raw()
    df_mac = build_macro_features()

    news_key = os.getenv('NEWSAPI_KEY')
    if not news_key:
        raise RuntimeError('Please set NEWSAPI_KEY in your environment')

    df_new      = fetch_gold_news(api_key=news_key, query='gold', limit=50)
    df_enriched = enrich_news(df_new)

    history_path = os.path.join(RAW_DIR, 'news_history.csv')
    df_history   = update_news_history(df_enriched, history_path)
    df_history['publishedAt'] = pd.to_datetime(df_history['publishedAt'])

    # Daily sentiment series
    df_news = (
        df_history
        .set_index('publishedAt')['sentiment']
        .astype(float)
        .resample('D').mean()
        .ffill()
    )
    df_news.index = df_news.index.tz_localize(None)

    # —— Bloque 3: Features de noticias ——
    news_count = (
        df_history
        .set_index('publishedAt')
        .resample('D')
        .size()
        .rename('news_count')
    )
    news_count.index = news_count.index.tz_localize(None)

    sentiment_diff = df_news.diff().rename('sentiment_diff')
    sentiment_diff.index = sentiment_diff.index.tz_localize(None)

    df_news_feat = pd.concat([df_news, news_count, sentiment_diff], axis=1).ffill()
    # Ensure tz-naive index for join
    df_news_feat.index = df_news_feat.index.tz_localize(None)

    df_mac = df_mac.join(df_news_feat, how='left').ffill()

    df_feat = engineer_features(df_px, df_mac)
    out_path = os.path.join(OUT_DIR, f'features_{today}.parquet')
    df_feat.to_parquet(out_path)
    print(f'[OK] Features saved to {out_path}')

if __name__ == '__main__':
    main()


