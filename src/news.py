import requests
import pandas as pd
from textblob import TextBlob

NEWS_URL = "https://newsapi.org/v2/everything"

def fetch_gold_news(*, api_key: str, query: str = "gold", limit: int = 20) -> pd.DataFrame:
    """Fetch recent gold headlines via NewsAPI."""
    params = {
        "q":        query,
        "sortBy":   "publishedAt",
        "pageSize": limit,
        "language": "en",
        "apiKey":   api_key,
    }
    resp = requests.get(NEWS_URL, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    records = []
    for art in articles:
        records.append({
            "publishedAt": art.get("publishedAt"),
            "title":       art.get("title"),
            "description": art.get("description"),
            "url":         art.get("url"),
        })
    return pd.DataFrame.from_records(records)


def enrich_news(df: pd.DataFrame) -> pd.DataFrame:
    """Add a TextBlob‐based sentiment polarity column."""
    df = df.copy()
    df["sentiment"] = (
        df["title"]
          .fillna("")
          .apply(lambda txt: TextBlob(txt).sentiment.polarity)
    )
    return df
