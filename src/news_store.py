import pandas as pd
from pathlib import Path

COLUMNS = ["publishedAt", "title", "description", "url", "sentiment"]

def load_news_history(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=COLUMNS)

def save_news_history(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def update_news_history(df_new: pd.DataFrame, path: str) -> pd.DataFrame:
    df_old = load_news_history(path)
    combined = (
        pd.concat([df_old, df_new])
          .drop_duplicates(subset=["url"], keep="last")
          .sort_values("publishedAt")
          .reset_index(drop=True)
    )
    save_news_history(combined, path)
    return combined
