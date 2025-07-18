# scripts/optimize_ma.py
import os
import datetime as dt
from src.data_fetch  import download_gold
from src.indicators  import add_indicators
from src.backtester  import optimize_ma


def main():
    df = add_indicators(download_gold(start="2010-01-01"))

    best_stats = optimize_ma(df)             # <-- ya es una Serie

    # Extract winning parameters from the Strategy object
    best_params = {
        "n1": best_stats._strategy.n1,
        "n2": best_stats._strategy.n2,
    }

    print("Best parameters :", best_params)
    print("Best Sharpe     :", best_stats["Sharpe Ratio"])

    # Persist results
    os.makedirs("results", exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_stats.to_csv(f"results/opt_ma_{ts}.csv")


if __name__ == "__main__":
    main()

