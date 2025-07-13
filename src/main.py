import os
import datetime as dt

from src.data_fetch   import download_gold
from src.indicators   import add_indicators
from src.strategy     import MACrossStrategy
from src.backtester   import run_backtest


def main():
    # 1. Download historical data
    gold = download_gold(start="2015-01-01")
    print("Available columns:", gold.columns.tolist())

    # 2. Add technical indicators
    gold = add_indicators(gold)

    # 3. Run the backtest
    stats, bt = run_backtest(
        gold,
        MACrossStrategy,
        cash=10_000,
        commission=0.001
    )

    # Print the performance summary in the console
    print(stats)

    # 4. Persist results for later auditing / presentations
    os.makedirs("results", exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp

    out_csv  = f"results/stats_{ts}.csv"
    out_html = f"results/equity_{ts}.html"

    # Save metrics as CSV (one-line DataFrame)
    stats.to_frame().T.to_csv(out_csv, index=False)

    # Save the interactive equity curve without opening a browser
    bt.plot(filename=out_html, open_browser=False)

    print(f"[OK] Metrics saved to {out_csv}")
    print(f"[OK] Equity curve saved to {out_html}")


if __name__ == "__main__":
    main()



