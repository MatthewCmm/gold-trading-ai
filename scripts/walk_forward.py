"""
scripts/walk_forward.py
Manual walk-forward:
 - in-sample length: window_size days
 - step length:      step_size days
Back-tests each window and records OOS metrics.
"""

import os
import datetime as dt
import pandas as pd

from src.data_fetch  import download_gold
from src.indicators  import add_indicators
from src.strategy    import MACrossStrategy
from src.backtester  import run_backtest


def main():
    # 1) Load and augment the full series
    df_full = add_indicators(download_gold(start="2010-01-01"))

    # 2) Define window/step sizes (in trading days)
    window_size = 3 * 252   # ≈ 3 years
    step_size   = 1 * 252   # ≈ 1 year

    results = []

    # 3) Slide the window from 0 → len(df_full)-window_size
    for start in range(0, len(df_full) - window_size + 1, step_size):
        df_win = df_full.iloc[start : start + window_size]

        # 4) Run back-test on this window
        stats, _ = run_backtest(
            df_win,
            MACrossStrategy,
            cash=10_000,
            commission=0.001
        )

        # 5) Collect key metrics
        results.append({
            "window_start":    df_win.index[0].strftime("%Y-%m-%d"),
            "window_end":      df_win.index[-1].strftime("%Y-%m-%d"),
            "Sharpe":          stats["Sharpe Ratio"],
            "MaxDD[%]":        stats["Max. Drawdown [%]"],
            "CAGR[%]":         stats["CAGR [%]"],
            "Trades":          stats["# Trades"],
        })

    # 6) Save to CSV
    df_res = pd.DataFrame(results)
    ts     = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/walkfwd_{ts}.csv"
    df_res.to_csv(out_csv, index=False)

    print(df_res)
    print(f"[OK] Walk-forward results saved to {out_csv}")


if __name__ == "__main__":
    main()

