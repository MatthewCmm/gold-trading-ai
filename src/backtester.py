"""
backtester.py
Core helpers to run and optimise back-tests.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from backtesting import Backtest
from src.strategy import MACrossStrategy


# --------------------------------------------------------------------------- #
# 1) Standard back-test wrapper
# --------------------------------------------------------------------------- #
def run_backtest(
    df: pd.DataFrame,
    strategy,
    *,
    cash: float = 10_000,
    commission: float = 0.001,
) -> Tuple[pd.Series, Backtest]:
    """
    Executes a back-test and returns (stats, Backtest instance).

    Parameters
    ----------
    df : pd.DataFrame
        Price data with columns Open, High, Low, Close, Volume.
    strategy : backtesting.Strategy
        A strategy subclass to be evaluated.
    cash : float, default 10_000
        Initial capital.
    commission : float, default 0.001
        Commission per trade (fraction).

    Returns
    -------
    stats : pd.Series
        Performance metrics.
    bt : Backtest
        Backtest object (for .plot(), etc.).
    """
    bt = Backtest(df, strategy, cash=cash, commission=commission)
    stats = bt.run()
    return stats, bt


# --------------------------------------------------------------------------- #
# 2) Parameter sweep for MA-Cross strategy
# --------------------------------------------------------------------------- #
def optimize_ma(
    df,
    *,
    cash: float = 10_000,
    commission: float = 0.001,
    n1_range=range(10, 55, 5),
    n2_range=range(60, 210, 10),
):
    """
    Grid-search on n1 / n2, maximise Sharpe Ratio.
    Returns the optimizer Series (best stats).
    """
    bt = Backtest(df, MACrossStrategy, cash=cash, commission=commission)
    # --- only the Series is needed, so unpack the first element -----------
    best_stats = bt.optimize(
        n1=n1_range,
        n2=n2_range,
        constraint=lambda p: p.n1 < p.n2,
        maximize="Sharpe Ratio",
        return_heatmap=False, 
    )
    return best_stats          # <- just the Series


