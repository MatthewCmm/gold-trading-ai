import pandas as pd
from src.backtester import run_backtest
from src.strategy import MACrossStrategy

def test_run_backtest_basic():
    df = pd.DataFrame({
        'Open':[1,2,3,4,5,6,7,8,9,10],
        'High':[1,2,3,4,5,6,7,8,9,10],
        'Low':[1,2,3,4,5,6,7,8,9,10],
        'Close':[1,2,3,4,5,6,7,8,9,10],
        'Volume':[100]*10
    })
    stats, bt = run_backtest(df, MACrossStrategy, cash=1000)
    assert 'Return [%]' in stats
    assert stats['Return [%]'] >= -100
