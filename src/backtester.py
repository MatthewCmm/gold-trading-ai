from backtesting import Backtest

def run_backtest(df, strategy, cash=10000, commission=0.001):
    bt = Backtest(df, strategy, cash=cash, commission=commission)
    stats = bt.run()
    return stats, bt
