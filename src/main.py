from src.data_fetch   import download_gold
from src.indicators   import add_indicators
from src.strategy     import MACrossStrategy
from src.backtester   import run_backtest

def main():
    gold = download_gold(start="2015-01-01")
    print("Columnas disponibles:", gold.columns.tolist())   # <<-- línea de debug
    gold = add_indicators(gold)

    stats, bt = run_backtest(
        gold,
        MACrossStrategy,
        cash=10_000,
        commission=0.001
    )

    print(stats)
    bt.plot()

if __name__ == "__main__":
    main()


