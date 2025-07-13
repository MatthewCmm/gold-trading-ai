from src.data_fetch import download_gold
from src.strategy import MACrossStrategy
from src.backtester import run_backtest


def test_pipeline_runs():
    df = download_gold(period="3mo")  # descarga pequeña = test rápido
    stats, _ = run_backtest(df, MACrossStrategy, cash=10_000)

    # stats es Series → valores están en su índice
    assert "Equity Final [$]" in stats.index
    # También puedes comprobar que la equity final es > 0
    assert stats["Equity Final [$]"] > 0
