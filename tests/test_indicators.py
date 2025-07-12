import pandas as pd
import numpy as np
from src.indicators import compute_rsi, compute_macd, compute_bollinger, compute_atr, moving_averages

def test_compute_rsi_basic():
    series = pd.Series([1,2,3,4,5,6,7,8,9,10], dtype=float)
    rsi = compute_rsi(series, period=3)
    assert len(rsi) == len(series)
    assert rsi.max() <= 100
    assert rsi.min() >= 0

def test_compute_macd_shape():
    series = pd.Series(np.linspace(1,100,100))
    macd = compute_macd(series)
    assert list(macd.columns) == ["MACD", "Signal", "Hist"]
    assert len(macd) == len(series)

def test_compute_bollinger_bounds():
    series = pd.Series(np.arange(20), dtype=float)
    bb = compute_bollinger(series, window=5)
    comp = bb.dropna()
    assert (comp['BB_Upper'] >= comp['BB_Lower']).all()


def test_compute_atr_returns_atr_column():
    data = pd.DataFrame({
        'High': [10,11,12],
        'Low': [8,9,10],
        'Close': [9,10,11]
    })
    out = compute_atr(data.copy(), period=2)
    assert 'ATR' in out.columns
    assert len(out) == 3

def test_moving_averages_multiple_windows():
    df = pd.DataFrame({'Close': np.arange(10, dtype=float)})
    out = moving_averages(df.copy(), windows=(2,4))
    assert 'MA2' in out.columns
    assert 'MA4' in out.columns
