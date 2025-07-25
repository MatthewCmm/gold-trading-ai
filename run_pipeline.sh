#!/usr/bin/env bash
# exportar clave de NewsAPI
export NEWSAPI_KEY="9f7fb73c5676438db4b659de1eda132e"

# activar el virtualenv
source venv/bin/activate

# 1) descarga datos
python3 -m src.data_fetch
python3 scripts/download_prices.py

# 2) genera features
python3 -m scripts.features

# 3) entrena modelo
python3 -m scripts.train_model

# 4) backtests
python3 -m src.backtest_compare
python3 -m src.threshold_sweep

# 5) walk-forward
python3 -m src.walk_forward

# 6) simulación futura (paper trade)
python3 scripts/paper_trade.py
