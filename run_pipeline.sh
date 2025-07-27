#!/usr/bin/env bash
export NEWSAPI_KEY="…"
source venv/bin/activate

# 1) datos diarios y precios
python3 -m src.data_fetch
python3 scripts/download_prices.py

# 1b) datos intradía 5m
python3 scripts/download_intraday.py

# 2) genera features diarias
python3 -m scripts.features

# 2.1) genera features intradía
python3 scripts/features_intraday.py

# 3) entrena modelo diario
python3 -m scripts.train_model

# 4) backtests diarios
python3 -m src.backtest_compare
python3 -m src.threshold_sweep

# 4.1) backtests intradía
python3 src/backtest_intraday.py
python3 src/threshold_sweep_intraday.py

# 5) walk-forward diario
python3 -m src.walk_forward

# 5.1) walk-forward intradía
python3 src/walk_forward_intraday.py

# 6) simulación futura diario
python3 scripts/paper_trade.py

# 6.1) simulación futura intradía
python3 scripts/paper_trade_intraday.py

