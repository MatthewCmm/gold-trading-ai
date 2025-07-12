# gold-trading-ai

Utilities for backtesting simple COMEX gold futures strategies in Python.

## Overview
- Download historical gold prices with `yfinance`.
- Compute indicators such as moving averages, RSI, MACD, Bollinger Bands and ATR.
- Run a moving-average crossover strategy using the `backtesting` package.
- Includes unit tests covering indicator helpers and the backtester.

## Project Structure
```
src/      # Source modules (data fetch, indicators, backtester, strategy)
tests/    # Unit tests
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Run an end-to-end example fetching data, adding indicators and
backtesting the strategy:
```bash
python src/main.py
```

## Testing
```bash
pytest -q
```

## License
MIT License
