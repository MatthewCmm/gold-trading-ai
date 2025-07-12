# gold-trading-ai
Repository for implementing and analyzing COMEX gold futures trading strategies with backtesting and predictive modeling.

📋 Overview

This project provides a complete pipeline to:

Fetch historical COMEX gold futures prices using yfinance.

Compute technical indicators: moving averages (MA), RSI, ATR.

Backtest strategies based on moving average crossovers or custom rules.

Visualize results with interactive plots and performance statistics.

(Optional) Train machine learning models for price or signal prediction.

Code is organized into reusable modules and Jupyter notebooks for exploratory analysis.

📦 Project Structure
gold-trading-ai/
│
├── data/                   # Data directory
│   ├── raw/                # Original fetched data
│   └── processed/          # Cleaned data ready for analysis
│
├── notebooks/              # Exploratory Jupyter notebooks
│   ├── 01_fetch_data.ipynb
│   ├── 02_compute_indicators.ipynb
│   ├── 03_backtest_strategy.ipynb
│   └── 04_train_ml_model.ipynb
│
├── src/                    # Source Python modules
│   ├── data_fetch.py       # Functions to download and save data
│   ├── indicators.py       # MA, RSI, ATR and signal calculations
│   ├── backtester.py       # Backtesting wrapper (Backtest, Strategy)
│   ├── model.py            # ML model definitions and training routines
│   └── main.py             # Orchestrator to run the full pipeline
│
├── tests/                  # Unit and integration tests
│   ├── test_indicators.py
│   └── test_backtester.py
│
├── config/                 # Configuration files
│   └── config.yaml         # Parameters (symbol, windows, commission, etc.)
│
├── requirements.txt        # Python dependencies
└── README.md               # This document
⚙️ Installation

Clone the repository:

git clone https://github.com/your_username/gold-trading-ai.git
cd gold-trading-ai

Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Customize parameters in config/config.yaml (e.g., ticker symbol, moving average periods, commission rates).

🚀 Usage

Fetch and preprocess data:

python src/data_fetch.py --config config/config.yaml

Compute technical indicators:

python src/indicators.py --input data/raw/gold.csv --output data/processed/indicators.csv

Run backtest:

python src/backtester.py --config config/config.yaml

Interactive exploration:

jupyter lab

Then open notebooks in the notebooks/ folder.

✅ Testing

Run the test suite:

pytest tests/

📄 License

MIT License – see the LICENSE file for full details.

 