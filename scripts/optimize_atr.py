"""
scripts/optimize_atr.py
Grid-search sobre los multiplicadores de ATR:
  • atr_factor  – filtro de volatilidad (ATR > atr_factor × ATR_mean)
  • sl_mult     – stop-loss inicial = sl_mult × ATR
  • trail_mult  – trailing stop      = trail_mult × ATR
Guarda el ranking en results/opt_atr_<timestamp>.csv
"""

import itertools, datetime as dt, os
import pandas as pd

from src.data_fetch   import download_gold
from src.indicators   import add_indicators
from src.strategy     import MACrossStrategy
from src.backtester   import run_backtest


# ---------- 1. Configura el rango de búsqueda ----------------------------
ATR_FACTORS  = [1.1, 1.2, 1.3, 1.4]
SL_MULTS     = [1.5, 2.0, 2.5]
TRAIL_MULTS  = [0.8, 1.0, 1.2]

# ---------- 2. Descarga datos una sola vez --------------------------------
df = add_indicators(download_gold(start="2010-01-01"))

results = []

# ---------- 3. Recorre todas las combinaciones ----------------------------
for atr_f, sl_m, tr_m in itertools.product(ATR_FACTORS, SL_MULTS, TRAIL_MULTS):
    # Ajusta los parámetros de la estrategia
    MACrossStrategy.atr_factor = atr_f
    MACrossStrategy.sl_mult    = sl_m
    MACrossStrategy.trail_mult = tr_m

    stats, _ = run_backtest(df, MACrossStrategy, cash=10_000, commission=0.001)

    results.append(
        {
            "atr_factor": atr_f,
            "sl_mult":    sl_m,
            "trail_mult": tr_m,
            "Sharpe":     stats["Sharpe Ratio"],
            "MaxDD[%]":   stats["Max. Drawdown [%]"],
            "CAGR[%]":    stats["CAGR [%]"],
            "Trades":     stats["# Trades"],
        }
    )

# ---------- 4. Guarda el ranking ------------------------------------------
os.makedirs("results", exist_ok=True)
ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

df_res = pd.DataFrame(results).sort_values(
    by=["Sharpe", "MaxDD[%]"], ascending=[False, True]
)
out_csv = f"results/opt_atr_{ts}.csv"
df_res.to_csv(out_csv, index=False)

print(df_res.head(10))            # muestra las 10 mejores filas
print(f"[OK] Resultados guardados en {out_csv}")
