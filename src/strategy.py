import pandas as pd
from backtesting import Strategy

# --- Average True Range helper ------------------------------------------
def atr(high, low, close, n=14):
    """
    Vectorised ATR that works with Backtesting's _Array objects.
    """
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)

    tr = pd.concat(
        [
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(n).mean()
# ------------------------------------------------------------------------

class MACrossStrategy(Strategy):
    # Hyper-parameters
    n1, n2 = 10, 80
    atr_win     = 14
    atr_factor  = 1.2
    sl_mult     = 2.0
    trail_mult  = 1.0

    def init(self):
        # Moving averages
        self.ma1 = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(),
                          self.data.Close)
        self.ma2 = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(),
                          self.data.Close)

        # ATR (custom helper) and its 100-day mean
        self.atr = self.I(
            atr,
            self.data.High, self.data.Low, self.data.Close,
            self.atr_win,
        )
        self.atr_mean = self.I(
            lambda x: pd.Series(x).rolling(100).mean(),
            self.atr
        )

    def next(self):
        # ---------------------------------------------------------------
        # 1) Volatility filter – skip if ATR is below threshold
        # ---------------------------------------------------------------
        if self.atr[-1] < self.atr_factor * self.atr_mean[-1]:
            return  # market too quiet → no trading

        # ---------------------------------------------------------------
        # 2) Moving-average crossover signals
        # ---------------------------------------------------------------
        long_sig = (
            self.ma1[-1] > self.ma2[-1] and
            self.ma1[-2] <= self.ma2[-2]
        )
        exit_sig = (
            self.ma1[-1] < self.ma2[-1] and
            self.ma1[-2] >= self.ma2[-2]
        )

        # ---------------------------------------------------------------
        # 3) Entry – open long with initial stop-loss
        # ---------------------------------------------------------------
        if long_sig and not self.position.is_long:
            sl = self.data.Close[-1] - self.sl_mult * self.atr[-1]
            self.buy(sl=sl)                 # initial protective stop
            return                          # wait for next bar

        # ---------------------------------------------------------------
        # 4) Exit on bearish crossover
        # ---------------------------------------------------------------
        if exit_sig and self.position.is_long:
            self.position.close()
            return

        # ---------------------------------------------------------------
        # 5) Trailing-stop management (long positions only)
        # ---------------------------------------------------------------
        if self.position.is_long:
            new_sl = self.data.Close[-1] - self.trail_mult * self.atr[-1]

        # Tighten the stop only if it can be moved up
        current_sl = getattr(self.position, "sl_price", None)
        if current_sl is None or new_sl > current_sl:
            # For backtesting >= 0.6 use update_sl / sl_price
            if hasattr(self.position, "update_sl"):
                self.position.update_sl(price=new_sl)
        else:  # fallback for even newer versions
            self.position.sl_price = new_sl



