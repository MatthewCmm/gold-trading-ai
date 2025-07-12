import pandas as pd
from backtesting import Strategy

class MACrossStrategy(Strategy):
    n1, n2 = 20, 50
    def init(self):
        # Backtesting provides price data as numpy arrays which do not expose
        # pandas rolling. Convert to Series to compute moving averages.
        self.ma1 = self.I(
            lambda x: pd.Series(x).rolling(self.n1).mean().values,
            self.data.Close,
        )
        self.ma2 = self.I(
            lambda x: pd.Series(x).rolling(self.n2).mean().values,
            self.data.Close,
        )
    def next(self):
        if self.ma1[-1] > self.ma2[-1] and self.ma1[-2] <= self.ma2[-2]:
            self.buy()
        elif self.ma1[-1] < self.ma2[-1] and self.ma1[-2] >= self.ma2[-2]:
            self.sell()
