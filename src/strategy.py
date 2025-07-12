from backtesting import Strategy

class MACrossStrategy(Strategy):
    n1, n2 = 20, 50
    def init(self):
        self.ma1 = self.I(lambda x: x.rolling(self.n1).mean(), self.data.Close)
        self.ma2 = self.I(lambda x: x.rolling(self.n2).mean(), self.data.Close)
    def next(self):
        if self.ma1[-1] > self.ma2[-1] and self.ma1[-2] <= self.ma2[-2]:
            self.buy()
        elif self.ma1[-1] < self.ma2[-1] and self.ma1[-2] >= self.ma2[-2]:
            self.sell()
