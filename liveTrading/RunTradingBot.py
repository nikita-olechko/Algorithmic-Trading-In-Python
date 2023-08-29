from liveTrading.LiveTradingBot import Bot
from strategies.greaterthan60barsma import sampleSMABuySellStrategy, generate60PeriodSMAWholeDataFrame

# Start Bot(s)
bot1 = Bot(symbol="AAPL", quantity=1, buySellConditionFunc=sampleSMABuySellStrategy,
           generateNewDataFunc=generate60PeriodSMAWholeDataFrame)
