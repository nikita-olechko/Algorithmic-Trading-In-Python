from liveTrading.LiveTradingBot import Bot
from strategies.greaterthan10barsma import sampleSMABuySellStrategy, generate10PeriodSMAWholeDataFrame

# Start Bot(s)

for ticker in ["XOM"]:

    bot = Bot(symbol=ticker, quantity=1, buySellConditionFunc=sampleSMABuySellStrategy,
              generateNewDataFunc=generate10PeriodSMAWholeDataFrame,
              last_row_only=False, periods_to_analyze=15)
