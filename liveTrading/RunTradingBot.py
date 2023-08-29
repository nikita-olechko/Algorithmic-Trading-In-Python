from liveTrading.LiveTradingBot import Bot
from strategies.greaterthan10barsma import sampleSMABuySellStrategy, generate10PeriodSMAWholeDataFrame

# Start Bot(s)
bot = Bot(symbol="AAPL", quantity=1, buySellConditionFunc=sampleSMABuySellStrategy,
          generateNewDataFunc=generate10PeriodSMAWholeDataFrame,
          last_row_only=False, periods_to_analyze=15, operate_on_minute_data=True)
