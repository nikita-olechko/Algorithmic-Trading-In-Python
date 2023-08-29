from liveTrading.LiveTradingBot import Bot
from strategies.greaterthan60barsma import sampleSMABuySellStrategy, generate60PeriodSMAWholeDataFrame

# Start Bot(s)
bot1 = Bot(symbol="AAPL", quantity=1, buySellConditionFunc=sampleSMABuySellStrategy,
           generateNewDataFunc=generate60PeriodSMAWholeDataFrame,
           last_row_only=False, periods_to_analyze=65, operate_on_minute_data=True)
