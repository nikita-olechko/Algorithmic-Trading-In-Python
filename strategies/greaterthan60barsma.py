# A sample strategy to test the live trading bot's functionality. Only run on paper trading!
# This strategy is a simple moving average strategy that buys when the average price of the last 60 bars is greater than
# the average price of the last 60 bars, and sells when the opposite is true.

def generate60PeriodSMA(barDataFrame):
    barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA'] = barDataFrame['Average'].tail(60).mean()
    return barDataFrame


# Note that the difference between these two functions, is that the backtesting function applies to the whole dataframe,
# whereas the live trading function only applies to the last row of the dataframe. This is because the live trading
# function is called every time a new bar is received, whereas the backtesting function is called once for the whole
# dataframe.
def generate60PeriodSMA_backtest(barDataFrame):
    barDataFrame['60PeriodSMA'] = barDataFrame['Average'].rolling(60).mean()
    return barDataFrame


def sampleSMABuySellStrategy(barDataFrame, last_order_index=0, ticker=""):
    """
    A function that returns 1 (buy), -1 (sell), 2 (hold), or 0 (nothing) depending on some condition, in
    this case Average > 60Period SMA, BUY, and vice versa
    """
    if barDataFrame.loc[barDataFrame.index[-1], 'Average'] < \
            barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA']:
        if barDataFrame["Orders"][last_order_index] != 1:
            return 1
        else:
            return 2
    elif barDataFrame.loc[barDataFrame.index[-1], 'Average'] >= \
            barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA']:
        # Note: checking whether last_order_index != 0 ensures that we do not sell as our first order,
        # this can be changed for accounts that support short selling
        if barDataFrame["Orders"][last_order_index] != -1 and last_order_index != 0:
            return -1
        else:
            return 0
    else:
        return 0
