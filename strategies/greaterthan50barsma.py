# A sample strategy to test the live trading bot's functionality. Only run on paper trading!
# This strategy is a simple moving average strategy that buys when the average price of the last 60 bars is greater than
# the average price of the last 60 bars, and sells when the opposite is true.

def generate60PeriodSMA(barDataFrame):
    barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA'] = barDataFrame['Average'].tail(60).mean()
    return barDataFrame


def generate60PeriodSMA_backtest(barDataFrame):
    barDataFrame['60PeriodSMA'] = barDataFrame['Average'].rolling(60).mean()
    return barDataFrame


def sampleSMABuySellStrategy(barDataFrame, last_order_index, ticker=None, current_index=-1):
    """
    A function that returns "BUY", "SELL" or "" depending on some condition, in
    this case Average > 60Period SMA, BUY, and vice versa
    """
    if barDataFrame.loc[barDataFrame.index[current_index], 'Average'] < \
            barDataFrame.loc[barDataFrame.index[current_index], '60PeriodSMA']:
        if barDataFrame["Orders"][last_order_index] != 1:
            return 1
        else:
            return 2
    elif barDataFrame.loc[barDataFrame.index[current_index], 'Average'] >= \
            barDataFrame.loc[barDataFrame.index[current_index], '60PeriodSMA']:
        if barDataFrame["Orders"][last_order_index] != -1:
            return -1
        else:
            return 0
    else:
        return 0
