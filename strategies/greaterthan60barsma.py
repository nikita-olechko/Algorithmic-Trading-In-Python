# A sample strategy to test the live trading bot's functionality. Only run on paper trading!
# This strategy is a simple moving average strategy that buys when the average price of the last 60 bars is less than
# the average price of the last 60 bars, and sells when the opposite is true.

def generate60PeriodSMALastRow(barDataFrame):
    """
    A function that generates the 60 period simple moving average for the last row.
    """
    barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA'] = barDataFrame['Average'].tail(60).mean()
    return barDataFrame


def generate60PeriodSMAWholeDataFrame(barDataFrame):
    """
    A function that generates the 60 period simple moving average for the entire dataframe.
    """
    barDataFrame['60PeriodSMA'] = barDataFrame['Average'].rolling(60).mean()
    return barDataFrame


def sampleSMABuySellStrategy(barDataFrame, last_order_index=0, ticker="", current_index=0):
    """
    A function that returns 1 (buy), -1 (sell), 2 (hold), or 0 (nothing) depending on some condition, in
    this case Average > 60Period SMA, BUY, and vice versa
    """
    if barDataFrame.loc[barDataFrame.index[-1], 'Average'] < barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA']:
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
