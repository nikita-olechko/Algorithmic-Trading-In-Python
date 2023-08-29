# A sample strategy to test the live trading bot's functionality. Only run on paper trading!
# This strategy is a simple moving average strategy that buys when the average price of the last 10 bars is less than
# the average price of the last 10 bars, and sells when the opposite is true.

def generate10PeriodSMALastRow(barDataFrame):
    """
    A function that generates the 10 period simple moving average for the last row.
    """
    barDataFrame.loc[barDataFrame.index[-1], '10PeriodSMA'] = barDataFrame['Average'].tail(10).mean()
    return barDataFrame


def generate10PeriodSMAWholeDataFrame(barDataFrame):
    """
    A function that generates the 10 period simple moving average for the entire dataframe.
    """
    barDataFrame['10PeriodSMA'] = barDataFrame['Average'].rolling(10).mean()
    return barDataFrame


def sampleSMABuySellStrategy(barDataFrame, last_order_index=0, ticker="", current_index=0):
    """
    A function that returns 1 (buy), -1 (sell), 2 (hold), or 0 (nothing) depending on some condition, in
    this case Average > 10Period SMA, BUY, and vice versa
    """
    current_price = barDataFrame.loc[barDataFrame.index[-1], 'Average']
    current_sma = barDataFrame.loc[barDataFrame.index[-1], '10PeriodSMA']

    # if current_price < current_sma:
    if True:
        if barDataFrame["Orders"][last_order_index] != 1 and barDataFrame["Orders"][last_order_index] != 2:
            return 1
        else:
            return 2
    elif current_price >= current_sma:
        # Note: checking whether last_order_index != 0 ensures that we do not sell as our first order,
        # this can be changed for accounts that support short selling
        if barDataFrame["Orders"][last_order_index] != -1 and last_order_index != 0:
            return -1
        else:
            return 0
    else:
        return 0
