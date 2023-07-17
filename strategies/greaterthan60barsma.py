# A sample strategy to test the live trading bot's functionality. Only run on paper trading!

def generate60PeriodSMA(barDataFrame):
    barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA'] = barDataFrame['Average'].tail(60).mean()
    return barDataFrame


def sampleSMABuySellStrategy(barDataFrame):
    """
    A function that returns "BUY", "SELL" or "" depending on some condition, in
    this case Average > 60Period SMA, BUY, and vice versa
    """
    if barDataFrame.loc[barDataFrame.index[-1], 'Average'] < barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA']:
        return "BUY"
    elif barDataFrame.loc[barDataFrame.index[-1], 'Average'] > barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA']:
        return "SELL"
    else:
        return ""
