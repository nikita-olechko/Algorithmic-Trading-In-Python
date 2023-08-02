# A sample strategy to test the live trading bot's functionality. Only run on paper trading!
from utilities.dataGenerationUtilities import modify_func_from_last_row_to_all_rows


def generate60PeriodSMA(barDataFrame):
    barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA'] = barDataFrame['Average'].tail(60).mean()
    return barDataFrame


generate60PeriodSMA_backtest = modify_func_from_last_row_to_all_rows(generate60PeriodSMA)


def sampleSMABuySellStrategy(barDataFrame, last_order_index):
    """
    A function that returns "BUY", "SELL" or "" depending on some condition, in
    this case Average > 60Period SMA, BUY, and vice versa
    """
    if barDataFrame.loc[barDataFrame.index[-1], 'Average'] < barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA']:
        if barDataFrame["Orders"][last_order_index] != "BUY":
            return "BUY"
        else:
            return "HOLD"
    elif barDataFrame.loc[barDataFrame.index[-1], 'Average'] >= barDataFrame.loc[barDataFrame.index[-1], '60PeriodSMA']:
        if barDataFrame["Orders"][last_order_index] != "SELL":
            return "SELL"
        else:
            return ""
    else:
        return ""
