import math


def calculate_log_price(barDataFrame):
    barDataFrame['LogPrice'] = barDataFrame['Average'].apply(lambda x: math.log(x))
    return barDataFrame


def generate_change_in_log_price_shift_X(barDataFrame, x=1):
    barDataFrame[f'ShiftedLog{x}Price'] = barDataFrame['LogPrice'].shift(x)
    barDataFrame[f'ChangeIn{x}PeriodLogPrice'] = barDataFrame['LogPrice'] - barDataFrame[f'ShiftedLog{x}Price']
    barDataFrame[f'PercentageChangeIn{x}PeriodLogPrice'] = barDataFrame[f'ChangeIn{x}PeriodLogPrice'] \
                                                           / barDataFrame['LogPrice'].shift(x) * 100
    return barDataFrame


def calculate_change_in_price(barDataFrame, x=1):
    barDataFrame[f'Shifted{x}Price'] = barDataFrame['Average'].shift(x)
    barDataFrame[f'ChangeIn{x}PeriodPrice'] = barDataFrame['Average'] - barDataFrame[f'Shifted{x}Price']
    barDataFrame[f'PercentageChangeIn{x}PeriodPrice'] = barDataFrame[f'ChangeIn{x}PeriodPrice'] \
                                                        / barDataFrame['Average'].shift(x) * 100
    return barDataFrame


def generate_price_shifts_backtest(barDataFrame):
    # barDataFrame = calculate_log_price(barDataFrame)
    # barDataFrame = generate_change_in_log_price_shift_X(barDataFrame, x=10)
    barDataFrame = calculate_change_in_price(barDataFrame, x=10)
    return barDataFrame


def profit_taker(barDataFrame, last_order_index, current_index):
    """
    A function that sees if we have bounced back more than 0.75% from our 1% drop
    """
    if percentage_price_change_since_last_order(barDataFrame, last_order_index, current_index) > 0.75:
        return True
    else:
        return False


def stop_loss(barDataFrame, last_order_index, current_index):
    """
    A function that sees if we have dropped more than 1% from our buy price
    """
    if percentage_price_change_since_last_order(barDataFrame, last_order_index, current_index) < -1:
        return True
    else:
        return False


def percentage_price_change_since_last_order(barDataFrame, last_order_index, current_index):
    return (barDataFrame.loc[barDataFrame.index[current_index], 'Average'] - barDataFrame.loc[
        barDataFrame.index[last_order_index],
        'Average']) / barDataFrame.loc[barDataFrame.index[current_index], 'Average'] * 100


def log_price_change_strategy(barDataFrame, last_order_index=0, current_index=-1, ticker=None):
    """
    A strategy that searches for periods where the price has gone down > 1% in 10 periods
    """
    if barDataFrame["Orders"][last_order_index] == 1:
        if profit_taker(barDataFrame, last_order_index, current_index):
            return -1
        if stop_loss(barDataFrame, last_order_index, current_index):
            return -1
        return 2
    if barDataFrame.loc[barDataFrame.index[current_index], 'PercentageChangeIn10PeriodPrice'] < -1:
        return 1
    return 0
