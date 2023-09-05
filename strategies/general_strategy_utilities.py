def profit_taker(barDataFrame, last_order_index, current_index, profit_taker_percentage=0.75):
    """
    A function that sees if we have bounced back more than 0.75% from our 1% drop
    """
    if percentage_price_change_since_last_order(barDataFrame, last_order_index,
                                                current_index) > profit_taker_percentage:
        return True
    else:
        return False


def stop_loss(barDataFrame, last_order_index, current_index, stop_loss_percentage=1):
    """
    A function that sees if we have dropped more than 1% from our buy price
    """
    if percentage_price_change_since_last_order(barDataFrame, last_order_index, current_index) < -stop_loss_percentage:
        return True
    else:
        return False


def percentage_price_change_since_last_order(barDataFrame, last_order_index, current_index):
    return (barDataFrame.loc[barDataFrame.index[current_index], 'Average'] - barDataFrame.loc[
        barDataFrame.index[last_order_index],
        'Average']) / barDataFrame.loc[barDataFrame.index[current_index], 'Average'] * 100
