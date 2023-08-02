def order_selector(order):
    order = str(order)
    order_dict = {
        "1": buy_order,
        "-1": sell_order,
        "2": hold_order,
        "0": nothing_order
    }
    return order_dict[order]


def buy_order(stk_data, index, last_order_index, ticker_wap):
    stk_data.at[index, 'Orders'] = 1
    last_order_index = index
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position'] - stk_data.at[index, ticker_wap]
    return stk_data, last_order_index


def sell_order(stk_data, index, last_order_index, ticker_wap):
    stk_data.at[index, 'Orders'] = -1
    last_order_index = index
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position'] + stk_data.at[index, ticker_wap]
    return stk_data, last_order_index


def hold_order(stk_data, index, last_order_index):
    stk_data.at[index, 'Orders'] = 2
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position']
    return stk_data, last_order_index


def nothing_order(stk_data, index, last_order_index):
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position']
    return stk_data, last_order_index
