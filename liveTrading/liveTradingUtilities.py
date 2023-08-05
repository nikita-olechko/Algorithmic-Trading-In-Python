# Create Bracket Order Method
from ibapi.contract import Contract


def create_stock_contract_object(symbol):
    # Initial entry
    contract = Contract()
    contract.symbol = symbol.upper()
    # Set type of object to trade
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def holding_gross_return(barDataFrame, current_average):
    if len(barDataFrame["Average"]) > 0:
        return round(current_average / barDataFrame["Average"][0], 8)
    else:
        return 1


def check_current_position(ib, ticker):
    positions = ib.positions()
    for position in positions:
        if position.contract.symbol == ticker:
            return {ticker: {"Position": position.position, "Average Cost": position.avgCost,
                             "Market Value": position.marketPrice}}
    return {}


def get_order_history(ib, ticker):
    order_history = ib.fills()
    order_history_dict = {}
    for order in order_history:
        if order.contract.symbol == ticker:
            order_name = ticker + "-" + str(order.execution.orderId) + "-" + order.execution.time.strftime(
                "%Y-%m-%d-%H:%M:%S")
            order_history_dict[order_name] = {
                "Ticker": order.contract.symbol,
                "Order Id": order.execution.orderId,
                "Order Action": order.execution.side,
                "Order Quantity": order.execution.shares,
                "Order Price": order.execution.price,
                "Order Average Price": order.execution.avgPrice,
                "Order Fill Price": order.execution.price,
                "Order Fill Time": order.execution.time.strftime("%Y-%m-%d-%H:%M:%S"),
                "Order Commission": order.commissionReport.commission}
    return order_history_dict


def get_time_of_day_as_int(time):
    return int(time.strftime("%H%M%S"))