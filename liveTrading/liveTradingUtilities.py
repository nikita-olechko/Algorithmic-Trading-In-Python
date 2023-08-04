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
        if ticker is not None:
            if position.contract.symbol == ticker:
                return {ticker: {"Position": position.position, "Average Cost": position.avgCost,
                                 "Market Value": position.marketPrice}}
    return {}


def calculate_current_return(barDataFrame, current_average, last_order_index):
    pass
