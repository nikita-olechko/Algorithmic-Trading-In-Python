from ib_insync import OrderState
from ibapi.contract import Contract
from ibapi.client import EClient
from ibapi.common import OrderId
from ibapi.order import Order
from ibapi.wrapper import EWrapper


# Class for interactive brokers connection within Bot
class IBApi(EWrapper, EClient):
    def __init__(self, bot):
        EClient.__init__(self, self)
        self.bot = bot
        self.openOrders = []

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState: OrderState):
        self.openOrders.append((orderId, contract, order))

    def openOrderEnd(self):
        print("Received all open orders")

    def historicalData(self, reqId, bar):
        try:
            self.bot.on_bar_update(reqId, bar, False)
        except Exception as e:
            print(e)

    def historicalDataUpdate(self, reqId, bar):
        try:
            self.bot.on_bar_update(reqId, bar, True)
        except Exception as e:
            print(e)

    # On historical data end
    def historicalDataEnd(self, reqId, start, end):
        print("End of Historical Data\n\n")

    # Get next Order ID
    def nextValidId(self, nextOrderId: int):
        order_id = nextOrderId
        return order_id

    # Request Option Chain function
    def reqOptionChain(self, reqId, symbol):
        self.reqSecDefOptParams(reqId, symbol, "", "STK", 0)

    # Handle output of option chain
    def secDefOptParams(self, reqId, exchange, underlyingConId, underlyingSymbol, futFopExchange, underlyingSecType,
                        multiplier, expirations, strikes):
        print("Exchange:", exchange, "Underlying conId:", underlyingConId, "Symbol:", underlyingSymbol,
              "FUT-FOP exchange:", futFopExchange, "Underlying secType:", underlyingSecType,
              "Multiplier:", multiplier, "Expirations:", expirations, "Strikes:", strikes)
