# Imports
import os
import sys

# Necessary to run project as a scheduled Batch File, DO NOT DELETE
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from ib_insync import OrderState
from ibapi.client import EClient
from ibapi.common import OrderId
from ibapi.order import Order
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import pandas as pd
import pytz
import math
from datetime import datetime
import time

from liveTrading.customOrders import marketBuyOrder, marketSellOrder
from utilities.dataGenerationUtilities import average_bars_by_minute
from utilities.generalUtilities import get_starter_order_id, get_tws_connection_id
from liveTrading.liveTradingUtilities import create_stock_contract_object, holding_gross_return, \
    calculate_current_return
from strategies.greaterthan60barsma import generate60PeriodSMALastRow, sampleSMABuySellStrategy, generate60PeriodSMAWholeDataFrame


# TODO: Set up IBController to Run TWS Automatically (including Login and shutdown)
# TODO: Record a video to demo project outside of market hours for interviews
# TODO: Take into account current position at start of day
# TODO: Document trading returns (on EOD and on Keyboard interruption OR on loop completion)
# TODO: Incorporate greaterThanXPercentJump strategy into random forest model strategy

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


# Bot Logic
class Bot:
    ib = None
    reqId = 0
    initialbartime = datetime.now().astimezone(pytz.timezone("America/New_York"))

    def __init__(self, symbol, buySellConditionFunc, quantity=1, generateNewDataFunc=None, last_row_only=False,
                 periods_to_analyze=50, operate_on_minute_data=False):
        # Connect to IB on init
        twsConnectionID = get_tws_connection_id()
        orderIDStarter = get_starter_order_id()
        self.ib = IBApi(self)
        self.ib.connect("127.0.0.1", 4000, twsConnectionID)
        # Listen to socket on another thread
        while not self.ib.isConnected():
            twsConnectionID += 1
            self.ib.connect("127.0.0.1", 4000, twsConnectionID)
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        time.sleep(1)
        self.symbol = symbol.upper()
        self.buySellConditionFunc = buySellConditionFunc
        self.generateNewDataFunc = generateNewDataFunc
        self.quantity = quantity
        self.last_row_only = last_row_only
        self.periods_to_analyze = periods_to_analyze
        self.operate_on_minute_data = operate_on_minute_data
        self.last_order_index = 0
        self.completedOrders = 0
        self.orderId = orderIDStarter
        self.openOrders = self.get_open_orders()
        data_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Average", "BarCount", "Orders",
                        "HoldingGrossReturn"]
        self.barDataFrame = pd.DataFrame(columns=data_columns)
        self.minuteDataFrame = pd.DataFrame()
        # Get Bar Size
        self.barsize = "1 min"
        # Create IB Contract Object
        contract = Contract()
        contract.symbol = self.symbol.upper()
        # Set type of object to trade
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        self.ib.reqIds(-1)
        # Request Market Data
        # self.ib.reqRealTimeBars(0, contract, 5, "TRADES", useRTH=True, realTimeBarsOptions=[])
        self.ib.reqHistoricalData(self.reqId, contract, "", "1 D", self.barsize, "TRADES", 1, 1, True, [])

    # Listen to socket
    def run_loop(self):
        self.ib.run()

    # Retrieve any open orders
    def get_open_orders(self):
        self.ib.reqAllOpenOrders()
        return self.ib.openOrders

    def place_orders(self, order_bracket, contract, oca=False):
        # Place the Bracket Order
        if type(order_bracket) != list:
            order_bracket = [order_bracket]
        for order in order_bracket:
            # One Cancels All
            if oca:
                order.ocaGroup = "OCA_" + str(self.orderId)
            self.ib.placeOrder(order.orderId, contract, order)
        self.orderId = self.orderId + len(order_bracket)
        print(
            f"{len(order_bracket)} order(s) placed. Next order id: "
            f"{self.orderId}. Order Details: \n{order_bracket}")

    def createOrderColumnLatestOrder(self):
        """
        A method to determine whether an order should be placed or not based on a conditional strategy function
        """
        if self.barDataFrame["Orders"][self.last_order_index] != 1 and self.buySellConditionFunc(
                self.barDataFrame, self.last_order_index, self.symbol) == 1:
            self.barDataFrame.at[len(self.barDataFrame) - 1, "Orders"] = 1
        elif self.barDataFrame["Orders"][self.last_order_index] != -1 and self.buySellConditionFunc(
                self.barDataFrame, self.last_order_index, self.symbol) == -1:
            self.barDataFrame.at[len(self.barDataFrame) - 1, "Orders"] = -1
        else:
            pass

    def place_orders_if_needed(self):
        """
        A method to place orders based on the "Orders" column in self.barDataFrame. This method is separate
        from createOrderColumnLatestOrder to support advanced order routing in the future (e.g. conditional
        bracket orders). Any type of order can be added for future functionality (e.g., bracket and limit orders)
        """
        if self.barDataFrame.at[len(self.barDataFrame) - 1, "Orders"] == 1:
            contract = create_stock_contract_object(self.symbol)
            market_buy_order = marketBuyOrder(self.orderId, quantity=self.quantity)
            self.place_orders(market_buy_order, contract)
            self.last_order_index = len(self.barDataFrame) - 1

        if self.barDataFrame.at[len(self.barDataFrame) - 1, "Orders"] == -1:
            contract = create_stock_contract_object(self.symbol)
            market_sell_order = marketSellOrder(self.orderId, quantity=self.quantity)
            self.place_orders(market_sell_order, contract)
            self.last_order_index = len(self.barDataFrame) - 1

    # Pass realtimebar data to our bot object
    def on_bar_update(self, reqId, bar, realtime):
        # Historical Data to Catch Up
        bar_row = {"Date": bar.date, "Open": bar.open, "High": bar.high, "Low": bar.low, "Volume": bar.volume,
                   "Close": bar.close, "Average": bar.average, "BarCount": bar.barCount, "Orders": "",
                   "HoldingGrossReturn": holding_gross_return(self.barDataFrame, bar.average),
                   "Current_Return": calculate_current_return(self.barDataFrame, bar.average, self.last_order_index)}
        self.barDataFrame.loc[len(self.barDataFrame)] = bar_row
        if not realtime:
            print("New Bar Received: ", self.symbol)
            print(bar_row)

            # Switch to bar by bar instead of minute by minute if reqested
            if self.operate_on_minute_data:
                self.minuteDataFrame = average_bars_by_minute(self.barDataFrame, self.minuteDataFrame)
            else:
                self.minuteDataFrame = self.barDataFrame.copy()

            # Switch to calculate only the last row if higher performance is requested
            if self.last_row_only:
                if self.generateNewDataFunc is not None:
                    self.minuteDataFrame = self.generateNewDataFunc(self.minuteDataFrame)

            # Otherwise, calculate new data for a slice of the dataframe and concatenate together
            else:
                if self.generateNewDataFunc is not None:
                    last_periods = self.minuteDataFrame.tail(self.periods_to_analyze).copy()
                    last_periods = self.generateNewDataFunc(last_periods)
                    last_row = last_periods.iloc[-1]
                    self.minuteDataFrame = self.minuteDataFrame.iloc[:-1]
                    self.minuteDataFrame = pd.concat([self.minuteDataFrame, pd.DataFrame([last_row])],
                                                     ignore_index=False)
            self.minuteDataFrame.at[len(self.minuteDataFrame) - 1, "Orders"] = self.buySellConditionFunc(
                self.minuteDataFrame, self.last_order_index, self.symbol)
            self.place_orders_if_needed()


# Start Bot(s)
bot1 = Bot(symbol="AAPL", quantity=1, buySellConditionFunc=sampleSMABuySellStrategy,
           generateNewDataFunc=generate60PeriodSMAWholeDataFrame)
