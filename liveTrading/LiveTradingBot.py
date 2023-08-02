# Imports
import os
import sys
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

from utilities.dataGenerationUtilities import average_bars_by_minute
from utilities.generalUtilities import get_starter_order_id
from liveTrading.liveTradingUtilities import create_stock_contract_object, marketBuyOrder, marketSellOrder, \
    holding_gross_return
from strategies.greaterthan60barsma import generate60PeriodSMA, sampleSMABuySellStrategy

# Necessary to run project as a scheduled Batch File, DO NOT DELETE
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)


# TODO: Set up IBController to Run TWS Automatically (including Login and shutdown)
# TODO: Document trading returns (on EOD and on Keyboard interruption OR on loop completion)
# TODO: Research some new strategies
# TODO: Implement some new strategies
# TODO: Record a video to demo project outside of market hours for interviews
# TODO: Take into account current position at start OR sell off positions at end for day trading,
#  make customizable for strategy
# TODO: Write a readme for backtesting
# TODO: Split Github Repos into testing and live trading [In Progress]
# TODO: Create Summary statistics for backtesting [Done]
# TODO: Write backtest data somewhere [Done]
# TODO: Transform data to have minute-by-minute analysis [Done]
# TODO: Refactor Global variables to exist within classes so code is more modular [Done]
# TODO: Trade multiple tickers/strategies at once [Done]
# TODO: Code a sample strategy to test new system on [Done]
# TODO: Set up system to run at 6:35 am [Done]
# TODO: Change IBKR restart time to after market close [Done]


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

    def __init__(self, symbol, buySellConditionFunc, quantity=1, generateNewDataFunc=None, twsConnectionID=1,
                 orderIDStarter=get_starter_order_id(0)):
        # Connect to IB on init
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
        # Request all open orders
        self.ib.reqAllOpenOrders()

        # Wait for a short while to allow the request to be processed
        time.sleep(1)

        return self.ib.openOrders

    # Create Bracket Order Method
    def bracketOrder(self, action, quantity, profitTarget, stopLoss):
        # Initial entry
        contract = Contract()
        contract.symbol = self.symbol.upper()
        # Set type of object to trade
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        # Create Parent Order / Initial Entry
        parent = Order()
        parent.orderId = self.orderId
        parent.orderType = "MTK"
        parent.action = action
        parent.totalQuantity = quantity
        parent.transmit = False
        parent.eTradeOnly = False
        parent.firmQuoteOnly = False
        # Profit Target Order
        profitTargetOrder = Order()
        profitTargetOrder.orderId = self.orderId + 1
        profitTargetOrder.orderType = "LMT"
        profitTargetOrder.action = "-1"
        profitTargetOrder.totalQuantity = quantity
        profitTargetOrder.lmtPrice = round(profitTarget, 2)
        profitTargetOrder.transmit = True
        profitTargetOrder.eTradeOnly = False
        profitTargetOrder.firmQuoteOnly = False
        # Stop Loss Order
        stopLossOrder = Order()
        stopLossOrder.orderId = self.orderId + 2
        stopLossOrder.orderType = "STP"
        stopLossOrder.action = "-1"
        stopLossOrder.totalQuantity = quantity
        stopLossOrder.auxPrice = round(stopLoss, 2)
        stopLossOrder.transmit = True
        stopLossOrder.eTradeOnly = False
        stopLossOrder.firmQuoteOnly = False

        # Bracket Orders Array
        bracketOrders = [parent, profitTargetOrder, stopLossOrder]
        return bracketOrders

    def place_orders(self, order_bracket, contract, oca=False):
        # Place the Bracket Order
        with threading.Lock():
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
                self.barDataFrame) == 1:
            self.barDataFrame.at[len(self.barDataFrame) - 1, "Orders"] = 1
        elif self.barDataFrame["Orders"][self.last_order_index] != -1 and self.buySellConditionFunc(
                self.barDataFrame) == -1:
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
            # with threading.Lock():
            contract = create_stock_contract_object(self.symbol)
            market_buy_order = marketBuyOrder(self.orderId, quantity=self.quantity)
            self.place_orders(market_buy_order, contract)
            self.last_order_index = len(self.barDataFrame) - 1

        if self.barDataFrame.at[len(self.barDataFrame) - 1, "Orders"] == -1:
            # with threading.Lock():
            contract = create_stock_contract_object(self.symbol)
            market_sell_order = marketSellOrder(self.orderId, quantity=self.quantity)
            self.place_orders(market_sell_order, contract)
            self.last_order_index = len(self.barDataFrame) - 1

    # Pass realtimebar data to our bot object
    def on_bar_update(self, reqId, bar, realtime):
        # Historical Data to Catch Up
        bar_row = {"Date": bar.date, "Open": bar.open, "High": bar.high, "Low": bar.low, "Volume": bar.volume,
                   "Close": bar.close, "Average": bar.average, "BarCount": bar.barCount, "Orders": "",
                   "HoldingGrossReturn": holding_gross_return(self.barDataFrame, bar.average)}
        self.barDataFrame.loc[len(self.barDataFrame)] = bar_row
        # self.bars.append(bar)
        if realtime:
            # self.ib.reqOptionChain(self.reqId, self.symbol)
            print("New Bar Received: ", self.symbol)
            print(bar_row)
            bartime = datetime.strptime(bar.date, "%Y%m%d %H:%M:%S").astimezone(pytz.timezone("America/New_York"))
            minutes_diff = (bartime - self.initialbartime).total_seconds() / 60.0
            # On Bar Close, after a minute has passed
            if minutes_diff > 0 and math.floor(minutes_diff) % int(self.barsize.split()[0]) == 0:
                self.minuteDataFrame = average_bars_by_minute(self.barDataFrame, self.minuteDataFrame)
                if self.generateNewDataFunc is not None:
                    self.barDataFrame = self.generateNewDataFunc(self.barDataFrame)
                self.barDataFrame.at[len(self.barDataFrame) - 1, "Orders"] = self.buySellConditionFunc(self.barDataFrame)
                self.place_orders_if_needed()


# Start Bot(s)
bot1 = Bot(symbol="XOM", quantity=1, buySellConditionFunc=sampleSMABuySellStrategy,
           generateNewDataFunc=generate60PeriodSMA, twsConnectionID=1, orderIDStarter=get_starter_order_id(0))
# bot2 = Bot(symbol="XOM", quantity=2, buySellConditionFunc=sampleSMABuySellStrategy,
#            generateNewDataFunc=generate60PeriodSMA, twsConnectionID=2, orderIDStarter=get_starter_order_id(1))
