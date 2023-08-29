# Imports
import os
import sys

from ibapi.contract import Contract

from utilities.ibkrUtilities import IBApi

# Necessary to run project as a scheduled Batch File, DO NOT DELETE
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import threading
import pandas as pd
import pytz
from datetime import datetime
import time

from liveTrading.customOrders import marketBuyOrder, marketSellOrder
from utilities.dataGenerationUtilities import average_bars_by_minute
from utilities.generalUtilities import get_starter_order_id, get_tws_connection_id
from liveTrading.liveTradingUtilities import create_stock_contract_object, holding_gross_return, \
    calculate_current_return
from strategies.greaterthan60barsma import sampleSMABuySellStrategy, generate60PeriodSMAWholeDataFrame


# Bot Logic
class Bot:
    """
        A trading bot that interacts with the Interactive Brokers (IB) trading platform to execute buy and sell orders.

        :param symbol: The trading symbol (stock symbol) for the financial instrument that the bot will trade.
        :type symbol: str

        :param buySellConditionFunc: A function defining the conditions for placing buy or sell orders.
            This function takes the bar data, last order index, and symbol as parameters and returns a signal for placing an order
            (1 for buy, -1 for sell, 2 for hold, or 0 for no action).
        :type buySellConditionFunc: callable

        :param quantity: The quantity of shares to be traded in each order. Default is 1.
        :type quantity: int, optional

        :param generateNewDataFunc: A function that generates additional data for analysis.
        :type generateNewDataFunc: callable, optional

        :param last_row_only: Whether to operate on the last row only for higher performance. Default is False.
        :type last_row_only: bool, optional

        :param periods_to_analyze: The number of periods to analyze when generating new data or making trading decisions.
        :type periods_to_analyze: int, only used if last_row_only is False

        :param operate_on_minute_data: Whether to operate on minute data or not. Default is True.
        :type operate_on_minute_data: bool, optional
        """
    ib = None
    reqId = 0
    initialbartime = datetime.now().astimezone(pytz.timezone("America/New_York"))

    def __init__(self, symbol: str, buySellConditionFunc: callable, quantity: int = 1,
                 generateNewDataFunc: callable = None,
                 last_row_only: bool = False,
                 periods_to_analyze: int = 50, operate_on_minute_data: bool = True):
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
        if realtime:
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
        else:
            self.minuteDataFrame = self.barDataFrame
