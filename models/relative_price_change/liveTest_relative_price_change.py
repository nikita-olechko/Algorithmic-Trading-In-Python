import joblib
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
import numpy as np
import pytz
import math
from datetime import datetime
import time

from utilities.dataGenerationUtilities import average_bars_by_minute
from utilities.generalUtilities import get_tws_connection_id, get_starter_order_id
from utilities.ibkrUtilities import IBApi


def create_log_price_variables_last_row(stk_data, list_of_periods=range(1, 11)):
    log_price = np.log(stk_data["Average"].iloc[-1])
    stk_data.at[stk_data.index[-1], "log_price"] = log_price
    for period in list_of_periods:
        shifted_log_price = stk_data["log_price"].iloc[-period - 1]
        stk_data.at[stk_data.index[-1], f'{period}period_shifted_log_price'] = shifted_log_price
        stk_data.at[stk_data.index[-1], f'{period}period_change_in_log_price'] = log_price - shifted_log_price
    return stk_data


def create_volume_change_variables_last_row(stk_data, list_of_periods=range(1, 11)):
    log_volume = np.log(stk_data["Volume"].iloc[-1])
    stk_data.at[stk_data.index[-1], "log_volume"] = log_volume
    # Note: volume is actually upshifted one period here so all the names are one off
    # but the model is built and works with it so whatever: 1period_change_in_log_volume is always 0,
    # 2period_change_in_log_volume is actually 1period_change_in_log_volume. Etc.
    # NOTE. THIS ^^^ DOES MATTER. FIX IT TO BE ACCURATE BECAUSE IT IS ACCURATE IN THE MODEL AND THUS WRONG.
    # NOTE: Reference x_test to see the variables that the model is created from and make sure they are the same
    for period in list_of_periods:
        shifted_log_volume = stk_data["log_volume"].iloc[-period - 1]
        stk_data.at[stk_data.index[-1], f'{period}period_shifted_log_volume'] = shifted_log_volume
        stk_data.at[stk_data.index[-1], f'{period}period_change_in_log_volume'] = log_volume - shifted_log_volume
    return stk_data


def generate_bollinger_bands_last_row(dataFrame, period=20):
    last_rows = dataFrame.iloc[-period:]
    dataFrame.at[dataFrame.index[-1], 'MA_20'] = last_rows['Average'].mean()
    dataFrame.at[dataFrame.index[-1], 'SD_20'] = last_rows['Average'].std()
    dataFrame.at[dataFrame.index[-1], 'UpperBB2SD'] = dataFrame.at[dataFrame.index[-1], 'MA_20'] + 2 * dataFrame.at[
        dataFrame.index[-1], 'SD_20']
    dataFrame.at[dataFrame.index[-1], 'LowerBB2SD'] = dataFrame.at[dataFrame.index[-1], 'MA_20'] - 2 * dataFrame.at[
        dataFrame.index[-1], 'SD_20']
    dataFrame.at[dataFrame.index[-1], 'UpperBB1SD'] = dataFrame.at[dataFrame.index[-1], 'MA_20'] + dataFrame.at[
        dataFrame.index[-1], 'SD_20']
    dataFrame.at[dataFrame.index[-1], 'LowerBB1SD'] = dataFrame.at[dataFrame.index[-1], 'MA_20'] - dataFrame.at[
        dataFrame.index[-1], 'SD_20']
    return dataFrame


def boolean_bollinger_band_location_last_row(minuteDataFrame):
    last_row = minuteDataFrame.iloc[-1]
    minuteDataFrame.at[minuteDataFrame.index[-1], 'PriceAboveUpperBB2SD'] = 1 if last_row['Average'] > last_row[
        'UpperBB2SD'] else 0
    minuteDataFrame.at[minuteDataFrame.index[-1], 'PriceAboveUpperBB1SD'] = 1 if (last_row['Average'] > last_row[
        'UpperBB1SD']) and (minuteDataFrame.at[minuteDataFrame.index[-1], 'PriceAboveUpperBB2SD'] == 0) else 0
    minuteDataFrame.at[minuteDataFrame.index[-1], 'PriceBelowLowerBB2SD'] = 1 if last_row['Average'] < last_row[
        'LowerBB2SD'] else 0
    minuteDataFrame.at[minuteDataFrame.index[-1], 'PriceBelowLowerBB1SD'] = 1 if (last_row['Average'] < last_row[
        'LowerBB1SD']) and (minuteDataFrame.at[minuteDataFrame.index[-1], 'PriceBelowLowerBB2SD'] == 0) else 0
    return minuteDataFrame


def last_period_change_in_log_price(minuteDataFrame):
    current_price = np.log(minuteDataFrame.at[minuteDataFrame.index[-1], 'Average'])
    previous_price = np.log(minuteDataFrame.at[minuteDataFrame.index[-2], 'Average'])
    minuteDataFrame.at[minuteDataFrame.index[-2], 'ActualPriceChange'] = current_price - previous_price
    minuteDataFrame.at[minuteDataFrame.index[-1], 'ActualPriceChange'] = np.nan
    return minuteDataFrame


def generate_model_data(minuteDataFrame, symbol, model):
    minuteDataFrame = create_log_price_variables_last_row(minuteDataFrame)
    minuteDataFrame = create_volume_change_variables_last_row(minuteDataFrame)
    minuteDataFrame = generate_bollinger_bands_last_row(minuteDataFrame)
    minuteDataFrame = boolean_bollinger_band_location_last_row(minuteDataFrame)
    minuteDataFrame = last_period_change_in_log_price(minuteDataFrame)
    minuteDataFrame.at[minuteDataFrame.index[-1], 'PredictedPriceChange'] = 0
    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'BarCount', 'Orders']
    additional_non_model_columns = ['ActualPriceChange', 'PredictedPriceChange']
    model_filename = f'model_objects/{model}.pkl'
    x_columns = list(minuteDataFrame.columns)
    for column in always_redundant_columns + additional_non_model_columns:
        x_columns.remove(column)
    with open(model_filename, 'rb') as file:
        loaded_lm = joblib.load(file)
        predicted_price = loaded_lm.predict(minuteDataFrame[x_columns].iloc[-1].values.reshape(1, -1))
    minuteDataFrame.at[minuteDataFrame.index[-1], 'PredictedPriceChange'] = predicted_price
    return minuteDataFrame

class ModelAccuracyBot:
    ib = None
    reqId = 0
    initialbartime = datetime.now().astimezone(pytz.timezone("America/New_York"))

    def __init__(self, symbol, model, generateNewDataFunc=None):
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
        self.model = model
        self.generateNewDataFunc = generateNewDataFunc
        self.last_order_index = 0
        self.completedOrders = 0
        self.orderId = orderIDStarter
        data_columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Average", "BarCount", "Orders"]
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
        self.ib.reqHistoricalData(self.reqId, contract, "", "1 D", self.barsize, "TRADES", 1, 1, True, [])

    # Listen to socket
    def run_loop(self):
        self.ib.run()

    # Pass realtimebar data to our bot object
    def on_bar_update(self, reqId, bar, realtime):
        # Historical Data to Catch Up
        bar_row = {"Date": bar.date, "Open": bar.open, "High": bar.high, "Low": bar.low, "Volume": bar.volume,
                   "Close": bar.close, "Average": bar.average, "BarCount": bar.barCount, "Orders": ""}
        self.barDataFrame.loc[len(self.barDataFrame)] = bar_row
        # Note: minuteDataFrame is above realtime condition to ensure historical data is in minuteDataFrame.
        # We do not want this in liveTrading - this technically performs redundant operations but is insignificant for testing.
        self.minuteDataFrame = average_bars_by_minute(self.barDataFrame, self.minuteDataFrame)

        if realtime:
            # self.ib.reqOptionChain(self.reqId, self.symbol)
            print("New Bar Received: ", self.symbol)
            print(bar_row)
            # On Bar Close, after a minute has passed
            if self.generateNewDataFunc is not None:
                self.minuteDataFrame = self.generateNewDataFunc(self.minuteDataFrame, self.symbol, self.model)
                self.minuteDataFrame.to_csv(f"liveTests/liveTest_{self.model}_{self.symbol}.csv")


# bot1 = ModelAccuracyBot(symbol="XOM", model="relative_price_change",
#                         generateNewDataFunc=generate_model_data)

for ticker in ["XOM", "NVDA", "AMD"]:
    bot = ModelAccuracyBot(symbol=ticker, model=f"relative_price_change_linear_model_{ticker}_5mins_12M",
                            generateNewDataFunc=generate_model_data)
    bot2 = ModelAccuracyBot(symbol=ticker, model=f"relative_price_change_random_forest_model_{ticker}_5mins_12M",
                            generateNewDataFunc=generate_model_data)
    bot3 = ModelAccuracyBot(symbol=ticker, model=f"relative_price_change_mlp_model_{ticker}_5mins_12M",
                            generateNewDataFunc=generate_model_data)

