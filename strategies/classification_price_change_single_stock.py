import pickle

import numpy as np
import pandas as pd

from utilities.general_strategy_utilities import profit_taker, minutes_since_last_order


def create_log_price_variables(stk_data, list_of_periods=range(1, 11)):
    """
    Create log price and related variables for a given DataFrame.

    :param stk_data: DataFrame containing stock data.
    :param list_of_periods: List of periods to calculate shifted log prices.
    :return: Modified DataFrame with log price variables.
    """
    log_price = np.log(stk_data["Average"])
    stk_data["log_price"] = log_price

    for period in list_of_periods:
        shifted_log_price = stk_data["log_price"].shift(period)
        stk_data[f'{period}period_shifted_log_price'] = shifted_log_price
        stk_data[f'{period}period_change_in_log_price'] = stk_data["log_price"] - shifted_log_price
        stk_data[f'{period}period_percentage_change_in_log_price'] = (
                stk_data[f'{period}period_change_in_log_price'] / shifted_log_price * 100
        )

    return stk_data


def create_price_variables(stk_data, list_of_periods=range(1, 11)):
    """
    Create price change and related variables for a given DataFrame.

    :param stk_data: DataFrame containing stock data.
    :param list_of_periods: List of periods to calculate shifted prices.
    :return: Modified DataFrame with price change variables.
    """
    for period in list_of_periods:
        shifted_price = stk_data["Average"].shift(period)
        stk_data[f'{period}period_shifted_price'] = shifted_price
        stk_data[f'{period}period_change_in_price'] = stk_data["Average"] - shifted_price
        stk_data[f'{period}period_percentage_change_in_price'] = (
                stk_data[f'{period}period_change_in_price'] / shifted_price * 100
        )
        stk_data[f'sum_of_absolute_percentage_price_changes_over_{period}_periods'] = stk_data[
            f'{period}period_percentage_change_in_price'].abs().rolling(window=period).sum()
    return stk_data


def create_volume_change_variables(stk_data, list_of_periods=range(1, 11)):
    """
    Create log volume and related variables for a given DataFrame.

    :param stk_data: DataFrame containing stock data.
    :param list_of_periods: List of periods to calculate shifted log volumes.
    :return: Modified DataFrame with log volume variables.
    """
    log_volume = np.log(stk_data["Volume"])
    stk_data["log_volume"] = log_volume
    for period in list_of_periods:
        shifted_log_volume = stk_data["log_volume"].shift(period)
        stk_data[f'{period}period_shifted_log_volume'] = shifted_log_volume
        stk_data[f'{period}period_change_in_log_volume'] = stk_data["log_volume"] - shifted_log_volume
    return stk_data


def generate_bollinger_bands(dataFrame, period=20):
    """
    Generate Bollinger Bands based on moving averages and standard deviations.

    :param dataFrame: DataFrame containing stock data.
    :param period: Period for calculating moving averages and standard deviations.
    :return: DataFrame with Bollinger Bands columns added.
    """
    # Calculate the moving average and standard deviation over the last 'period' rows
    dataFrame['MA_20'] = dataFrame['Average'].rolling(window=period).mean()
    dataFrame['SD_20'] = dataFrame['Average'].rolling(window=period).std()

    # Calculate the Bollinger Bands
    dataFrame['UpperBB2SD'] = dataFrame['MA_20'] + 2 * dataFrame['SD_20']
    dataFrame['LowerBB2SD'] = dataFrame['MA_20'] - 2 * dataFrame['SD_20']
    dataFrame['UpperBB1SD'] = dataFrame['MA_20'] + dataFrame['SD_20']
    dataFrame['LowerBB1SD'] = dataFrame['MA_20'] - dataFrame['SD_20']

    return dataFrame


def boolean_bollinger_band_location(minuteDataFrame):
    """
    Determine whether prices are above or below Bollinger Bands.

    :param minuteDataFrame: DataFrame containing stock data.
    :return: DataFrame with Boolean columns indicating Bollinger Bands positions.
    """
    minuteDataFrame['PriceAboveUpperBB2SD'] = np.where(minuteDataFrame['Average'] > minuteDataFrame['UpperBB2SD'], 1, 0)
    minuteDataFrame['PriceAboveUpperBB1SD'] = np.where(
        (minuteDataFrame['Average'] > minuteDataFrame['UpperBB1SD']) & (minuteDataFrame['PriceAboveUpperBB2SD'] == 0),
        1, 0)
    minuteDataFrame['PriceBelowLowerBB2SD'] = np.where(minuteDataFrame['Average'] < minuteDataFrame['LowerBB2SD'], 1, 0)
    minuteDataFrame['PriceBelowLowerBB1SD'] = np.where(
        (minuteDataFrame['Average'] < minuteDataFrame['LowerBB1SD']) & (minuteDataFrame['PriceBelowLowerBB2SD'] == 0),
        1, 0)
    return minuteDataFrame


def predict_based_on_model(barDataFrame, model_object):
    x_columns = list(barDataFrame.columns)

    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount', 'Orders',
                                'Position']

    for column in always_redundant_columns:
        x_columns.remove(column)

    data = barDataFrame.dropna()

    x_test = data[x_columns]

    predict_price_change = model_object.predict(x_test)
    predict_price_change = predict_price_change.reshape(-1, 1)
    predict_price_change = pd.DataFrame(predict_price_change, columns=['Model_Prediction'])
    barDataFrame['Model_Prediction'] = predict_price_change
    return barDataFrame


def generate_model_data(barDataFrame, model_object, Z_periods=120, periodicity=1):
    barDataFrame = create_log_price_variables(barDataFrame, list_of_periods=range(1, Z_periods, periodicity))
    barDataFrame = create_price_variables(barDataFrame, list_of_periods=range(1, Z_periods, periodicity))
    barDataFrame = create_volume_change_variables(barDataFrame, list_of_periods=range(1, Z_periods, periodicity))
    barDataFrame = generate_bollinger_bands(barDataFrame)
    barDataFrame = boolean_bollinger_band_location(barDataFrame)
    barDataFrame = predict_based_on_model(barDataFrame, model_object)
    return barDataFrame


def classification_model_strategy(barDataFrame, last_order_index=0, current_index=-1, ticker=None):
    """
    A strategy that searches for periods where the price has gone down > 1% in 10 periods
    """
    if barDataFrame["Orders"][last_order_index] == 1:
        if profit_taker(barDataFrame, last_order_index, current_index, 1.5):
            return -1
        if minutes_since_last_order(barDataFrame, last_order_index, current_index) > 120:
            return -1
        return 2
    if barDataFrame.loc[barDataFrame.index[current_index], 'Prediction'] == 1:
        return 1
    return 0
