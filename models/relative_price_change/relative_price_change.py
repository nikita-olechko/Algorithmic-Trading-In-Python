import pickle

import numpy as np
import pandas as pd
from sklearn import (
    linear_model
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor

from backtesting.backtestingUtilities.simulationUtilities import get_stock_data
from utilities.generalUtilities import initialize_ib_connection


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
        stk_data[f'{period}period_shifted_log_price'] = stk_data["log_price"].shift(period)
        stk_data[f'{period}period_change_in_log_price'] = stk_data["log_price"] - stk_data[
            f'{period}period_shifted_log_price']
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
        stk_data[f'{period}period_shifted_log_volume'] = stk_data["log_volume"].shift(period)
        stk_data[f'{period}period_change_in_log_volume'] = stk_data["log_volume"] - stk_data[
            f'{period}period_shifted_log_volume']
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


def SD_correct_direction(actual, predicted, condition_series):
    """
    Determine if the price change and prediction have the same direction based on a condition.

    :param actual: Actual price change.
    :param predicted: Predicted price change.
    :param condition_series: Condition series indicating a specific situation.
    :return: 1 if directions match, 0 if they do not, or NaN if not applicable.
    """
    if condition_series == 1:
        if actual * predicted >= 0:
            return 1
        else:
            return 0
    else:
        return np.nan


def prepare_training_data(data, barsize, duration, endDateTime):
    """
    Prepare training data for machine learning models.

    :param data: DataFrame containing stock data.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param endDateTime: End date and time for the data.
    :return: Processed data, feature columns, and target column.
    """
    stk_data = data
    if data is None:
        stk_data = get_stock_data("XOM", barsize, duration, directory_offset=2, endDateTime=endDateTime)
    stk_data = create_log_price_variables(stk_data)
    stk_data['NextPeriodChangeInLogPrice'] = stk_data['log_price'].shift(-1) - stk_data['log_price']
    stk_data = create_volume_change_variables(stk_data)
    stk_data = generate_bollinger_bands(stk_data)
    stk_data = boolean_bollinger_band_location(stk_data)

    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount', 'Orders',
                                'Position']
    extra_columns_to_remove = ['NextPeriodChangeInLogPrice']

    x_columns = list(stk_data.columns)
    y_column = 'NextPeriodChangeInLogPrice'

    for column in always_redundant_columns + extra_columns_to_remove:
        x_columns.remove(column)

    data = stk_data.dropna()
    return data, x_columns, y_column


def create_relative_price_change_linear_regression_model(symbol, endDateTime='', save_model=True, barsize="1 min",
                                                         duration="2 M", data=None):
    """
    Create a linear regression model for predicting relative price changes.

    :param symbol: Ticker symbol of the stock.
    :param endDateTime: End date and time for the data formatted as YYYYMMDD HH:MM:SS.
    :param save_model: Whether to save the trained model.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param data: DataFrame containing stock data.
    :return: Trained linear regression model.
    """
    data, x_columns, y_column = prepare_training_data(data, barsize, duration, endDateTime)
    train = data

    X_train = train[x_columns]
    y_train = train[y_column]

    # Create and train the model
    lm = linear_model.LinearRegression()
    lm.fit(X_train, y_train)

    if save_model:
        model_filename = f'model_objects/relative_price_change_linear_model_{symbol}_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(lm, file)
    return lm


def create_relative_price_change_random_forest_model(symbol, endDateTime='', save_model=True, barsize="1 min",
                                                     duration="2 M", data=None):
    """
    Create a random forest model for predicting relative price changes.

    :param symbol: Ticker symbol of the stock.
    :param endDateTime: End date and time for the data.
    :param save_model: Whether to save the trained model.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param data: DataFrame containing stock data.
    :return: Trained random forest model.
    """
    data, x_columns, y_column = prepare_training_data(data, barsize, duration, endDateTime)
    train = data

    x_train = train[x_columns]
    y_train = train[y_column]

    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)

    if save_model:
        model_filename = f'model_objects/relative_price_change_random_forest_model_{symbol}_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(forest, file)
    return forest


def create_relative_price_change_mlp_model(symbol, endDateTime='', save_model=True, barsize="1 min",
                                           duration="2 M", data=None):
    """
    Create a multi-layer perceptron (MLP) model for predicting relative price changes.

    :param symbol: Ticker symbol of the stock.
    :param endDateTime: End date and time for the data.
    :param save_model: Whether to save the trained model.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param data: DataFrame containing stock data.
    :return: Trained MLP model.
    """
    data, x_columns, y_column = prepare_training_data(data, barsize, duration, endDateTime)
    x_train, x_test, y_train, y_test = train_test_split(data[x_columns], data[y_column], test_size=0.2, random_state=42)

    nn_regressor = MLPRegressor(max_iter=1000, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
    }

    grid_search = GridSearchCV(nn_regressor, param_grid, cv=3, scoring='neg_mean_squared_error')

    grid_search.fit(x_train, y_train)

    best_nn_regressor = grid_search.best_estimator_

    if save_model:
        model_filename = f'model_objects/relative_price_change_mlp_model_{symbol}_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(best_nn_regressor, file)

    return best_nn_regressor


def analyze_model_performance(model_object, test_data, additional_columns_to_remove=None):
    """
    Analyze the performance of a predictive model.

    :param model_object: Trained predictive model.
    :param test_data: DataFrame containing test data.
    :param additional_columns_to_remove: Additional columns to remove from analysis.
    :return: DataFrame with analysis results.
    """
    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount', 'Orders',
                                'Position']
    extra_columns_to_remove = ['NextPeriodChangeInLogPrice']

    x_columns = list(test_data.columns)
    y_column = 'NextPeriodChangeInLogPrice'

    if additional_columns_to_remove is None:
        additional_columns_to_remove = []

    for column in always_redundant_columns + extra_columns_to_remove + additional_columns_to_remove:
        x_columns.remove(column)

    data = test_data.dropna()

    x_test = data[x_columns]
    y_test = data[y_column]

    # Predict based on test data
    predict_price_lm = model_object.predict(x_test)
    predict_price_lm = predict_price_lm.reshape(-1, 1)
    predict_price_lm = pd.DataFrame(predict_price_lm, columns=['Predicted'])
    predict_price_lm.index = y_test.index

    # Conduct further analysis on results data
    results = pd.DataFrame()
    results['Predicted'] = predict_price_lm
    results['Actual'] = y_test
    results['Residual'] = results['Actual'] - results['Predicted']
    results['Correct_Direction'] = results.apply(lambda x: 1 if x['Actual'] * x['Predicted'] >= 0 else 0, axis=1)

    results['PriceAboveUpperBB2SD'] = x_test['PriceAboveUpperBB2SD']
    results['PriceAboveUpperBB1SD'] = x_test['PriceAboveUpperBB1SD']
    results['PriceBelowLowerBB2SD'] = x_test['PriceBelowLowerBB2SD']
    results['PriceBelowLowerBB1SD'] = x_test['PriceBelowLowerBB1SD']

    results['Above_2SD_Correct_Direction'] = results.apply(
        lambda x: SD_correct_direction(x['Actual'], x['Predicted'], x['PriceAboveUpperBB2SD']), axis=1)
    results['Above_1SD_Correct_Direction'] = results.apply(
        lambda x: SD_correct_direction(x['Actual'], x['Predicted'], x['PriceAboveUpperBB1SD']), axis=1)
    results['Below_2SD_Correct_Direction'] = results.apply(
        lambda x: SD_correct_direction(x['Actual'], x['Predicted'], x['PriceBelowLowerBB2SD']), axis=1)
    results['Below_1SD_Correct_Direction'] = results.apply(
        lambda x: SD_correct_direction(x['Actual'], x['Predicted'], x['PriceBelowLowerBB1SD']), axis=1)

    above_two_sd_series = results['Above_2SD_Correct_Direction'].dropna()
    above_one_sd_series = results['Above_1SD_Correct_Direction'].dropna()
    below_two_sd_series = results['Below_2SD_Correct_Direction'].dropna()
    below_one_sd_series = results['Below_1SD_Correct_Direction'].dropna()

    print("\nModel: ", model_object)
    print(f"Overall Correct_Direction: {results['Correct_Direction'].sum() / len(results)}")
    print(f"Above_2SD_Correct_Direction: {above_two_sd_series.sum() / len(above_two_sd_series)}")
    print(f"Above_1SD_Correct_Direction: {above_one_sd_series.sum() / len(above_one_sd_series)}")
    print(f"Below_2SD_Correct_Direction: {below_two_sd_series.sum() / len(below_two_sd_series)}")
    print(f"Below_1SD_Correct_Direction: {below_one_sd_series.sum() / len(below_one_sd_series)}\n")

    results.drop(['PriceAboveUpperBB2SD', 'PriceAboveUpperBB1SD', 'PriceBelowLowerBB2SD', 'PriceBelowLowerBB1SD'],
                 axis=1, inplace=True)

    results = pd.concat([results, data], axis=1)
    return results
