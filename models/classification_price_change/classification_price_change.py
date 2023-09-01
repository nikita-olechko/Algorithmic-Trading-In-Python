import os
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
from models.classification_price_change.classification_utilities import create_classification_report_name
from utilities.__init__ import DATE_FORMAT
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
        stk_data[f'{period}period_percentage_change_in_log_price'] = stk_data[f'{period}period_change_in_log_price'] \
                                                                        / stk_data["log_price"].shift(period) * 100
    return stk_data


def create_price_variables(stk_data, list_of_periods=range(1, 11)):
    """
    Create price change and related variables for a given DataFrame.

    :param stk_data: DataFrame containing stock data.
    :param list_of_periods: List of periods to calculate shifted prices.
    :return: Modified DataFrame with price change variables.
    """
    for period in list_of_periods:
        stk_data[f'{period}period_shifted_price'] = stk_data["Average"].shift(period)
        stk_data[f'{period}period_change_in_price'] = stk_data["Average"] - stk_data[f'{period}period_shifted_price']
        stk_data[f'{period}period_percentage_change_in_price'] = stk_data[f'{period}period_change_in_price'] \
                                                                    / stk_data["Average"].shift(period) * 100
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


def price_change_over_next_Z_periods_greater_than_X_boolean(dataFrame, periods, percentage_change):
    dataFrame[f'maximum_percentage_price_change_over_next_{periods}_periods_greater_than_{percentage_change}'] = \
        (dataFrame['Average'].rolling(window=periods).max() - dataFrame['Average']) / dataFrame['Average'] * 100 > \
        percentage_change
    return dataFrame


def prepare_training_data_classification_model(data, barsize, duration, endDateTime, Z_periods=60, X_percentage=3):
    """
    Prepare training data for machine learning models.

    :param data: DataFrame containing stock data.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param endDateTime: End date and time for the data.
    :return: Processed data, feature columns, and target column.
    """
    ib = initialize_ib_connection()
    stk_data = data
    if data is None:
        stk_data = get_stock_data(ib, "XOM", barsize, duration, directory_offset=2, endDateTime=endDateTime)
    stk_data = create_log_price_variables(stk_data)
    stk_data = create_price_variables(stk_data)
    stk_data = create_volume_change_variables(stk_data)
    stk_data = generate_bollinger_bands(stk_data)
    stk_data = boolean_bollinger_band_location(stk_data)
    # Y Variable
    stk_data = price_change_over_next_Z_periods_greater_than_X_boolean(stk_data, Z_periods, X_percentage)

    x_columns = list(stk_data.columns)
    y_column = f'maximum_percentage_price_change_over_next_{Z_periods}_periods_greater_than_{X_percentage}'

    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount', 'Orders',
                                'Position']
    extra_columns_to_remove = [y_column]

    for column in always_redundant_columns + extra_columns_to_remove:
        x_columns.remove(column)

    data = stk_data.dropna()
    return data, x_columns, y_column


def create_classification_price_change_linear_regression_model(symbol, endDateTime='', save_model=True, barsize="1 min",
                                                         duration="2 M", data=None):
    """
    Create a linear regression model for predicting classification price changes.

    :param symbol: Ticker symbol of the stock.
    :param endDateTime: End date and time for the data formatted as YYYYMMDD HH:MM:SS.
    :param save_model: Whether to save the trained model.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param data: DataFrame containing stock data.
    :return: Trained linear regression model.
    """
    data, x_columns, y_column = prepare_training_data_classification_model(data, barsize, duration, endDateTime)
    train = data

    X_train = train[x_columns]
    y_train = train[y_column]

    # Create and train the model
    lm = linear_model.LinearRegression()
    lm.fit(X_train, y_train)

    if save_model:
        model_filename = f'model_objects/classification_price_change_linear_model_{symbol}_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(lm, file)
    return lm


def create_classification_price_change_random_forest_model(symbol, endDateTime='', save_model=True, barsize="1 min",
                                                     duration="2 M", data=None):
    """
    Create a random forest model for predicting classification price changes.

    :param symbol: Ticker symbol of the stock.
    :param endDateTime: End date and time for the data.
    :param save_model: Whether to save the trained model.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param data: DataFrame containing stock data.
    :return: Trained random forest model.
    """
    data, x_columns, y_column = prepare_training_data_classification_model(data, barsize, duration, endDateTime)
    train = data

    x_train = train[x_columns]
    y_train = train[y_column]

    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)

    if save_model:
        model_filename = f'model_objects/classification_price_change_random_forest_model_{symbol}_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(forest, file)
    return forest


def create_classification_price_change_mlp_model(symbol, endDateTime='', save_model=True, barsize="1 min",
                                           duration="2 M", data=None):
    """
    Create a multi-layer perceptron (MLP) model for predicting classification price changes.

    :param symbol: Ticker symbol of the stock.
    :param endDateTime: End date and time for the data.
    :param save_model: Whether to save the trained model.
    :param barsize: Bar size for historical data.
    :param duration: Duration of historical data.
    :param data: DataFrame containing stock data.
    :return: Trained MLP model.
    """
    data, x_columns, y_column = prepare_training_data_classification_model(data, barsize, duration, endDateTime)
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
        model_filename = f'model_objects/classification_price_change_mlp_model_{symbol}_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(best_nn_regressor, file)

    return best_nn_regressor


def analyze_classification_model_performance(ticker, model_object, test_data, additional_columns_to_remove=None, Z_periods=60,
                                              X_percentage=3):
    """
    Analyze the performance of a predictive model.

    :param model_object: Trained predictive model.
    :param test_data: DataFrame containing test data.
    :param additional_columns_to_remove: Additional columns to remove from analysis.
    :return: DataFrame with analysis results.
    """

    x_columns = list(test_data.columns)
    y_column = f'maximum_percentage_price_change_over_next_{Z_periods}_periods_greater_than_{X_percentage}'

    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount', 'Orders',
                                'Position']
    extra_columns_to_remove = [y_column]

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
    results = pd.DataFrame()
    results['Predicted'] = predict_price_lm
    results['Actual'] = y_test

    # Unique Analysis to individual models from here

    results['Residual'] = results['Actual'] - results['Predicted']
    results['Correctly_Predicted_Change'] = results.apply(lambda x: 1 if x['Actual'] * x['Predicted'] > 0 else 0, axis=1)

    results['PriceAboveUpperBB2SD'] = x_test['PriceAboveUpperBB2SD']
    results['PriceAboveUpperBB1SD'] = x_test['PriceAboveUpperBB1SD']
    results['PriceBelowLowerBB2SD'] = x_test['PriceBelowLowerBB2SD']
    results['PriceBelowLowerBB1SD'] = x_test['PriceBelowLowerBB1SD']

    results['Above_2SD_Correctly_Predicted'] = np.where(results['PriceAboveUpperBB2SD'] == 1,
                                                      results['Correctly_Predicted_Change'], np.nan)
    results['Above_1SD_Correctly_Predicted'] = np.where(results['PriceAboveUpperBB1SD'] == 1,
                                                        results['Correctly_Predicted_Change'], np.nan)
    results['Below_2SD_Correctly_Predicted'] = np.where(results['PriceBelowLowerBB2SD'] == 1,
                                                        results['Correctly_Predicted_Change'], np.nan)
    results['Below_1SD_Correctly_Predicted'] = np.where(results['PriceBelowLowerBB1SD'] == 1,
                                                        results['Correctly_Predicted_Change'], np.nan)

    above_two_sd_series = results['Above_2SD_Correct_Direction'].dropna()
    above_one_sd_series = results['Above_1SD_Correct_Direction'].dropna()
    below_two_sd_series = results['Below_2SD_Correct_Direction'].dropna()
    below_one_sd_series = results['Below_1SD_Correct_Direction'].dropna()

    prediction_dict = {f"{ticker}": ticker,
                       "Overall_Correct_Direction": results['Correctly_Predicted_Change'].sum() / len(results['Correctly_Predicted_Change'].dropna()),
                       "Above_2SD_Correct_Direction": above_two_sd_series.sum() / len(above_two_sd_series),
                       "Above_1SD_Correct_Direction": above_one_sd_series.sum() / len(above_one_sd_series),
                       "Below_2SD_Correct_Direction": below_two_sd_series.sum() / len(below_two_sd_series),
                       "Below_1SD_Correct_Direction": below_one_sd_series.sum() / len(below_one_sd_series)}

    print("\nModel: ", model_object)
    print(prediction_dict)

    results.drop(['PriceAboveUpperBB2SD', 'PriceAboveUpperBB1SD', 'PriceBelowLowerBB2SD', 'PriceBelowLowerBB1SD'],
                 axis=1, inplace=True)

    results = pd.concat([results, data], axis=1)

    model_results_file = create_classification_report_name(Z_periods, X_percentage)

    if os.path.isfile(os.path.join(model_results_file)):
        try:
            model_results = pd.read_csv(os.path.join(model_results_file), parse_dates=True, index_col=0,
                                   date_format=DATE_FORMAT)
        except FileNotFoundError:
            model_results = pd.DataFrame()

    model_results.loc[len(model_results)] = prediction_dict
    model_results.to_csv(os.path.join(model_results_file))

    return results
