import numpy as np
import pandas as pd
from ib_insync import IB
from sklearn import (
    linear_model
)
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from backtesting.backtestingUtilities.simulationUtilities import get_stock_data
from utilities.generalUtilities import get_tws_connection_id


def create_log_price_variables(stk_data, list_of_periods=range(1, 11)):
    log_price = np.log(stk_data["Average"])
    stk_data["log_price"] = log_price
    for period in list_of_periods:
        stk_data[f'{period}period_shifted_log_price'] = stk_data["log_price"].shift(period)
        stk_data[f'{period}period_change_in_log_price'] = stk_data["log_price"] - stk_data[
            f'{period}period_shifted_log_price']
    return stk_data


def create_volume_change_variables(stk_data, list_of_periods=range(1, 11)):
    log_volume = np.log(stk_data["Volume"])
    stk_data["log_volume"] = log_volume
    for period in list_of_periods:
        stk_data[f'{period}period_shifted_log_volume'] = stk_data["log_volume"].shift(period)
        stk_data[f'{period}period_change_in_log_volume'] = stk_data["log_volume"] - stk_data[
            f'{period}period_shifted_log_volume']
    return stk_data


def generate_bollinger_bands(dataFrame, period=20):
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
    minuteDataFrame['PriceAboveUpperBB2SD'] = np.where(minuteDataFrame['Average'] > minuteDataFrame['UpperBB2SD'], 1, 0)
    minuteDataFrame['PriceAboveUpperBB1SD'] = np.where(
        (minuteDataFrame['Average'] > minuteDataFrame['UpperBB1SD']) & (minuteDataFrame['PriceAboveUpperBB2SD'] == 0),
        1, 0)
    minuteDataFrame['PriceBelowLowerBB2SD'] = np.where(minuteDataFrame['Average'] < minuteDataFrame['LowerBB2SD'], 1, 0)
    minuteDataFrame['PriceBelowLowerBB1SD'] = np.where(
        (minuteDataFrame['Average'] < minuteDataFrame['LowerBB1SD']) & (minuteDataFrame['PriceBelowLowerBB2SD'] == 0),
        1, 0)
    return minuteDataFrame


def above_X_correct_direction(actual, predicted, x=0):
    if actual * predicted >= 0 and abs(predicted) > x:
        return 1
    elif actual * predicted < 0:
        return 0
    else:
        return np.nan


def create_relative_price_change_linear_regression_model(symbol):
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4000, clientId=get_tws_connection_id())
    except Exception:
        print("Could not connect to IBKR. Check that Trader Workstation or IB Gateway is running.")
    stk_data = get_stock_data(ib, "XOM", "1 Min", "2 M", directory_offset=2)
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

    train = data

    X_train = train[x_columns]
    y_train = train[y_column]

    # Create and train the model
    lm = linear_model.LinearRegression()
    lm.fit(X_train, y_train)

    model_filename = f'model_objects/relative_price_change_linear_model_{symbol}.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(lm, file)


def create_relative_price_change_random_forest_model(symbol):
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4000, clientId=get_tws_connection_id())
    except Exception:
        print("Could not connect to IBKR. Check that Trader Workstation or IB Gateway is running.")
    stk_data = get_stock_data(ib, "XOM", "1 Min", "2 M", directory_offset=2)
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

    train = data

    x_train = train[x_columns]
    y_train = train[y_column]

    forest = RandomForestRegressor()
    forest.fit(x_train, y_train)

    model_filename = f'model_objects/relative_price_change_random_forest_{symbol}.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(forest, file)


def analyze_model_performance(model_object, test_data):
    lm = model_object

    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount', 'Orders',
                                'Position']
    extra_columns_to_remove = ['NextPeriodChangeInLogPrice']

    x_columns = list(test_data.columns)
    y_column = 'NextPeriodChangeInLogPrice'

    for column in always_redundant_columns + extra_columns_to_remove:
        x_columns.remove(column)

    data = test_data.dropna()

    x_test = data[x_columns]
    y_test = data[y_column]

    # Predict based on test data
    predict_price_lm = lm.predict(x_test)
    predict_price_lm = predict_price_lm.reshape(-1, 1)
    predict_price_lm = pd.DataFrame(predict_price_lm, columns=['Predicted'])
    predict_price_lm.index = y_test.index
    rmse_lm = np.sqrt(mean_squared_error(predict_price_lm, y_test))

    # Conduct further analysis on results data
    results = pd.DataFrame()
    results['Predicted'] = predict_price_lm
    results['Actual'] = y_test
    results['Residual'] = results['Actual'] - results['Predicted']
    results['Correct_Direction'] = results.apply(lambda x: 1 if x['Actual'] * x['Predicted'] >= 0 else 0, axis=1)
    twoSD = np.std(results['Predicted']) * 2
    oneSD = np.std(results['Predicted'])
    results['Above_2SD_Correct_Direction'] = results.apply(
        lambda x: above_X_correct_direction(x['Actual'], x['Predicted'], x=twoSD), axis=1)
    results['Above_1SD_Correct_Direction'] = results.apply(
        lambda x: above_X_correct_direction(x['Actual'], x['Predicted'], x=oneSD), axis=1)

    results['Between_1and2SD_Correct_Direction'] = results.apply(
        lambda x: x['Above_1SD_Correct_Direction'] if np.isnan(x['Above_2SD_Correct_Direction']) else np.nan, axis=1)

    above_two_sd_series = results['Above_2SD_Correct_Direction'].dropna()
    above_one_sd_series = results['Above_1SD_Correct_Direction'].dropna()
    between_one_and_two_sd_series = results['Between_1and2SD_Correct_Direction'].dropna()

    print(f"Overall Correct_Direction: {results['Correct_Direction'].sum() / len(results)}")
    print(f"Above_2SD_Correct_Direction: {above_two_sd_series.sum() / len(above_two_sd_series)}")
    print(f"Above_1SD_Correct_Direction: {above_one_sd_series.sum() / len(above_one_sd_series)}")
    print(
        f"Between_1and2SD_Correct_Direction: {between_one_and_two_sd_series.sum() / len(between_one_and_two_sd_series)}")

    results = pd.concat([results, data], axis=1)
    return results
