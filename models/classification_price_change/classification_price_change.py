import os
import pickle

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

from backtesting.backtestingUtilities.simulationUtilities import retrieve_base_data
from utilities.classification_utilities import create_classification_report_name, \
    occurences_more_than_Z_periods_apart, incorrect_detections_not_within_Z_periods_of_correct_detection, \
    price_changes_after_incorrect_detections, detection_indices_by_correctness
from utilities.__init__ import DATE_FORMAT
from utilities.dataGenerationUtilities import create_volume_variables, create_price_variables, \
    create_log_price_variables, generate_bollinger_bands
from utilities.generalUtilities import initialize_ib_connection, timer


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
    dataFrame['Max_Price_in_Next_Z_Periods'] = dataFrame['Average'].rolling(periods).max().shift(-(periods - 1))
    dataFrame[f'maximum_percentage_price_change_over_next_{periods}_periods_greater_than_{percentage_change}'] = \
        ((dataFrame['Max_Price_in_Next_Z_Periods'] - dataFrame['Average']) / dataFrame[
            'Average']) * 100 >= percentage_change
    dataFrame.drop(['Max_Price_in_Next_Z_Periods'], axis=1, inplace=True)
    return dataFrame


def prepare_data_classification_model(ticker, barsize, duration, endDateTime='', data=None,
                                      Z_periods=60, X_percentage=3, months_offset=0, very_large_data=False,
                                      try_errored_tickers=True, periodicity=1):
    """
    Prepare training data for machine learning models.
    """
    ib = initialize_ib_connection()
    stk_data = data
    if data is None:
        stk_data = retrieve_base_data(ib, ticker, barsize, duration, directory_offset=2,
                                      endDateTime=endDateTime, months_offset=months_offset,
                                      very_large_data=very_large_data, try_errored_tickers=try_errored_tickers)
    stk_data = create_log_price_variables(stk_data, list_of_periods=range(1, Z_periods, periodicity))
    stk_data = create_price_variables(stk_data, list_of_periods=range(1, Z_periods, periodicity))
    stk_data = create_volume_variables(stk_data, list_of_periods=range(1, Z_periods, periodicity))
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


@timer
def create_classification_price_change_logistic_regression_model(symbol, endDateTime='', save_model=True,
                                                                 barsize="1 min",
                                                                 duration="2 M", data=None, Z_periods=60,
                                                                 X_percentage=3,
                                                                 prepped_data_column_tuple=None):
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
    if prepped_data_column_tuple is None:
        data, x_columns, y_column = prepare_data_classification_model(data, barsize, duration, endDateTime)
    else:
        data = prepped_data_column_tuple[0]
        x_columns = prepped_data_column_tuple[1]
        y_column = prepped_data_column_tuple[2]
    train = data

    X_train = train[x_columns]
    y_train = train[y_column]

    # Create and train the logistic regression model
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)

    if save_model:
        model_filename = f'model_objects/classification_price_change_lm_{symbol}_{Z_periods}_periods_{X_percentage}_percent_change_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(logistic_reg, file)
    return logistic_reg


@timer
def create_classification_price_change_random_forest_model(symbol, data, endDateTime='', save_model=True,
                                                           barsize="1 min",
                                                           duration="2 M", Z_periods=60, X_percentage=3,
                                                           prepped_data_column_tuple=None):
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
    if prepped_data_column_tuple is None:
        data, x_columns, y_column = prepare_data_classification_model(data, barsize, duration, endDateTime)
    else:
        data = prepped_data_column_tuple[0]
        x_columns = prepped_data_column_tuple[1]
        y_column = prepped_data_column_tuple[2]
    train = data

    X_train = train[x_columns]
    y_train = train[y_column]

    # Create and train the Random Forest classification model
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)

    if save_model:
        model_filename = f'model_objects/classification_price_change_rf_{symbol}_{Z_periods}_periods_{X_percentage}_percent_change_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(random_forest, file)
    return random_forest


@timer
def create_classification_price_change_mlp_model(symbol, endDateTime='', save_model=True, barsize="1 min",
                                                 duration="2 M", data=None, Z_periods=60, X_percentage=3,
                                                 prepped_data_column_tuple=None):
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
    if prepped_data_column_tuple is None:
        data, x_columns, y_column = prepare_data_classification_model(data, barsize, duration, endDateTime)
    else:
        data = prepped_data_column_tuple[0]
        x_columns = prepped_data_column_tuple[1]
        y_column = prepped_data_column_tuple[2]

    x_train, x_test, y_train, y_test = train_test_split(data[x_columns], data[y_column], test_size=0.2, random_state=42)

    mlp_classifier = MLPClassifier(max_iter=1000, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
    }

    grid_search = GridSearchCV(mlp_classifier, param_grid, cv=3, scoring='accuracy')

    grid_search.fit(x_train, y_train)

    best_mlp_classifier = grid_search.best_estimator_

    if save_model:
        model_filename = f'model_objects/classification_price_change_mlp_{symbol}_{Z_periods}_periods_{X_percentage}_percent_change_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(best_mlp_classifier, file)

    return best_mlp_classifier


def analyze_classification_model_performance(ticker, model_object, test_data, additional_columns_to_remove=None,
                                             Z_periods=60,
                                             X_percentage=3, model_type='lm', allowable_error=0, model_data_range=None):
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

    results['Residual'] = results['Actual'].astype(int) - results['Predicted'].astype(int)
    results['Incorrect_Detections'] = (results["Residual"] == -1).astype(int)

    data['Max_Price_in_Next_Z_Periods'] = data['Average'].rolling(Z_periods).max().shift(-(Z_periods - 1))
    data[f'maximum_percentage_price_change_over_next_{Z_periods}'] = \
        ((data['Max_Price_in_Next_Z_Periods'] - data['Average']) / data[
            'Average']) * 100

    detections_within_error, detections_outside_error \
        = detection_indices_by_correctness(results, data, Z_periods, X_percentage, allowable_error)

    price_change_percentage_after_Z_periods_incorrect_detection \
        = price_changes_after_incorrect_detections(results, Z_periods, X_percentage, data, allowable_error,
                                                   detections_within_error, detections_outside_error)

    def correctly_predicted_change(actual, predicted):
        if actual == 1:
            return predicted
        return np.nan

    results['Correctly_Predicted_Change'] = results.apply(
        lambda x: correctly_predicted_change(x['Actual'], x['Predicted']),
        axis=1)

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

    above_two_sd_series = results['Above_2SD_Correctly_Predicted'].dropna()
    above_one_sd_series = results['Above_1SD_Correctly_Predicted'].dropna()
    below_two_sd_series = results['Below_2SD_Correctly_Predicted'].dropna()
    below_one_sd_series = results['Below_1SD_Correctly_Predicted'].dropna()

    prediction_dict = {"ticker": ticker,
                       "Overall_Correct_Prediction_When_Detected": results['Correctly_Predicted_Change'].sum() /
                                                                   results['Predicted'].sum(),
                       "Total_Occurences": results['Actual'].sum(),
                       "Correct_Detections": results['Correctly_Predicted_Change'].sum(),
                       "Incorrect_Detections": (results["Residual"] == -1).sum(),
                       f"Grouped_Occurences": len(
                           occurences_more_than_Z_periods_apart(
                               results, column_name='Actual', Z_periods=Z_periods)),
                       f"Grouped_Detections": len(
                           occurences_more_than_Z_periods_apart(
                               results, column_name='Predicted', Z_periods=Z_periods)),
                       f"Grouped_Correct_Detections":
                           len(occurences_more_than_Z_periods_apart(results, column_name='Correctly_Predicted_Change'
                                                                    , Z_periods=Z_periods)),
                       f"Grouped_Correct_Detections_Within_Error": len(detections_within_error) +
                                                                   len(occurences_more_than_Z_periods_apart(
                                                                       results,
                                                                       column_name='Correctly_Predicted_Change'
                                                                       , Z_periods=Z_periods)),
                       f"Grouped_Strictly_Incorrect_Detections":
                           len(incorrect_detections_not_within_Z_periods_of_correct_detection(results, Z_periods)),
                       f"Grouped_Incorrect_Detections_Outside_Error":
                           len(detections_outside_error),
                       "Grouped_Average_Incorrect_Detection_Price_Change": np.mean(
                           price_change_percentage_after_Z_periods_incorrect_detection),
                       "Grouped_Standard_Deviation_Incorrect_Detection_Price_Change": np.std(
                           price_change_percentage_after_Z_periods_incorrect_detection),
                       "Model_Data_Range": f"{model_data_range}",
                       "Test_Data_Range": f"{list(data['Date'])[0]} to {list(data['Date'])[-1]}",
                       "Above_2SD_Correctly_Predicted": above_two_sd_series.sum(),
                       "Above_1SD_Correctly_Predicted": above_one_sd_series.sum(),
                       "Below_2SD_Correctly_Predicted": below_two_sd_series.sum(),
                       "Below_1SD_Correctly_Predicted": below_one_sd_series.sum()}

    print("\nModel: ", model_object)
    print(prediction_dict)

    model_results_file = create_classification_report_name(Z_periods, X_percentage, model_type, allowable_error)

    if os.path.isfile(os.path.join(model_results_file)):
        try:
            model_results = pd.read_csv(os.path.join(model_results_file), parse_dates=True, index_col=0,
                                        date_format=DATE_FORMAT)
        except FileNotFoundError:
            model_results = pd.DataFrame(columns=list(prediction_dict.keys()))
    else:
        model_results = pd.DataFrame(columns=list(prediction_dict.keys()))

    model_results.loc[len(model_results)] = prediction_dict
    model_results.to_csv(os.path.join(model_results_file))
