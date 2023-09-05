import pickle

import pandas as pd

from models.classification_price_change.classification_price_change import create_log_price_variables, \
    create_price_variables, create_volume_change_variables, generate_bollinger_bands, boolean_bollinger_band_location
from utilities.general_strategy_utilities import profit_taker, minutes_since_last_order


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
