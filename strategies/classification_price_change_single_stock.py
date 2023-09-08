from utilities.dataGenerationUtilities import create_price_variables, create_log_price_variables, \
    generate_bollinger_bands, boolean_bollinger_band_location, days_crossover, \
    create_volume_variables
from utilities.general_strategy_utilities import profit_taker, minutes_since_last_order, stop_loss
from utilities.model_strategy_utilities import predict_based_on_model
from pandas.errors import PerformanceWarning
import warnings

warning_categories_to_ignore = [PerformanceWarning, RuntimeWarning]

# Filter out the specified warning categories
for warning_category in warning_categories_to_ignore:
    warnings.filterwarnings("ignore", category=warning_category)


def generate_model_data(barDataFrame, model_object=None, Z_periods=120, periodicity=1):
    barDataFrame = days_crossover(barDataFrame, Z_periods)
    barDataFrame = create_log_price_variables(barDataFrame, list_of_periods=range(1, Z_periods, periodicity))
    barDataFrame = create_price_variables(barDataFrame, list_of_periods=range(1, Z_periods, periodicity))
    barDataFrame = create_volume_variables(barDataFrame, list_of_periods=range(1, Z_periods, periodicity))
    barDataFrame = generate_bollinger_bands(barDataFrame)
    barDataFrame = boolean_bollinger_band_location(barDataFrame)
    barDataFrame = predict_based_on_model(barDataFrame, model_object)
    return barDataFrame


def classification_model_strategy(barDataFrame, last_order_index=0, current_index=-1, ticker=None):
    """
    A strategy that searches for periods where the price has gone down > 1% in 10 periods
    """
    current_prediction = barDataFrame.loc[barDataFrame.index[current_index], 'Prediction']
    if barDataFrame["Orders"][last_order_index] == 1:
        if stop_loss(barDataFrame, last_order_index, current_index, 2.0) & current_prediction != 1:
            return -1
        if profit_taker(barDataFrame, last_order_index, current_index, 1.5):
            if current_prediction == 1:
                return 3
            return -1
        if minutes_since_last_order(barDataFrame, last_order_index, current_index) > 120:
            if current_prediction == 1:
                return 3
            return -1
        return 2
    if current_prediction == 1:
        return 1
    return 0
