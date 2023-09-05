from pandas.errors import PerformanceWarning
import warnings

from utilities.classification_utilities import get_model
from classification_price_change import create_classification_price_change_logistic_regression_model, \
    create_classification_price_change_random_forest_model, create_classification_price_change_mlp_model, \
    analyze_classification_model_performance, prepare_data_classification_model

warning_categories_to_ignore = [PerformanceWarning, RuntimeWarning]


def run_classification_model_accuracy_tests(list_of_Z_periods, list_of_X_percentages, list_of_tickers,
                                            models_to_run=('lm', 'rf', 'mlp'), barsize="1 min", model_duration="12 M",
                                            test_duration="2 M", allowable_error=0, periodicity=1):
    for Z_periods in list_of_Z_periods:
        for X_percentage in list_of_X_percentages:
            for ticker in list_of_tickers:
                try:
                    model_data, x_columns, y_column = prepare_data_classification_model(barsize=barsize,
                                                                                        duration=model_duration,
                                                                                        ticker=ticker,
                                                                                        Z_periods=Z_periods,
                                                                                        X_percentage=X_percentage,
                                                                                        months_offset=int(
                                                                                            test_duration.split(" ")[
                                                                                                0]) + 1,
                                                                                        very_large_data=True,
                                                                                        try_errored_tickers=True,
                                                                                        periodicity=periodicity)

                    test_data = prepare_data_classification_model(barsize=barsize, duration=test_duration,
                                                                  ticker=ticker,
                                                                  Z_periods=Z_periods, X_percentage=X_percentage,
                                                                  months_offset=0, very_large_data=True,
                                                                  try_errored_tickers=True,
                                                                  periodicity=periodicity)[0]

                    model_data_tuple = (model_data, x_columns, y_column)

                    model_creation_dict = {'lm': create_classification_price_change_logistic_regression_model,
                                           'rf': create_classification_price_change_random_forest_model,
                                           'mlp': create_classification_price_change_mlp_model}

                    for model in models_to_run:
                        model_object = get_model(model_creation_dict, model, symbol=ticker,
                                                 Z_periods=Z_periods, X_percentage=X_percentage,
                                                 prepped_data_column_tuple=model_data_tuple,
                                                 barsize=barsize, duration=model_duration)

                        analyze_classification_model_performance(ticker=ticker, model_object=model_object,
                                                                 test_data=test_data,
                                                                 model_type=model, Z_periods=Z_periods,
                                                                 X_percentage=X_percentage,
                                                                 allowable_error=allowable_error)
                    print(f"Finished Ticker: {ticker}, Periods: {Z_periods}, Percentage: {X_percentage}")
                except Exception as e:
                    print(f"Error with {ticker}: {e}")
                    continue


list_of_Z_periods = [120, 180]
list_of_X_percentages = [1.5, 1]
list_of_tickers = ['XOM', 'TSLA', 'MSFT', 'AMZN']
extra_tickers = ["TSLA", "NIO", "PLTR", "ROKU", "ZM", "MRNA", "SPCE", "NKLA", "ZI", "SNOW", "PTON", "GME", "AMC", "PLUG"]
complete_list_of_tickers = list_of_tickers + extra_tickers
models_to_run = ['rf']
allowable_error_percentage = 75
periodicity = 1
model_duration = "12 M"
test_duration = "3 M"
barsize = "1 min"

# Filter out the specified warning categories
for warning_category in warning_categories_to_ignore:
    warnings.filterwarnings("ignore", category=warning_category)
run_classification_model_accuracy_tests(list_of_Z_periods, list_of_X_percentages, complete_list_of_tickers,
                                        models_to_run, barsize=barsize,
                                        model_duration=model_duration, test_duration=test_duration,
                                        periodicity=periodicity,
                                        allowable_error=allowable_error_percentage)
