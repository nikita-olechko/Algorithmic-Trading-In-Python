from models.classification_price_change.classification_utilities import get_model
from classification_price_change import create_classification_price_change_logistic_regression_model, \
    create_classification_price_change_random_forest_model, create_classification_price_change_mlp_model, \
    analyze_classification_model_performance, prepare_data_classification_model


def run_classification_model_accuracy_tests(list_of_Z_periods, list_of_X_percentages, list_of_tickers,
                                            models_to_run=['lm', 'rf', 'mlp']):
    for Z_periods in list_of_Z_periods:
        for X_percentage in list_of_X_percentages:
            for ticker in list_of_tickers:
                try:
                    barsize = "1 min"
                    model_duration = "12 M"
                    test_duration = "2 M"

                    model_data, x_columns, y_column = prepare_data_classification_model(barsize=barsize,
                                                                                        duration=model_duration,
                                                                                        ticker=ticker,
                                                                   Z_periods=Z_periods, X_percentage=X_percentage,
                                                                   months_offset=int(test_duration.split(" ")[0]) + 1,
                                                                   very_large_data=True, try_errored_tickers=True)

                    test_data = prepare_data_classification_model(barsize=barsize, duration=test_duration,
                                                                  ticker=ticker,
                                                                  Z_periods=Z_periods, X_percentage=X_percentage,
                                                                  months_offset=0, very_large_data=False,
                                                                  try_errored_tickers=True)[0]

                    model_data_tuple = (model_data, x_columns, y_column)

                    model_creation_dict = {'lm': create_classification_price_change_logistic_regression_model,
                                           'rf': create_classification_price_change_random_forest_model,
                                           'mlp': create_classification_price_change_mlp_model}

                    for model in models_to_run:
                        model = get_model(model_creation_dict, model, ticker, Z_periods, X_percentage,
                                       prepped_data_column_tuple=model_data_tuple, barsize=barsize, duration=model_duration)

                        analyze_classification_model_performance(ticker=ticker, model_object=lm,
                                                                              test_data=test_data,
                                                                              model_type=model, Z_periods=Z_periods,
                                                                              X_percentage=X_percentage)
                    print(f"Finished Ticker: {ticker}, Periods: {Z_periods}, Percentage: {X_percentage}")
                except Exception as e:
                    print(f"Error with {ticker}: {e}")
                    continue


list_of_Z_periods = [60]
list_of_X_percentages = [3, 2, 1]
list_of_tickers = ['XOM', 'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOG', 'GOOGL', 'JNJ', 'V', 'PG', 'JPM', 'UNH', 'HD', 'MA']
models_to_run = ['lm', 'rf', 'mlp']

run_classification_model_accuracy_tests(list_of_Z_periods, list_of_X_percentages, list_of_tickers, models_to_run)
