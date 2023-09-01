from models.classification_price_change.classification_utilities import get_model
from classification_price_change import create_classification_price_change_linear_regression_model, \
    create_classification_price_change_random_forest_model, create_classification_price_change_mlp_model, \
    analyze_classification_model_performance, prepare_data_classification_model

Z_periods = 60
X_percentage = 3

for ticker in ["XOM"]:
    barsize = "1 min"
    model_duration = "12 M"
    test_duration = "2 M"

    test_data, x_columns, y_column = prepare_data_classification_model(barsize=barsize, duration=test_duration,
                                                                       ticker=ticker,
                                                                       Z_periods=Z_periods, X_percentage=X_percentage,
                                                                       months_offset=0, very_large_data=False,
                                                                       try_errored_tickers=True)

    test_data_tuple = (test_data, x_columns, y_column)

    model_data = prepare_data_classification_model(barsize=barsize, duration=model_duration, ticker=ticker,
                                                   Z_periods=Z_periods, X_percentage=X_percentage,
                                                   months_offset=int(test_duration.split(" ")[0])+1,
                                                   very_large_data=True, try_errored_tickers=True)[0]

    model_creation_dict = {'lm': create_classification_price_change_linear_regression_model,
                           'rf': create_classification_price_change_random_forest_model,
                           'mlp': create_classification_price_change_mlp_model}

    lm = get_model(model_creation_dict, 'lm', ticker, Z_periods, X_percentage,
                   prepped_data_column_tuple=test_data_tuple, barsize=barsize, duration=model_duration)
    rf = get_model(model_creation_dict, 'rf', ticker, Z_periods, X_percentage,
                   prepped_data_column_tuple=test_data_tuple, barsize=barsize, duration=model_duration)
    mlp = get_model(model_creation_dict, 'mlp', ticker, Z_periods, X_percentage,
                    prepped_data_column_tuple=test_data_tuple, barsize=barsize, duration=model_duration)

    lm_results = analyze_classification_model_performance(ticker=ticker, model_object=lm, test_data=test_data,
                                                          model_type='lm', Z_periods=Z_periods,
                                                          X_percentage=X_percentage)
    rf_results = analyze_classification_model_performance(ticker=ticker, model_object=rf, test_data=test_data,
                                                          model_type='rf', Z_periods=Z_periods,
                                                          X_percentage=X_percentage)
    mlp_results = analyze_classification_model_performance(ticker=ticker, model_object=mlp, test_data=test_data,
                                                           model_type='mlp', Z_periods=Z_periods,
                                                           X_percentage=X_percentage)
