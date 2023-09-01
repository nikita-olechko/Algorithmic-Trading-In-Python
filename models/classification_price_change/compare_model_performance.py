import numpy as np
import pandas as pd
import pickle

from backtesting.backtestingUtilities.simulationUtilities import get_stock_data
from utilities.generalUtilities import initialize_ib_connection, get_months_of_historical_data, ibkr_query_time_months
from classification_price_change import create_price_variables, create_log_price_variables, \
    create_volume_change_variables, generate_bollinger_bands, boolean_bollinger_band_location, \
    price_change_over_next_Z_periods_greater_than_X_boolean, create_classification_price_change_linear_regression_model, \
    create_classification_price_change_random_forest_model, create_classification_price_change_mlp_model, \
    analyze_classification_model_performance

Z_periods = 60
X_percentage = 3

for ticker in ["XOM"]:
    ib = initialize_ib_connection()
    barsize = "1 min"
    test_duration = "2 M"
    months = 12

    model_data = get_months_of_historical_data(ib, ticker, months=months, barsize=barsize, directory_offset=2,
                                               months_offset=2)

    model_data = create_log_price_variables(model_data, list_of_periods=range(1, Z_periods + 1))
    model_data = create_price_variables(model_data, list_of_periods=range(1, Z_periods + 1))
    model_data = create_volume_change_variables(model_data, list_of_periods=range(1, Z_periods + 1))
    model_data = generate_bollinger_bands(model_data)
    model_data = boolean_bollinger_band_location(model_data)
    # Y Variable
    model_data = price_change_over_next_Z_periods_greater_than_X_boolean(model_data, Z_periods, X_percentage)

    model_data = model_data.dropna()

    test_data = get_stock_data(ib, ticker, barsize=barsize, duration=test_duration, directory_offset=2)
    test_data = create_log_price_variables(test_data)
    test_data['NextPeriodChangeInLogPrice'] = test_data['log_price'].shift(-1) - test_data['log_price']
    test_data = create_volume_change_variables(test_data)
    test_data = generate_bollinger_bands(test_data)
    test_data = boolean_bollinger_band_location(test_data)
    test_data = test_data.dropna()

    lm = create_classification_price_change_linear_regression_model(ticker, save_model=True, data=model_data,
                                                                    barsize=barsize, duration=f"{months} M")
    rf = create_classification_price_change_random_forest_model(ticker, save_model=True, data=model_data,
                                                                barsize=barsize, duration=f"{months} M")
    mlp = create_classification_price_change_mlp_model(ticker, save_model=True, data=model_data, barsize=barsize,
                                                       duration=f"{months} M")

    # with open(f'model_objects/classification_price_change_linear_model_{ticker}_5mins_12M.pkl', 'rb') as file:
    #     lm = pickle.load(file)
    # with open(f'model_objects/classification_price_change_random_forest_model_{ticker}_5mins_12M.pkl', 'rb') as file:
    #     rf = pickle.load(file)
    # with open(f'model_objects/classification_price_change_mlp_model_{ticker}_5mins_12M.pkl', 'rb') as file:
    #     mlp = pickle.load(file)

    lm_results = analyze_classification_model_performance(lm, test_data, model_type='lm', Z_periods=Z_periods,
                                                          X_percentage=X_percentage)
    rf_results = analyze_classification_model_performance(rf, test_data, model_type='rf', Z_periods=Z_periods,
                                                          X_percentage=X_percentage)
    mlp_results = analyze_classification_model_performance(mlp, test_data, model_type='mlp', Z_periods=Z_periods,
                                                           X_percentage=X_percentage)

    # lm_results.to_csv(f'model_performance/lm_results_{ticker}.csv')
    # rf_results.to_csv(f'model_performance/rf_results_{ticker}.csv')
    # mlp_results.to_csv(f'model_performance/mlp_results_{ticker}.csv')

    # lm_performance = {'Overall_Correct_Direction': lm_results['Correct_Direction'].mean(),
    #                   'Above_1SD_Correct_Direction': lm_results['Above_1SD_Correct_Direction'].mean(),
    #                   'Above_2SD_Correct_Direction': lm_results['Above_2SD_Correct_Direction'].mean(),
    #                   'Below_1SD_Correct_Direction': lm_results['Below_1SD_Correct_Direction'].mean(),
    #                   'Below_2SD_Correct_Direction': lm_results['Below_2SD_Correct_Direction'].mean()}
    #
    # rf_performance = {'Overall_Correct_Direction': rf_results['Correct_Direction'].mean(),
    #                   'Above_1SD_Correct_Direction': rf_results['Above_1SD_Correct_Direction'].mean(),
    #                   'Above_2SD_Correct_Direction': rf_results['Above_2SD_Correct_Direction'].mean(),
    #                   'Below_1SD_Correct_Direction': rf_results['Below_1SD_Correct_Direction'].mean(),
    #                   'Below_2SD_Correct_Direction': rf_results['Below_2SD_Correct_Direction'].mean()}
    #
    # mlp_performance = {'Overall_Correct_Direction': mlp_results['Correct_Direction'].mean(),
    #                    'Above_1SD_Correct_Direction': mlp_results['Above_1SD_Correct_Direction'].mean(),
    #                    'Above_2SD_Correct_Direction': mlp_results['Above_2SD_Correct_Direction'].mean(),
    #                    'Below_1SD_Correct_Direction': mlp_results['Below_1SD_Correct_Direction'].mean(),
    #                    'Below_2SD_Correct_Direction': mlp_results['Below_2SD_Correct_Direction'].mean()}
    #
    # print("Ticker: ", ticker)
    # print("lm")
    # print(lm_performance)
    # print("rf")
    # print(rf_performance)
    # print("mlp")
    # print(mlp_performance)
    # print()
