import numpy as np
import pandas as pd
import pickle

from backtesting.backtestingUtilities.simulationUtilities import get_stock_data
from models.relative_price_change.relative_price_change import create_relative_price_change_linear_regression_model, \
    create_relative_price_change_random_forest_model, analyze_model_performance, create_log_price_variables, \
    create_volume_change_variables, generate_bollinger_bands, boolean_bollinger_band_location, \
    create_relative_price_change_mlp_model
from utilities.generalUtilities import initialize_ib_connection, get_months_of_historical_data, ibkr_query_time_months

for ticker in ["XOM"]:
    barsize = "5 mins"
    test_duration = "2 M"
    months = 12

    # model_data = get_months_of_historical_data(ib, ticker, months=months, barsize=barsize, directory_offset=2,
    #                                           months_offset=2)
    #
    # model_data = create_log_price_variables(model_data)
    # model_data['NextPeriodChangeInLogPrice'] = model_data['log_price'].shift(-1) - model_data['log_price']
    # model_data = create_volume_change_variables(model_data)
    # model_data = generate_bollinger_bands(model_data)
    # model_data = boolean_bollinger_band_location(model_data)
    # model_data = model_data.dropna()

    test_data = get_stock_data(ticker, barsize=barsize, duration=test_duration, directory_offset=2)
    test_data = create_log_price_variables(test_data)
    test_data['NextPeriodChangeInLogPrice'] = test_data['log_price'].shift(-1) - test_data['log_price']
    test_data = create_volume_change_variables(test_data)
    test_data = generate_bollinger_bands(test_data)
    test_data = boolean_bollinger_band_location(test_data)
    test_data = test_data.dropna()

    # lm = create_relative_price_change_linear_regression_model(ticker, save_model=True, data=model_data, barsize=barsize
    #                                                           , duration="12 M")
    # rf = create_relative_price_change_random_forest_model(ticker, save_model=True, data=model_data, barsize=barsize
    #                                                           , duration="12 M")
    # mlp = create_relative_price_change_mlp_model(ticker, save_model=True, data=model_data, barsize=barsize
    #                                                           , duration="12 M")
    with open(f'model_objects/relative_price_change_linear_model_{ticker}_5mins_12M.pkl', 'rb') as file:
        lm = pickle.load(file)
    with open(f'model_objects/relative_price_change_random_forest_model_{ticker}_5mins_12M.pkl', 'rb') as file:
        rf = pickle.load(file)
    with open(f'model_objects/relative_price_change_mlp_model_{ticker}_5mins_12M.pkl', 'rb') as file:
        mlp = pickle.load(file)

    lm_results = analyze_model_performance(lm, test_data)
    rf_results = analyze_model_performance(rf, test_data)
    mlp_results = analyze_model_performance(mlp, test_data)
    lm_results.to_csv(f'model_performance/lm_results_{ticker}.csv')
    rf_results.to_csv(f'model_performance/rf_results_{ticker}.csv')
    mlp_results.to_csv(f'model_performance/mlp_results_{ticker}.csv')

    lm_performance = {'Overall_Correct_Direction': lm_results['Correct_Direction'].mean(),
                      'Above_1SD_Correct_Direction': lm_results['Above_1SD_Correct_Direction'].mean(),
                      'Above_2SD_Correct_Direction': lm_results['Above_2SD_Correct_Direction'].mean(),
                      'Below_1SD_Correct_Direction': lm_results['Below_1SD_Correct_Direction'].mean(),
                      'Below_2SD_Correct_Direction': lm_results['Below_2SD_Correct_Direction'].mean()}

    rf_performance = {'Overall_Correct_Direction': rf_results['Correct_Direction'].mean(),
                      'Above_1SD_Correct_Direction': rf_results['Above_1SD_Correct_Direction'].mean(),
                      'Above_2SD_Correct_Direction': rf_results['Above_2SD_Correct_Direction'].mean(),
                      'Below_1SD_Correct_Direction': rf_results['Below_1SD_Correct_Direction'].mean(),
                      'Below_2SD_Correct_Direction': rf_results['Below_2SD_Correct_Direction'].mean()}

    mlp_performance = {'Overall_Correct_Direction': mlp_results['Correct_Direction'].mean(),
                       'Above_1SD_Correct_Direction': mlp_results['Above_1SD_Correct_Direction'].mean(),
                       'Above_2SD_Correct_Direction': mlp_results['Above_2SD_Correct_Direction'].mean(),
                       'Below_1SD_Correct_Direction': mlp_results['Below_1SD_Correct_Direction'].mean(),
                       'Below_2SD_Correct_Direction': mlp_results['Below_2SD_Correct_Direction'].mean()}

    print("Ticker: ", ticker)
    print("lm")
    print(lm_performance)
    print("rf")
    print(rf_performance)
    print("mlp")
    print(mlp_performance)
    print()
