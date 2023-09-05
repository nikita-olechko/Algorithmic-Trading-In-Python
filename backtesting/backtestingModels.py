from backtestingUtilities.simulationUtilities import run_strategy_on_list_of_tickers
from strategies.classification_price_change_single_stock import generate_model_data, classification_model_strategy
from utilities.classification_utilities import get_model_object

strategy_name = 'ClassificationModel_1.5_PercentThreshold'
strategy_condition_function = generate_model_data
generate_data_function = classification_model_strategy
barsize = "1 min"
duration = "3 M"
model_barsize = '1 min'
model_duration = '12 M'
Z_periods = 120
X_percentage = 1.5

for symbol in ['NKLA', 'SPCE', 'PTON', 'AMC', 'PLUG']:
    model_object = get_model_object(symbol=symbol, model_barsize=model_barsize, model_duration=model_duration,
                                    Z_periods=Z_periods, X_percentage=X_percentage)

    run_strategy_on_list_of_tickers(strategy_name,
                                    strategy_buy_or_sell_condition_function=strategy_condition_function,
                                    generate_additional_data_function=generate_data_function,
                                    barsize=barsize, duration=duration, list_of_tickers=[symbol],
                                    very_large_data=True, try_errored_tickers=True, model_object=model_object
                                    )
