from backtestingUtilities.simulationUtilities import run_strategy_on_list_of_tickers
from strategies.greaterthan60barsma import sampleSMABuySellStrategy, generate60PeriodSMAWholeDataFrame

strategy_name = '60PeriodSMA_Test'
strategy_buy_or_sell_condition_function = sampleSMABuySellStrategy
generate_additional_data_function = generate60PeriodSMAWholeDataFrame

run_strategy_on_list_of_tickers(strategy_name,
                                strategy_buy_or_sell_condition_function=strategy_buy_or_sell_condition_function,
                                generate_additional_data_function=generate_additional_data_function,
                                barsize="1 min", duration="3 M", list_of_tickers=["AAPL", "MSFT", "AMZN"], very_large_data=True
                                )
