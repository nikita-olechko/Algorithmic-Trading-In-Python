from backtestingUtilities.simulationUtilities import run_strategy_on_list_of_tickers
from strategies.greaterthan10barsma import sampleSMABuySellStrategy, generate10PeriodSMAWholeDataFrame

strategy_name = '10PeriodSMA_Test'
strategy_buy_or_sell_condition_function = sampleSMABuySellStrategy
generate_additional_data_function = generate10PeriodSMAWholeDataFrame
barsize = "1 min"
duration = "1 M"

run_strategy_on_list_of_tickers(strategy_name,
                                strategy_buy_or_sell_condition_function=strategy_buy_or_sell_condition_function,
                                generate_additional_data_function=generate_additional_data_function,
                                barsize=barsize, duration=duration,
                                very_large_data=False, try_errored_tickers=True
                                )
