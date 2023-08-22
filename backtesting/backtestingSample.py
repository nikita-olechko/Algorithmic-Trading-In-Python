from ib_insync import IB

from backtestingUtilities.simulationUtilities import run_strategy_on_list_of_tickers
from strategies.greaterthan60barsma import sampleSMABuySellStrategy, generate60PeriodSMA_backtest
from utilities.generalUtilities import initialize_ib_connection

ib = initialize_ib_connection()

strategy_name = '60PeriodSMA'
strategy_buy_or_sell_condition_function = sampleSMABuySellStrategy
generate_additional_data_function = generate60PeriodSMA_backtest

run_strategy_on_list_of_tickers(ib, strategy_name,
                                strategy_buy_or_sell_condition_function=strategy_buy_or_sell_condition_function,
                                generate_additional_data_function=generate_additional_data_function,
                                barsize="1 Min", duration="1 D"
                                )
