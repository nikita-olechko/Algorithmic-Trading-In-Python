import pandas as pd
import os
import gc
from ib_insync import util, Contract

from utilities.__init__ import DATE_FORMAT
from utilities.generalUtilities import get_months_of_historical_data, initialize_ib_connection


def run_strategy_on_list_of_tickers(strategy, strategy_buy_or_sell_condition_function,
                                    generate_additional_data_function=None,
                                    barsize="1 day", duration="3 Y", what_to_show="TRADES", list_of_tickers=None,
                                    initializing_order=1,
                                    directory_offset=1, months_offset=0, very_large_data=False,
                                    ticker_limit=None, try_errored_tickers=False,
                                    model=None,
                                    *args, **kwargs):
    """
        Run a trading strategy on a list of tickers using historical data.

        :param ib: An instance of the IB API used for data retrieval.
        :type ib: IBApi

        :param strategy: The name of the trading strategy being applied.
        :type strategy: str

        :param strategy_buy_or_sell_condition_function: A function defining the conditions for placing buy or sell orders.
        :type strategy_buy_or_sell_condition_function: callable

        :param generate_additional_data_function: A function that generates additional data for analysis. (optional)
        :type generate_additional_data_function: callable, optional

        :param barsize: The time interval for bars in the historical data. Default is "1 day".
        :type barsize: str

        :param duration: The duration of historical data to retrieve. Default is "3 Y" (3 years).
        :type duration: str

        :param what_to_show: The data type to show in the historical data. Default is "TRADES".
        :type what_to_show: str

        :param list_of_tickers: A list of tickers to run the strategy on. If None, reads tickers from a CSV file.
        :type list_of_tickers: list or None

        :param initializing_order: The value indicating the order type to use for initializing the first position. Default is 1 (buy).
        :type initializing_order: int

        :param directory_offset: The offset for the data directory. Default is 1.
        :type directory_offset: int

        :param months_offset: The offset for the duration in months. Default is 0.
        :type months_offset: int

        :param very_large_data: Whether to retrieve very large historical data. Default is False.
        :type very_large_data: bool

        :param ticker_limit: The limit on the number of tickers to process. Default is None (no limit).
        :type ticker_limit: int or None

        :param try_errored_tickers: Whether to try processing tickers that previously encountered errors. Default is False.
        :type try_errored_tickers: bool

        :param args: Additional positional arguments passed to strategy functions.
        :param kwargs: Additional keyword arguments passed to strategy functions.
    """
    ib = initialize_ib_connection()

    if list_of_tickers is None:
        list_of_tickers = pd.read_csv("../backtesting/nyse-listed.csv")['ACT Symbol']
        if ticker_limit is not None:
            list_of_tickers = list_of_tickers[:ticker_limit]
    try:
        erred_tickers = pd.read_csv("../backtesting/data/ErroredTickers/ErroredTickers.csv", header=None,
                                    names=['Ticker'])
    except FileNotFoundError:
        erred_tickers = pd.DataFrame(columns=['Ticker'])
    folder_path = os.path.join(os.getcwd(), "../backtesting/data", "Strategy Results")
    summary_file_path_name = os.path.join(folder_path,
                                          f"{strategy.replace(' ', '')}{barsize.replace(' ', '')}{duration.replace(' ', '')}.csv")
    if os.path.exists(summary_file_path_name):
        all_tickers_summary = pd.read_csv(summary_file_path_name)
    else:
        all_tickers_summary = pd.DataFrame()

    completed_tickers = all_tickers_summary['ticker'].unique() if 'ticker' in all_tickers_summary.columns else []
    if not try_errored_tickers:
        list_of_tickers = [ticker for ticker in list_of_tickers if
                           ticker not in completed_tickers and ticker not in erred_tickers['Ticker'].values]
    else:
        list_of_tickers = [ticker for ticker in list_of_tickers if ticker not in completed_tickers]

    for ticker in list_of_tickers:
        gc.collect()
        stk_data = retrieve_base_data(ib, ticker, barsize=barsize, duration=duration, directory_offset=directory_offset,
                                      what_to_show=what_to_show, months_offset=months_offset,
                                      very_large_data=very_large_data)

        if stk_data is not None:
            summary_df = simulate_trading_on_strategy(stk_data, ticker, strategy_buy_or_sell_condition_function,
                                                      generate_additional_data_function=generate_additional_data_function,
                                                      initializing_order=initializing_order, model=model
                                                      *args, **kwargs)
            all_tickers_summary = pd.concat([all_tickers_summary, summary_df])
            all_tickers_summary.to_csv(summary_file_path_name, index=False)
            print(f"Completed {ticker}")
    print("Completed all tickers")


def simulate_trading_on_strategy(stk_data, ticker, strategy_buy_or_sell_condition_function,
                                 generate_additional_data_function=None,
                                 initializing_order=1, model=None, *args, **kwargs):
    if generate_additional_data_function is not None:
        if model is not None:
            stk_data = generate_additional_data_function(stk_data, model)
        else:
            stk_data = generate_additional_data_function(stk_data)

    last_order_index = 0

    for index, row in stk_data.iterrows():
        if row.isna().any():
            continue
        # Get order based on strategy conditions
        order = strategy_buy_or_sell_condition_function(stk_data[:index], ticker=ticker, current_index=index,
                                                        last_order_index=last_order_index, *args, **kwargs)

        # If this is the first order, wait until first Buy order to establish the position
        if last_order_index == 0:
            if order == initializing_order:
                stk_data.loc[index, 'Orders'] = 1
                stk_data.loc[index, 'Position'] = - stk_data.loc[index, "Average"]
                last_order_index = index
        else:
            stk_data, last_order_index = order_selector(order)(stk_data, index, last_order_index)

        # If we are at the last order, and the last order was a buy, sell the position
        if index == len(stk_data) - 1 and stk_data.loc[last_order_index, 'Orders'] == 1:
            stk_data.loc[index, 'Orders'] = -1
            last_order_index = index
            stk_data.loc[index, 'Position'] = stk_data.loc[index - 1, 'Position'] + stk_data.loc[index, "Average"]

    summary_df = create_summary_data(stk_data, ticker)
    return summary_df


def create_summary_data(stk_data, ticker, summary_df=None):
    stk_data['holdingNetReturn'] = stk_data["Average"] / stk_data["Average"].iloc[0]

    trade_completed_indices = stk_data.loc[stk_data['Orders'] == -1].index
    final_position = stk_data['Position'].iloc[-1]
    final_position_percentage_of_price = final_position / stk_data['Average'].iloc[0]

    average_position_post_trade = stk_data['Position'].loc[trade_completed_indices].mean()
    sd_position_post_trade = stk_data['Position'].loc[trade_completed_indices].std()
    max_position_post_trade = stk_data['Position'].loc[trade_completed_indices].max()
    min_position_post_trade = stk_data['Position'].loc[trade_completed_indices].min()

    changes_in_position_per_trade = stk_data['Position'].loc[trade_completed_indices].diff().dropna()

    average_change_in_position_per_trade = changes_in_position_per_trade.mean()
    sd_change_in_position_per_trade = changes_in_position_per_trade.std()
    min_change_in_position_per_trade = changes_in_position_per_trade.min()
    max_change_in_position_per_trade = changes_in_position_per_trade.max()

    average_position_post_trade_percentage = stk_data['Position'].loc[trade_completed_indices].mean() / \
                                             stk_data["Average"].iloc[0] * 100
    sd_position_post_trade_percentage = stk_data['Position'].loc[trade_completed_indices].std() / \
                                        stk_data["Average"].iloc[0] * 100
    max_position_post_trade_percentage = stk_data['Position'].loc[trade_completed_indices].max() / \
                                         stk_data["Average"].iloc[0] * 100
    min_position_post_trade_percentage = stk_data['Position'].loc[trade_completed_indices].min() / \
                                         stk_data["Average"].iloc[0] * 100

    average_change_in_position_per_trade_percentage = changes_in_position_per_trade.mean() / stk_data["Average"].iloc[
        0] * 100
    sd_change_in_position_per_trade_percentage = changes_in_position_per_trade.std() / stk_data["Average"].iloc[0] * 100
    min_change_in_position_per_trade_percentage = changes_in_position_per_trade.min() / stk_data["Average"].iloc[
        0] * 100
    max_change_in_position_per_trade_percentage = changes_in_position_per_trade.max() / stk_data["Average"].iloc[
        0] * 100

    final_holding_net_return = stk_data['holdingNetReturn'].iloc[-1]
    average_holding_net_return = stk_data['holdingNetReturn'].mean()
    sd_holding_net_return = stk_data['holdingNetReturn'].std()
    min_holding_net_return = stk_data['holdingNetReturn'].min()
    max_holding_net_return = stk_data['holdingNetReturn'].max()
    new_summary = pd.DataFrame({
        'ticker': [ticker],
        'finalPositionAsPercentage': [final_position_percentage_of_price],
        'AveragePositionPostTradeAsPercentage': [average_position_post_trade_percentage],
        'SDPositionPostTradeAsPercentage': [sd_position_post_trade_percentage],
        'MaxPositionPostTradeAsPercentage': [max_position_post_trade_percentage],
        'MinPositionPostTradeAsPercentage': [min_position_post_trade_percentage],
        'AvgChangeInPositionAsPercentage': [average_change_in_position_per_trade_percentage],
        'SDChangeInPositionAsPercentage': [sd_change_in_position_per_trade_percentage],
        'MinChangeInPositionAsPercentage': [min_change_in_position_per_trade_percentage],
        'MaxChangeInPositionAsPercentage': [max_change_in_position_per_trade_percentage],
        'finalPosition': [final_position],
        'AveragePositionPostTrade': [average_position_post_trade],
        'SDPositionPostTrade': [sd_position_post_trade],
        'MaxPositionPostTrade': [max_position_post_trade],
        'MinPositionPostTrade': [min_position_post_trade],
        'AvgChangeInPositionPerTrade': [average_change_in_position_per_trade],
        'SDChangeInPositionPerTrade': [sd_change_in_position_per_trade],
        'MinChangeInPositionPerTrade': [min_change_in_position_per_trade],
        'MaxChangeInPositionPerTrade': [max_change_in_position_per_trade],
        'FinalHoldingNetReturn': [final_holding_net_return],
        'AverageHoldingNetReturn': [average_holding_net_return],
        'SDHoldingNetReturn': [sd_holding_net_return],
        'MinHoldingNetReturn': [min_holding_net_return],
        'MaxHoldingNetReturn': [max_holding_net_return],
        'NumberOfTradesComplete': [len(trade_completed_indices)]
    })

    if summary_df is not None:
        new_summary = pd.concat([summary_df, new_summary])

    return new_summary


def retrieve_base_data(ib, ticker, barsize="1 day", duration="3 Y", what_to_show="TRADES", directory_offset=0,
                       endDateTime='', very_large_data=False, months_offset=0, try_errored_tickers=False):
    if not try_errored_tickers:
        csv_file_path = "../backtesting/data/ErroredTickers/ErroredTickers.csv"
        try:
            erred_tickers = pd.read_csv(csv_file_path)
            if ticker in erred_tickers['Ticker'].values:
                return None
        except FileNotFoundError as e:
            print("Warning: Error retrieving errored ticker list |", e)
            erred_tickers = pd.DataFrame(columns=['Ticker'])

    if very_large_data:
        stk_data = get_months_of_historical_data(ib, ticker, months=int(duration.split(" ")[0]), barsize=barsize,
                                                 what_to_show=what_to_show,
                                                 directory_offset=directory_offset, months_offset=months_offset
                                                 )
    else:
        stk_data = get_stock_data(ib, ticker, barsize=barsize, duration=duration, what_to_show=what_to_show,
                                  directory_offset=directory_offset, endDateTime=endDateTime)
    # TODO FIX: does not work atm as cannot find the file
    # if stk_data is None:
    #     csv_file_path = "../backtesting/data/ErroredTickers/ErroredTickers.csv"
    #     try:
    #         erred_tickers = pd.read_csv(csv_file_path)
    #     except FileNotFoundError:
    #         erred_tickers = pd.DataFrame(columns=['Ticker'])
    #     erred_tickers = pd.concat([erred_tickers, pd.DataFrame({'Ticker': [ticker]})], ignore_index=True)
    #     erred_tickers.to_csv(csv_file_path, index=False)

    return stk_data


def get_stock_data(ib, ticker, barsize='1 min', duration='1 M', what_to_show='TRADES', directory_offset=0,
                   endDateTime=''):
    contract = Contract()
    contract.symbol = ticker
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    contract.primaryExchange = 'NYSE'
    ticker = contract.symbol
    file_name = create_historical_data_file_name(ticker, barsize, duration, endDateTime)

    # Get the current working directory
    current_directory = os.getcwd()

    # Calculate the new directory path with offset
    new_directory = os.path.abspath(os.path.join(current_directory, "../" * directory_offset))
    folder_path = os.path.join(new_directory, "backtesting/data", "Historical Data")

    # If the data already exists, retrieve it
    if os.path.isfile(os.path.join(folder_path, file_name)):
        try:
            stk_data = pd.read_csv(os.path.join(folder_path, file_name), parse_dates=True, index_col=0,
                                   date_format=DATE_FORMAT)
        except Exception as e:
            print("An error occurred retrieving the file:", str(e))
            stk_data = None
    else:
        stk_data = None
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=endDateTime,
                durationStr=duration,
                barSizeSetting=barsize,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1)
            stk_data = util.df(bars)
            stk_data.columns = stk_data.columns.str.title()
            # if new data acquired, write it to the historical data folder
            if len(stk_data) <= 50:
                return None
            stk_data = add_analysis_data_to_historical_data(stk_data, ticker)
            stk_data.to_csv(os.path.join(folder_path, file_name))
            print("Historical Data Created")
        except Exception as e:
            print("An error occurred:", str(e))
            return None
    return stk_data


def add_analysis_data_to_historical_data(stk_data, ticker):
    stk_data["Orders"] = 0
    stk_data["Position"] = 0
    return stk_data


def create_historical_data_file_name(ticker, barsize, duration, endDateTime=''):
    file_ticker = ticker.replace(" ", "")
    file_barsize = barsize.replace(" ", "")
    file_duration = duration.replace(" ", "")
    endDateTime = endDateTime.split(" ")[0]
    filename = f"Historical{file_ticker}{file_barsize}{file_duration}{endDateTime}"
    return filename


def order_selector(order):
    """
    A function to select the correct order function based on the order
    """
    order = str(order)
    order_dict = {
        "1": buy_order,
        "-1": sell_order,
        "2": hold_order,
        "0": nothing_order
    }
    return order_dict[order]


def buy_order(stk_data, index, last_order_index):
    stk_data.at[index, 'Orders'] = 1
    last_order_index = index
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position'] - stk_data.at[index, "Average"]
    return stk_data, last_order_index


def sell_order(stk_data, index, last_order_index):
    stk_data.at[index, 'Orders'] = -1
    last_order_index = index
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position'] + stk_data.at[index, "Average"]
    return stk_data, last_order_index


def hold_order(stk_data, index, last_order_index):
    stk_data.at[index, 'Orders'] = 2
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position']
    return stk_data, last_order_index


def nothing_order(stk_data, index, last_order_index):
    stk_data.at[index, 'Position'] = stk_data.at[index - 1, 'Position']
    return stk_data, last_order_index
