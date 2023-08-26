import os

import pandas as pd
import datetime
from ib_insync import IB, util, Contract

# from backtesting.backtestingUtilities.simulationUtilities import add_analysis_data_to_historical_data
from utilities.__init__ import ROOT_DIRECTORY


def retrieve_stored_historical_data(ticker, barsize="1 day", duration="1 Y"):
    print("entered getStockData")
    file_name = create_historical_data_file_name(ticker, barsize, duration)
    folder_path = ROOT_DIRECTORY + "\\data\\Historical Data\\"
    if file_exists_in_folder(file_name, folder_path):
        file_path_name = folder_path + file_name + ".csv"
        try:
            stk_data = pd.read_csv(file_path_name)
            # stk_data = stk_data.melt
            return stk_data
        except Exception as e:
            print(e)
            return None
    print(f"{file_name} not found")
    return None


def create_historical_data_file_name(ticker, barsize, duration, endDateTime=''):
    file_ticker = str.replace(ticker, " ", "")
    file_bar_size = str.replace(barsize, " ", "")
    file_duration = str.replace(duration, " ", "")
    filename = "Historical" + file_ticker + file_bar_size + file_duration + endDateTime.split(" ")[0]
    return filename


def file_exists_in_folder(filename,
                          folderPath):
    file_path = os.path.join(folderPath, filename)
    if os.path.exists(file_path):
        print(filename, " exists")
        return True
    else:
        return False


def get_time_of_day_as_string():
    now = datetime.datetime.now()
    return now.strftime("%H%M%S")


def get_starter_order_id(n=[]):
    time = get_time_of_day_as_string()
    n.append(1)
    return int(time + str(len(n) * 1000000))


def get_tws_connection_id(n=[]):
    time = get_time_of_day_as_string()
    n.append(1)
    return int(time + str(len(n)))


def initialize_ib_connection(port=4000):
    ib = IB()
    try:
        ib.connect('127.0.0.1', port, clientId=get_tws_connection_id())
        print("Connected to IBKR")
    except Exception as e:
        print(e)
        print("Could not connect to IBKR. Check that Trader Workstation or IB Gateway is running.")
    return ib


def ibkr_query_time_months(month_offset=0):
    today = datetime.datetime.now()
    end_date = datetime.datetime(today.year, today.month, 1) - datetime.timedelta(days=1)
    end_date -= datetime.timedelta(days=30 * month_offset)

    return end_date.strftime("%Y%m%d %H:%M:%S")


def ibkr_query_time_days(day_offset=0):
    today = datetime.datetime.now()
    end_date = datetime.datetime(today.year, today.month, today.day) - datetime.timedelta(days=1)
    end_date -= datetime.timedelta(days=day_offset)

    return end_date.strftime("%Y%m%d %H:%M:%S")


def get_months_of_historical_data(ib, ticker, months=12, barsize='1 Min', what_to_show='TRADES',
                                  directory_offset=0, months_offset=0):
    duration = '1 M'
    contract = Contract()
    contract.symbol = ticker
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    contract.primaryExchange = 'NYSE'
    ticker = contract.symbol
    file_name = create_historical_data_file_name(ticker, barsize, duration=f"{months}M",
                                                 endDateTime=ibkr_query_time_months(0 + months_offset))

    # Get the current working directory
    current_directory = os.getcwd()

    # Calculate the new directory path with offset
    current_directory = os.path.abspath(os.path.join(current_directory, "../" * directory_offset))
    folder_path = os.path.join(current_directory, "backtesting/data", "Historical Data")

    # If the data already exists, retrieve it
    if os.path.isfile(os.path.join(folder_path, file_name)):
        try:
            stk_data = pd.read_csv(os.path.join(folder_path, file_name), parse_dates=True, index_col=0)
        except Exception as e:
            print("An error occurred retrieving the file:", str(e))
            stk_data = None
    else:
        stk_data = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount'])
        for month in range(months):
            endDateTime = ibkr_query_time_months(month + months_offset)
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=endDateTime,
                    durationStr=duration,
                    barSizeSetting=barsize,
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=1)
                incremental_data = util.df(bars)
                incremental_data.columns = incremental_data.columns.str.title()
                if len(incremental_data) <= 50:
                    incremental_data = None
                stk_data = pd.merge(left=incremental_data, right=stk_data, how='outer')
                print(f"Month {month} of {months} complete.")
            except Exception as e:
                print("An error occurred:", str(e))
                print(f"Month {month} of {months} skipped.")
        try:
            stk_data = stk_data.drop_duplicates()
            stk_data = stk_data.sort_values(by=['Date'])
            stk_data["Orders"] = 0
            stk_data["Position"] = 0
            stk_data.to_csv(os.path.join(folder_path, file_name))
            print("Historical Data Created")
        except Exception as e:
            print("An error occurred:", str(e))
            print(f"Historical Data for {ticker} NOT Created")
    if len(stk_data) <= 50:
        stk_data = None
    return stk_data


def get_days_of_historical_data(ib, ticker, days=1, barsize='1 secs', what_to_show='TRADES',
                                directory_offset=0):
    duration = '1 D'
    contract = Contract()
    contract.symbol = ticker
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    contract.primaryExchange = 'NYSE'
    ticker = contract.symbol
    file_name = create_historical_data_file_name(ticker, barsize, duration=f"{days}D",
                                                 endDateTime=ibkr_query_time_days(0))

    # Get the current working directory
    current_directory = os.getcwd()

    # Calculate the new directory path with offset
    current_directory = os.path.abspath(os.path.join(current_directory, "../" * directory_offset))
    folder_path = os.path.join(current_directory, "backtesting/data", "Historical Data")

    # If the data already exists, retrieve it
    if os.path.isfile(os.path.join(folder_path, file_name)):
        try:
            stk_data = pd.read_csv(os.path.join(folder_path, file_name), parse_dates=True, index_col=0)
        except Exception as e:
            print("An error occurred retrieving the file:", str(e))
            stk_data = None
    else:
        stk_data = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount'])
        for day in range(days):
            endDateTime = ibkr_query_time_days(day)
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=endDateTime,
                    durationStr=duration,
                    barSizeSetting=barsize,
                    whatToShow=what_to_show,
                    useRTH=True,
                    formatDate=1)
                incremental_data = util.df(bars)
                incremental_data.columns = incremental_data.columns.str.title()
                if len(incremental_data) <= 50:
                    incremental_data = None
                stk_data = pd.merge(left=incremental_data, right=stk_data, how='outer')
                print(f"Day {day} of {days} complete.")
            except Exception as e:
                print("An error occurred:", str(e))
                print(f"Day {day} of {days} skipped.")
        stk_data = stk_data.drop_duplicates()
        stk_data = stk_data.sort_values(by=['Date'])
        stk_data["Orders"] = 0
        stk_data["Position"] = 0
        stk_data.to_csv(os.path.join(folder_path, file_name))
        print("Historical Data Created")
    if len(stk_data) <= 50:
        stk_data = None
    return stk_data
