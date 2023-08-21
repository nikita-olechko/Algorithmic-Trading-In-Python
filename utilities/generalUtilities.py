import os
import pandas as pd
import datetime

from ib_insync import IB

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


def create_historical_data_file_name(ticker, barsize, duration):
    file_ticker = str.replace(ticker, " ", "")
    file_bar_size = str.replace(barsize, " ", "")
    file_duration = str.replace(duration, " ", "")
    filename = "Historical" + file_ticker + file_bar_size + file_duration
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


def initialize_ib_connection():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4000, clientId=get_tws_connection_id())
    except Exception:
        print("Could not connect to IBKR. Check that Trader Workstation or IB Gateway is running.")
    return ib