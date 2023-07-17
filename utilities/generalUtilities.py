import os
import pandas as pd
from visualizations.__init__ import ROOT_DIRECTORY


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
                          folderPath="C:\\Users\\nikit\\OneDrive - BCIT\\R Projects\\Cleaning Up Automated Trading\\data\\"):
    file_path = os.path.join(folderPath, filename)
    if os.path.exists(file_path):
        print(filename, " exists")
        return True
    else:
        return False


def get_starter_order_id(n):
    return n*1000000
