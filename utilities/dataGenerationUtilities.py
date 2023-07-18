import pandas as pd


def average_bars_by_minute(barDataFrame, minuteDataFrame, non_numeric_columns=None):
    """
    A function to transform continous bar data into minute by minute data
    :param barDataFrame: a Pandas Dataframe with barData from IBKR
    :param minuteDataFrame: a Pandas Dataframe
    :param non_numeric_columns: a list with columns not to take the average of
    :precondition: non_numeric_columns must be a list of strings
    :return: A Pandas Dataframe with the last full minute of barData averaged
    """
    second_last_date = barDataFrame.iloc[-2]["Date"]
    # If we have entered a new minute
    if barDataFrame.iloc[-1]["Date"] != second_last_date:
        # Get all rows where the "Date" column matches the second last date
        matching_rows = barDataFrame[barDataFrame["Date"] == second_last_date]
        if non_numeric_columns is None:
            non_numeric_columns = []
        non_numeric_columns += "Date", "Orders"
        matching_numeric_rows = matching_rows.drop(columns=non_numeric_columns)
        average_row = matching_numeric_rows.mean()
        for non_numeric in non_numeric_columns:
            average_row[non_numeric] = barDataFrame.iloc[-2][non_numeric]
        minuteDataFrame = pd.concat([minuteDataFrame, pd.DataFrame([average_row])], ignore_index=True)
    return minuteDataFrame


# barDataFrame = pd.read_csv(
#     'C:\\Users\\nikit\OneDrive\Personal Projects\Algorithmic Trading\Live-Algorithmic-Trading-In-Python\liveTrading\\barDataFrame.csv')
# barDataFrame = barDataFrame.iloc[:-1]
# barDataFrame['Orders'] = ""
#
# minuteDataFrame = pd.DataFrame()
# minuteDataFrame = average_bars_by_minute(barDataFrame, minuteDataFrame)
#
# print("Done")
