import pandas as pd
import inspect
import re


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


def modify_func_from_last_row_to_all_rows(func):
    # Get the source code of the function
    source_code = inspect.getsource(func)

    # Replace the code using regular expressions
    modified_code = re.sub(r"barDataFrame\.loc\[barDataFrame\.index\[-1\], '(.*?)'\]",
                           r"barDataFrame['\1']",
                           source_code)

    # Define a new (local) function using the modified code
    exec(modified_code, globals())

    # Return the newly defined function
    return globals()[func.__name__]
