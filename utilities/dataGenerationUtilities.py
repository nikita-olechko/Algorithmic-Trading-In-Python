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


def create_log_price_variables(stk_data, list_of_periods=range(1, 11)):
    """
    Create log price and related variables for a given DataFrame.

    :param stk_data: DataFrame containing stock data.
    :param list_of_periods: List of periods to calculate shifted log prices.
    :return: Modified DataFrame with log price variables.
    """
    log_price = np.log(stk_data["Average"])
    stk_data["log_price"] = log_price

    for period in list_of_periods:
        shifted_log_price = stk_data["log_price"].shift(period)
        stk_data[f'{period}period_shifted_log_price'] = shifted_log_price
        stk_data[f'{period}period_change_in_log_price'] = stk_data["log_price"] - shifted_log_price
        stk_data[f'{period}period_percentage_change_in_log_price'] = (
                stk_data[f'{period}period_change_in_log_price'] / shifted_log_price * 100
        )

    return stk_data


def create_price_variables(stk_data, list_of_periods=range(1, 11)):
    """
    Create price change and related variables for a given DataFrame.

    :param stk_data: DataFrame containing stock data.
    :param list_of_periods: List of periods to calculate shifted prices.
    :return: Modified DataFrame with price change variables.
    """
    for period in list_of_periods:
        shifted_price = stk_data["Average"].shift(period)
        stk_data[f'{period}period_shifted_price'] = shifted_price
        stk_data[f'{period}period_change_in_price'] = stk_data["Average"] - shifted_price
        stk_data[f'{period}period_percentage_change_in_price'] = (
                stk_data[f'{period}period_change_in_price'] / shifted_price * 100
        )
        stk_data[f'sum_of_absolute_percentage_price_changes_over_{period}_periods'] = stk_data[
            f'{period}period_percentage_change_in_price'].abs().rolling(window=period).sum()
    return stk_data


def create_volume_change_variables(stk_data, list_of_periods=range(1, 11)):
    """
    Create log volume and related variables for a given DataFrame.

    :param stk_data: DataFrame containing stock data.
    :param list_of_periods: List of periods to calculate shifted log volumes.
    :return: Modified DataFrame with log volume variables.
    """
    log_volume = np.log(stk_data["Volume"])
    stk_data["log_volume"] = log_volume
    for period in list_of_periods:
        shifted_log_volume = stk_data["log_volume"].shift(period)
        stk_data[f'{period}period_shifted_log_volume'] = shifted_log_volume
        stk_data[f'{period}period_change_in_log_volume'] = stk_data["log_volume"] - shifted_log_volume
    return stk_data


def generate_bollinger_bands(dataFrame, period=20):
    """
    Generate Bollinger Bands based on moving averages and standard deviations.

    :param dataFrame: DataFrame containing stock data.
    :param period: Period for calculating moving averages and standard deviations.
    :return: DataFrame with Bollinger Bands columns added.
    """
    # Calculate the moving average and standard deviation over the last 'period' rows
    dataFrame['MA_20'] = dataFrame['Average'].rolling(window=period).mean()
    dataFrame['SD_20'] = dataFrame['Average'].rolling(window=period).std()

    # Calculate the Bollinger Bands
    dataFrame['UpperBB2SD'] = dataFrame['MA_20'] + 2 * dataFrame['SD_20']
    dataFrame['LowerBB2SD'] = dataFrame['MA_20'] - 2 * dataFrame['SD_20']
    dataFrame['UpperBB1SD'] = dataFrame['MA_20'] + dataFrame['SD_20']
    dataFrame['LowerBB1SD'] = dataFrame['MA_20'] - dataFrame['SD_20']

    return dataFrame


def boolean_bollinger_band_location(minuteDataFrame):
    """
    Determine whether prices are above or below Bollinger Bands.

    :param minuteDataFrame: DataFrame containing stock data.
    :return: DataFrame with Boolean columns indicating Bollinger Bands positions.
    """
    minuteDataFrame['PriceAboveUpperBB2SD'] = np.where(minuteDataFrame['Average'] > minuteDataFrame['UpperBB2SD'], 1, 0)
    minuteDataFrame['PriceAboveUpperBB1SD'] = np.where(
        (minuteDataFrame['Average'] > minuteDataFrame['UpperBB1SD']) & (minuteDataFrame['PriceAboveUpperBB2SD'] == 0),
        1, 0)
    minuteDataFrame['PriceBelowLowerBB2SD'] = np.where(minuteDataFrame['Average'] < minuteDataFrame['LowerBB2SD'], 1, 0)
    minuteDataFrame['PriceBelowLowerBB1SD'] = np.where(
        (minuteDataFrame['Average'] < minuteDataFrame['LowerBB1SD']) & (minuteDataFrame['PriceBelowLowerBB2SD'] == 0),
        1, 0)
    return minuteDataFrame
