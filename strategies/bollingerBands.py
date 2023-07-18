import pandas as pd


def generate_bollinger_bands(minuteDataFrame, period=20):
    # The row to be updated
    row_to_update = minuteDataFrame.index[-1]

    # Calculate the moving average and standard deviation over the last 20 rows
    average_last_20 = minuteDataFrame['Average'].tail(period).mean()
    std_dev_last_20 = minuteDataFrame['Average'].tail(period).std()

    # Update the moving average and standard deviation of the last row
    minuteDataFrame.at[row_to_update, 'MA_20'] = average_last_20
    minuteDataFrame.at[row_to_update, 'SD_20'] = std_dev_last_20

    # Calculate and update the upper and lower Bollinger Bands of the last row
    minuteDataFrame.at[row_to_update, 'UpperBB2SD'] = average_last_20 + 2 * std_dev_last_20
    minuteDataFrame.at[row_to_update, 'LowerBB2SD'] = average_last_20 - 2 * std_dev_last_20
    minuteDataFrame.at[row_to_update, 'UpperBB1SD'] = average_last_20 + std_dev_last_20
    minuteDataFrame.at[row_to_update, 'LowerBB1SD'] = average_last_20 - std_dev_last_20
    return minuteDataFrame
