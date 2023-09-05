import pandas as pd


def predict_based_on_model(barDataFrame, model_object):
    x_columns = list(barDataFrame.columns)

    always_redundant_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'Barcount', 'Orders',
                                'Position']

    for column in always_redundant_columns:
        x_columns.remove(column)

    barDataFrame = barDataFrame.dropna()
    barDataFrame.reset_index(drop=True, inplace=True)

    predict_price_change = model_object.predict(barDataFrame[x_columns])
    predict_price_change = predict_price_change.reshape(-1, 1)
    barDataFrame['Prediction'] = predict_price_change
    return barDataFrame
