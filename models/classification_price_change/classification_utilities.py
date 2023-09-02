import os
import pickle
from itertools import groupby, count


def occurences_more_than_120_periods_apart(dataframe, column_name='Actual'):
    # get the indices of all where dataframe['Actual'] == 1
    indices = dataframe[dataframe[column_name] == 1].index
    # count all indice groups where the difference between the first and last indice is greater than 120
    return sum(1 for _ in (list(g) for _, g in groupby(indices, key=lambda n, c=count(): n - next(c))) if len(_) > 120)


def correctly_identified_occurences_more_than_120_periods_apart(dataframe):
    actual_indices = dataframe[dataframe['Actual'] == 1].index
    # group these indices when the difference between the first and last indice greater than 120
    ending_group_indices = [list(g) for _, g in groupby(actual_indices, key=lambda n, c=count(): n - next(c)) if
                            len(_) > 120]
    # indices of th
    predicted_indices = dataframe[dataframe['Predicted'] == 1].index
    # count the number of predicted indices that are in within 60 of an ending group indice
    return sum(1 for _ in ending_group_indices if any(abs(_ - i) <= 60 for i in predicted_indices))


def create_classification_report_name(Z_periods=60, X_percentage=3, model_type='lm'):
    """
    A function to return the classification report name. Must be called in the directory above model_performance.
    """
    return f'model_performance/classification_price_change_{model_type}_{Z_periods}_periods_{X_percentage}_percent.csv'


def model_exists(Z_periods=60, X_percentage=3, model_type='lm'):
    """
    A function to check if a model exists. Must be called in the directory above model_performance.
    """
    return os.path.isfile(create_classification_report_name(Z_periods, X_percentage, model_type))


def get_model(model_creation_dict, model_type, symbol, Z_periods, X_percentage, barsize, duration, model_data=None,
              prepped_data_column_tuple=None):
    # Get the current working directory
    current_directory = os.getcwd()
    model_filename = f'model_objects/classification_price_change_{model_type}_{symbol}_{Z_periods}_periods_{X_percentage}_percent_change_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
    if os.path.isfile(os.path.join(current_directory, model_filename)):
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    else:
        model = model_creation_dict[model_type](symbol, save_model=True, data=model_data, X_percentage=X_percentage,
                                                Z_periods=Z_periods,
                                                barsize=barsize, duration=duration,
                                                prepped_data_column_tuple=prepped_data_column_tuple)
    return model
