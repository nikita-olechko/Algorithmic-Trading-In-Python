import os
import pickle
from itertools import groupby, count


def occurences_more_than_Z_periods_apart(dataframe, column_name='Actual', Z_periods=60):
    indices = list(dataframe[dataframe[column_name] == 1].index)
    if len(indices) == 0:
        return []
    separate_occurences = []
    first_occurence_in_sequence = indices[0]
    for index, occurence in enumerate(indices):
        if index + 1 < len(indices):
            if indices[index + 1] - occurence > Z_periods:
                separate_occurences.append(first_occurence_in_sequence)
                first_occurence_in_sequence = indices[index + 1]
        else:
            separate_occurences.append(occurence)
    return separate_occurences


def incorrect_detections_not_within_Z_periods_of_correct_detection(dataframe, Z_periods):
    actual_occurences = occurences_more_than_Z_periods_apart(dataframe, column_name='Actual', Z_periods=Z_periods)
    predicted_occurences = occurences_more_than_Z_periods_apart(dataframe, column_name='Predicted', Z_periods=Z_periods)
    incorrect_detections = []
    for detection in predicted_occurences:
        # find the closest value in actual_occurences
        closest_value = min(actual_occurences, key=lambda x: abs(x - detection))
        if abs(closest_value - detection) > Z_periods:
            incorrect_detections.append(detection)
    return incorrect_detections


def incorrect_detection_within_allowable_error(results_dataframe, max_percent_change, Z_periods, X_percentage,
                                               allowable_error):
    # we need the predicted values
    # we need the actual percentages
    predicted_indices = list(results_dataframe[results_dataframe['Predicted'] == 1].index)
    percent_changes_at_predicted_indices = results_dataframe.loc[predicted_indices, 'Percent Change']


def create_classification_report_name(Z_periods=60, X_percentage=3, model_type='lm', allowable_error=""):
    """
    A function to return the classification report name. Must be called in the directory above model_performance.
    """
    return f'model_performance/classification_price_change_{model_type}_{Z_periods}_periods_{X_percentage}_percent_threshold_{allowable_error}_percent_error.csv'


def model_exists(Z_periods=60, X_percentage=3, model_type='lm', allowable_error=""):
    """
    A function to check if a model exists. Must be called in the directory above model_performance.
    """
    return os.path.isfile(create_classification_report_name(Z_periods, X_percentage, model_type, allowable_error))


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
