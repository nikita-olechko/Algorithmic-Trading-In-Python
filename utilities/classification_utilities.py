import os
import pickle


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


def create_classification_report_name(Z_periods=60, X_percentage=3, model_type='lm', allowable_error=""):
    """
    A function to return the classification report name. Must be called in the directory above model_performance.
    """
    return f'model_performance/classification_price_change_{model_type}_{Z_periods}_periods_{X_percentage}_percent_threshold_{allowable_error}_percent_error.csv'


def get_model_name(symbol, barsize, duration, Z_periods=60, X_percentage=3, model_type='lm'):
    return f'classification_price_change_{model_type}_{symbol}_{Z_periods}_periods_{X_percentage}_percent_change_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'


def model_file_path(symbol, barsize, duration, Z_periods, X_percentage, model_type='rf', directory_offset=1):
    # Get the current working directory
    current_directory = os.getcwd()

    # Calculate the new directory path with offset
    current_directory = os.path.abspath(os.path.join(current_directory, "../" * directory_offset))
    folder_path = os.path.join(current_directory, "models/classification_price_change/model_objects")
    file_name = get_model_name(symbol, barsize, duration, Z_periods=Z_periods, X_percentage=X_percentage,
                               model_type=model_type)
    return os.path.join(folder_path, file_name)


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


def get_model_object(symbol='NKLA', model_barsize='1 min', model_duration='12 M', Z_periods=120,
                     X_percentage=1.5):
    model_file = model_file_path(symbol, model_barsize, model_duration, Z_periods, X_percentage,
                                 model_type='rf', directory_offset=1)

    model_object = pickle.load(open(model_file, 'rb'))
    return model_object


def detection_indices_by_correctness(results, data, Z_periods, X_percentage, allowable_error):
    strictly_incorrect_detection_indices = incorrect_detections_not_within_Z_periods_of_correct_detection(results,
                                                                                                          Z_periods)

    detections_within_error = []
    detections_outside_error = []

    for index in strictly_incorrect_detection_indices:
        max_percentage_price_change = data[f'maximum_percentage_price_change_over_next_{Z_periods}'][index]
        if max_percentage_price_change >= (X_percentage * allowable_error / 100):
            detections_within_error.append(index)
        else:
            detections_outside_error.append(index)

    return detections_within_error, detections_outside_error


def price_changes_after_incorrect_detections(results, Z_periods, X_percentage, data, allowable_error,
                                             detections_within_error=None, detections_outside_error=None):
    if detections_within_error is None or detections_outside_error is None:
        detections_within_error, detections_outside_error = detection_indices_by_correctness(results, data, Z_periods,
                                                                                             X_percentage, allowable_error)
    if len(detections_outside_error) == 0:
        return []
    indices_after_detection_outside_error = [index + Z_periods for index in detections_outside_error]
    if indices_after_detection_outside_error[-1] > len(data):
        indices_after_detection_outside_error = indices_after_detection_outside_error[:-1]
        detections_outside_error = detections_outside_error[:-1]

    price_at_detection_outside_error = list(data['Average'][detections_outside_error])
    price_at_detection_after_Z_periods = list(data['Average'][indices_after_detection_outside_error])

    price_change_percentage_after_Z_periods_incorrect_detection = [(price_at_detection_after_Z_periods[index] -
                                                                    price_at_detection_outside_error[index]) / \
                                                                   price_at_detection_outside_error[index] * 100
                                                                   for index in
                                                                   range(len(price_at_detection_outside_error))]

    return price_change_percentage_after_Z_periods_incorrect_detection
