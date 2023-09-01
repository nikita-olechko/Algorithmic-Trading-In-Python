import os
import pickle


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


def get_model(model_creation_dict, model_type, symbol, Z_periods, X_percentage, barsize, duration, model_data=None, prepped_data_column_tuple=None):
    # Get the current working directory
    current_directory = os.getcwd()
    model_filename = f'model_objects/classification_price_change_{model_type}_{symbol}_{Z_periods}_periods_{X_percentage}_percent_change_{barsize.replace(" ", "")}_{duration.replace(" ", "")}.pkl'
    if os.path.isfile(os.path.join(current_directory, model_filename)):
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    else:
        model = model_creation_dict[model_type](symbol, save_model=True, data=model_data,
                                                barsize=barsize, duration=duration,
                                                prepped_data_column_tuple=prepped_data_column_tuple)
    return model
