def create_classification_report_name(Z_periods=60, X_percentage=3):
    """
    A function to return the classification report name. Must be called in the directory above model_performance.
    """
    return f'model_performance/classification_price_change_{Z_periods}_periods_{X_percentage}_percent.csv'
