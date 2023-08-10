import os


def get_root_directory_as_path():
    wd = os.path.dirname(os.path.abspath(__file__))
    wd = wd[:-14]
    wd = wd.replace('/', '\\\\')
    return wd


ROOT_DIRECTORY = get_root_directory_as_path()
