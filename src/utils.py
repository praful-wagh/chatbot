import os, sys
import pickle
from src.exception import CustomException

def getPath():
    current_dir = os.getcwd()
    project_root = os.path.abspath(current_dir)
    while not os.path.isfile(os.path.join(project_root, 'README.md')):
        project_root = os.path.dirname(project_root)
    return project_root

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)