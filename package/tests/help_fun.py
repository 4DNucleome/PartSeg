import os

def get_test_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_data")
