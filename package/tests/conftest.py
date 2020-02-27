import os

import pytest


@pytest.fixture(scope="module")
def data_test_dir():
    """Return path to directory with test data"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_data")
