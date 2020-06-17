import dataclasses
import itertools
import os
from copy import deepcopy

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import SegmentationProfile
from PartSegCore.image_operations import RadiusType
from PartSegImage import Image
from PartSegCore.mask.io_functions import SegmentationTuple
from PartSegCore.mask_create import MaskProperty


@pytest.fixture(scope="module")
def data_test_dir():
    """Return path to directory with test data"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_data")


@pytest.fixture
def image():
    data = np.zeros([20, 20, 20, 2], dtype=np.uint8)
    data[10:-1, 1:-1, 1:-1, 0] = 20
    data[1:10, 1:-1, 1:-1, 1] = 20
    data[1:-1, 1:5, 1:-1, 1] = 20
    data[1:-1, -5:-1, 1:-1, 1] = 20

    return Image(data, (1, 1, 1), axes_order="ZYXC")


@pytest.fixture
def stack_image():
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for x, y in itertools.product([0, 20], repeat=2):
        data[1:-1, x + 2 : x + 18, y + 2 : y + 18] = 100
    for x, y in itertools.product([0, 20], repeat=2):
        data[3:-3, x + 4 : x + 16, y + 4 : y + 16] = 120
    for x, y in itertools.product([0, 20], repeat=2):
        data[5:-5, x + 6 : x + 14, y + 6 : y + 14] = 140

    return SegmentationTuple("test_path", Image(data, (2, 1, 1), axes_order="ZYX", file_path="test_path"))


@pytest.fixture
def algorithm_parameters():
    algorithm_parameters = {
        "algorithm_name": "Lower threshold",
        "values": {
            "threshold": {"name": "Manual", "values": {"threshold": 10}},
            "channel": 0,
            "noise_filtering": {"name": "None", "values": {}},
            "minimum_size": 1,
            "side_connection": False,
        },
    }
    return deepcopy(algorithm_parameters)


@pytest.fixture
def mask_segmentation_parameters():
    return SegmentationProfile(
        name="",
        algorithm="Threshold",
        values={
            "channel": 0,
            "noise_filtering": {"name": "None", "values": {}},
            "threshold": {"name": "Manual", "values": {"threshold": 10}},
            "close_holes": False,
            "close_holes_size": 200,
            "smooth_border": {"name": "None", "values": {}},
            "side_connection": False,
            "minimum_size": 10000,
            "use_convex": False,
        },
    )


@pytest.fixture
def stack_segmentation1(stack_image: SegmentationTuple, mask_segmentation_parameters):
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for i, (x, y) in enumerate(itertools.product([0, 20], repeat=2), start=1):
        data[1:-1, x + 2 : x + 18, y + 2 : y + 18] = i
    data = stack_image.image.fit_array_to_image(data)
    parameters = {i: deepcopy(mask_segmentation_parameters) for i in range(1, 5)}
    return dataclasses.replace(
        stack_image, segmentation=data, segmentation_parameters=parameters, selected_components=[1, 3]
    )


@pytest.fixture
def stack_segmentation2(stack_image: SegmentationTuple, mask_segmentation_parameters):
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for i, (x, y) in enumerate(itertools.product([0, 20], repeat=2), start=1):
        data[3:-3, x + 4 : x + 16, y + 4 : y + 16] = i
    data = stack_image.image.fit_array_to_image(data)
    mask_segmentation_parameters.values["threshold"]["values"]["threshold"] = 110
    parameters = {i: deepcopy(mask_segmentation_parameters) for i in range(1, 5)}
    return dataclasses.replace(
        stack_image, segmentation=data, segmentation_parameters=parameters, selected_components=[1, 3]
    )


@pytest.fixture
def mask_property():
    return MaskProperty(RadiusType.NO, 0, RadiusType.NO, 0, False, False, False)
