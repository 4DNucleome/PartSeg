import dataclasses
import itertools
import multiprocessing as mp
import os
import signal
from copy import deepcopy
from pathlib import Path
from queue import Empty

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import ProjectTuple, SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.analysis.measurement_base import AreaType, MeasurementEntry, PerComponent
from PartSegCore.analysis.measurement_calculation import ComponentsNumber, MeasurementProfile, Volume
from PartSegCore.image_operations import RadiusType
from PartSegCore.mask.io_functions import MaskProjectTuple
from PartSegCore.mask_create import MaskProperty
from PartSegCore.project_info import HistoryElement
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.restartable_segmentation_algorithms import BorderRim, LowerThresholdAlgorithm
from PartSegCore.segmentation.segmentation_algorithm import ThresholdAlgorithm
from PartSegImage import Image


@pytest.fixture(scope="module")
def data_test_dir():
    """Return path to directory with test data"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "test_data")


@pytest.fixture(scope="module")
def bundle_test_dir():
    """Return path to directory with test data"""
    return Path(os.path.join(os.path.dirname(__file__), "test_data"))


def wait_sigint(q: mp.Queue, pid):
    try:
        q.get(timeout=20 * 60)
    except Empty:
        print("Timeout")
        os.kill(pid, signal.SIGINT)
        import time

        time.sleep(30)
        try:
            os.kill(pid, 0)
        except OSError:
            pass
        else:
            os.kill(pid, signal.SIGKILL)


@pytest.fixture(scope="session", autouse=True)
def sigint_after_time():
    manager = mp.Manager()
    q = manager.Queue()
    p = mp.Process(target=wait_sigint, args=(q, os.getpid()))
    p.start()
    yield
    q.put(1)
    p.join()


@pytest.fixture
def image(tmp_path):
    data = np.zeros([20, 20, 20, 2], dtype=np.uint8)
    data[10:-1, 1:-1, 1:-1, 0] = 20
    data[1:10, 1:-1, 1:-1, 1] = 20
    data[1:-1, 1:5, 1:-1, 1] = 20
    data[1:-1, -5:-1, 1:-1, 1] = 20

    return Image(data, (10**-3, 10**-3, 10**-3), axes_order="ZYXC", file_path=str(tmp_path / "test.tiff"))


@pytest.fixture
def image2(image, tmp_path):
    data = np.zeros([20, 20, 20, 1], dtype=np.uint8)
    data[10:-1, 1:-1, 1:-1, 0] = 20
    img = image.merge(Image(data, (10**-3, 10**-3, 10**-3), axes_order="ZYXC"), "C")
    img.file_path = str(tmp_path / "test2.tiff")
    return img


@pytest.fixture
def image2d(tmp_path):
    data = np.zeros([20, 20], dtype=np.uint8)
    data[10:-1, 1:-1] = 20
    return Image(data, (10**-3, 10**-3), axes_order="YX", file_path=str(tmp_path / "test.tiff"))


@pytest.fixture
def stack_image():
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for x, y in itertools.product([0, 20], repeat=2):
        data[1:-1, x + 2 : x + 18, y + 2 : y + 18] = 100
    for x, y in itertools.product([0, 20], repeat=2):
        data[3:-3, x + 4 : x + 16, y + 4 : y + 16] = 120
    for x, y in itertools.product([0, 20], repeat=2):
        data[5:-5, x + 6 : x + 14, y + 6 : y + 14] = 140

    return MaskProjectTuple("test_path", Image(data, (2, 1, 1), axes_order="ZYX", file_path="test_path"))


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
def roi_extraction_profile():
    return ROIExtractionProfile(
        name="test",
        algorithm=LowerThresholdAlgorithm.get_name(),
        values=LowerThresholdAlgorithm.get_default_values(),
    )


@pytest.fixture
def mask_segmentation_parameters():
    return ROIExtractionProfile(
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
def stack_segmentation1(stack_image: MaskProjectTuple, mask_segmentation_parameters):
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for i, (x, y) in enumerate(itertools.product([0, 20], repeat=2), start=1):
        data[1:-1, x + 2 : x + 18, y + 2 : y + 18] = i
    data = ROIInfo(stack_image.image.fit_array_to_image(data))
    parameters = {i: deepcopy(mask_segmentation_parameters) for i in range(1, 5)}
    return dataclasses.replace(
        stack_image, roi_info=data, roi_extraction_parameters=parameters, selected_components=[1, 3]
    )


@pytest.fixture
def analysis_segmentation(stack_image: MaskProjectTuple):
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for i, (x, y) in enumerate(itertools.product([0, 20], repeat=2), start=1):
        data[1:-1, x + 2 : x + 18, y + 2 : y + 18] = i
    data = ROIInfo(stack_image.image.fit_array_to_image(data))
    return ProjectTuple(file_path=stack_image.file_path, image=stack_image.image, roi_info=data)


@pytest.fixture
def analysis_segmentation2(analysis_segmentation: ProjectTuple):
    mask = (analysis_segmentation.roi_info.roi > 0).astype(np.uint8)
    return dataclasses.replace(analysis_segmentation, mask=mask)


@pytest.fixture
def stack_segmentation2(stack_image: MaskProjectTuple, mask_segmentation_parameters):
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for i, (x, y) in enumerate(itertools.product([0, 20], repeat=2), start=1):
        data[3:-3, x + 4 : x + 16, y + 4 : y + 16] = i
    data = ROIInfo(stack_image.image.fit_array_to_image(data))
    mask_segmentation_parameters.values.threshold.values.threshold = 110
    parameters = {i: deepcopy(mask_segmentation_parameters) for i in range(1, 5)}
    return dataclasses.replace(
        stack_image, roi_info=data, roi_extraction_parameters=parameters, selected_components=[1, 3]
    )


@pytest.fixture
def mask_property():
    return MaskProperty.simple_mask()


@pytest.fixture
def mask_property_non_default():
    return MaskProperty(
        dilate=RadiusType.R2D,
        dilate_radius=10,
        fill_holes=RadiusType.R3D,
        max_holes_size=10,
        save_components=True,
        clip_to_mask=True,
        reversed_mask=True,
    )


@pytest.fixture
def measurement_profiles():
    statistics = [
        MeasurementEntry(
            name="Segmentation Volume",
            calculation_tree=Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
        ),
        MeasurementEntry(
            name="ROI Components Number",
            calculation_tree=ComponentsNumber.get_starting_leaf().replace_(
                area=AreaType.ROI, per_component=PerComponent.No
            ),
        ),
    ]
    statistics2 = [
        MeasurementEntry(
            name="Mask Volume",
            calculation_tree=Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No),
        ),
    ]
    return MeasurementProfile(name="statistic1", chosen_fields=statistics), MeasurementProfile(
        name="statistic2", chosen_fields=statistics + statistics2
    )


@pytest.fixture
def border_rim_profile():
    return ROIExtractionProfile(
        name="border_profile", algorithm=BorderRim.get_name(), values=BorderRim.get_default_values()
    )


@pytest.fixture
def lower_threshold_profile():
    return ROIExtractionProfile(
        name="lower_profile",
        algorithm=LowerThresholdAlgorithm.get_name(),
        values=LowerThresholdAlgorithm.get_default_values(),
    )


@pytest.fixture
def mask_threshold_profile():
    return ROIExtractionProfile(
        name="mask_profile",
        algorithm=ThresholdAlgorithm.get_name(),
        values=ThresholdAlgorithm.get_default_values(),
    )


@pytest.fixture
def sample_pipeline(border_rim_profile, lower_threshold_profile, mask_property):
    return SegmentationPipeline(
        name="sample_pipeline",
        segmentation=border_rim_profile,
        mask_history=[SegmentationPipelineElement(segmentation=lower_threshold_profile, mask_property=mask_property)],
    )


@pytest.fixture
def sample_pipeline2(border_rim_profile, lower_threshold_profile, mask_property):
    return SegmentationPipeline(
        name="sample_pipeline2",
        segmentation=lower_threshold_profile,
        mask_history=[SegmentationPipelineElement(segmentation=border_rim_profile, mask_property=mask_property)],
    )


@pytest.fixture
def history_element(image, lower_threshold_profile):
    roi = np.zeros(image.shape, dtype=np.uint8)
    roi[0, 2:10] = 1
    roi[0, 10:-2] = 2

    return HistoryElement.create(
        roi_info=ROIInfo(roi),
        mask=None,
        roi_extraction_parameters={
            "algorithm_name": lower_threshold_profile.name,
            "values": lower_threshold_profile.values,
        },
        mask_property=MaskProperty.simple_mask(),
    )


def pytest_collection_modifyitems(session, config, items):
    image_tests = [x for x in items if "PartSegImage" in str(x.fspath)]
    core_tests = [x for x in items if "PartSegCore" in str(x.fspath)]
    other_test = [x for x in items if "PartSegCore" not in str(x.fspath) and "PartSegImage" not in str(x.fspath)]
    items[:] = image_tests + core_tests + other_test
