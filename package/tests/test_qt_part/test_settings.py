import itertools
import os
import typing
from copy import deepcopy

import numpy as np
import pytest
import SimpleITK as sitk


from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.segmentation_analysis.partseg_settings import PartSettings
from PartSeg.segmentation_mask.stack_gui_main import ChosenComponents
from PartSeg.segmentation_mask.stack_settings import StackSettings
from PartSegCore.analysis import analysis_algorithm_dict
from PartSegCore.analysis.io_utils import create_history_element_from_project
from PartSegCore.analysis.load_functions import LoadProject
from PartSegCore.analysis.save_functions import SaveProject
from PartSegCore.image_operations import RadiusType
from PartSegCore.io_utils import HistoryElement, HistoryProblem
from PartSegCore.mask.history_utils import create_history_element_from_segmentation_tuple
from PartSegCore.mask.io_functions import LoadSegmentationImage, LoadStackImage, SegmentationTuple
from PartSegCore.mask_create import calculate_mask, MaskProperty
from PartSegImage import Image


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
    return {
        "algorithm": "Threshold",
        "values": {
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
    }


@pytest.fixture
def stack_segmentation1(stack_image: SegmentationTuple, mask_segmentation_parameters):
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for i, (x, y) in enumerate(itertools.product([0, 20], repeat=2), start=1):
        data[1:-1, x + 2 : x + 18, y + 2 : y + 18] = i
    parameters = {i: deepcopy(mask_segmentation_parameters) for i in range(1, 5)}
    return stack_image._replace(segmentation=data, segmentation_parameters=parameters, selected_components=[1, 3])


@pytest.fixture
def stack_segmentation2(stack_image: SegmentationTuple, mask_segmentation_parameters):
    data = np.zeros([20, 40, 40], dtype=np.uint8)
    for i, (x, y) in enumerate(itertools.product([0, 20], repeat=2), start=1):
        data[3:-3, x + 4 : x + 16, y + 4 : y + 16] = i
    mask_segmentation_parameters["values"]["threshold"]["values"]["threshold"] = 110
    parameters = {i: deepcopy(mask_segmentation_parameters) for i in range(1, 5)}
    return stack_image._replace(segmentation=data, segmentation_parameters=parameters, selected_components=[1, 3])


@pytest.fixture
def mask_property():
    return MaskProperty(RadiusType.NO, 0, RadiusType.NO, 0, False, False, False)


@pytest.fixture(scope="module")
def nucleus_project(data_test_dir):
    # TODO change on synthetic example
    return LoadStackImage.load([os.path.join(data_test_dir, "test_nucleus.tif")])


@pytest.fixture
def segmentation_fun(nucleus_project, mask_segmentation_parameters):
    channel = nucleus_project.image.get_channel(0)[0]
    threshold_img = (channel > 8000).astype(np.uint8)
    components = sitk.GetArrayFromImage(
        sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_img)), 200)
    )

    def calculate(threshold=None):
        if threshold is not None:
            if threshold < 8000:
                raise ValueError("Not supported")

            threshold_img = channel > threshold
            comp = np.copy(components)
            comp[threshold_img == 0] = 0
        else:
            comp = components

        return nucleus_project.replace_(
            segmentation=comp, segmentation_parameters={i: deepcopy(mask_segmentation_parameters) for i in range(1, 6)}
        )

    return calculate


@pytest.fixture
def stack_settings(qtbot, tmp_path):
    settings = StackSettings(tmp_path)
    chose = ChosenComponents()
    qtbot.addWidget(chose)
    settings.chosen_components_widget = chose
    return settings


class TestStackSettings:
    def test_add_project(self, stack_settings, stack_segmentation1, data_test_dir):
        stack_settings.set_project_info(stack_segmentation1)
        project1_res = stack_settings.get_project_info()
        assert (
            isinstance(project1_res.selected_components, typing.Iterable) and len(project1_res.selected_components) == 2
        )
        assert project1_res.file_path == "test_path"
        project2 = LoadStackImage.load([os.path.join(data_test_dir, "test_lsm.tif")])
        stack_settings.set_project_info(project2)
        project2_res = stack_settings.get_project_info()
        assert project2_res.segmentation is None
        assert (
            isinstance(project2_res.selected_components, typing.Iterable) and len(project2_res.selected_components) == 0
        )
        assert project2_res.file_path == os.path.join(data_test_dir, "test_lsm.tif")

    def test_set_project(self, stack_settings, stack_segmentation1, stack_segmentation2):
        stack_settings.set_project_info(stack_segmentation1)
        segmentation1 = stack_settings.segmentation
        stack_settings.set_project_info(stack_segmentation2)
        segmentation2 = stack_settings.segmentation
        assert np.any(segmentation1 != segmentation2)

    def test_set_project_with_components(self, stack_settings, stack_segmentation1):
        seg1 = stack_segmentation1._replace(selected_components=[1, 2])
        seg2 = stack_segmentation1._replace(selected_components=[3])
        stack_settings.set_project_info(seg1)
        assert stack_settings.chosen_components() == [1, 2]
        stack_settings.set_project_info(seg2)
        assert stack_settings.chosen_components() == [3]
        stack_settings.keep_chosen_components = True
        stack_settings.set_project_info(seg1)
        assert stack_settings.chosen_components() == [1, 2, 3]

    def test_set_project_with_history(self, stack_settings, stack_segmentation1, mask_property):
        seg2 = stack_segmentation1._replace(
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[1],
        )
        seg3 = stack_segmentation1._replace(
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[2],
        )
        stack_settings.set_project_info(seg2)
        assert stack_settings.chosen_components() == [1]
        stack_settings.set_project_info(seg3)
        assert stack_settings.chosen_components() == [2]
        stack_settings.keep_chosen_components = True
        stack_settings.set_project_info(seg2)
        assert stack_settings.chosen_components() == [1, 2]

    def test_set_project_with_history_components_fail(self, stack_settings, stack_segmentation1, mask_property):
        seg2 = stack_segmentation1._replace(
            history=[
                create_history_element_from_segmentation_tuple(
                    stack_segmentation1._replace(selected_components=[1]), mask_property
                )
            ],
            selected_components=[1],
        )
        seg3 = stack_segmentation1._replace(
            history=[
                create_history_element_from_segmentation_tuple(
                    stack_segmentation1._replace(selected_components=[1, 2]), mask_property
                )
            ],
            selected_components=[2],
        )
        stack_settings.set_project_info(seg2)
        assert stack_settings.chosen_components() == [1]
        stack_settings.set_project_info(seg3)
        assert stack_settings.chosen_components() == [2]
        stack_settings.keep_chosen_components = True
        with pytest.raises(HistoryProblem):
            stack_settings.set_project_info(seg2)

    def test_set_project_with_history_parameters_fail(
        self, stack_settings, stack_segmentation1, stack_segmentation2, mask_property
    ):
        seg2 = stack_segmentation1._replace(
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[1],
        )
        seg3 = stack_segmentation1._replace(
            history=[create_history_element_from_segmentation_tuple(stack_segmentation2, mask_property)],
            selected_components=[2],
        )
        stack_settings.set_project_info(seg2)
        assert stack_settings.chosen_components() == [1]
        stack_settings.set_project_info(seg3)
        assert stack_settings.chosen_components() == [2]
        stack_settings.keep_chosen_components = True
        with pytest.raises(HistoryProblem):
            stack_settings.set_project_info(seg2)

    def test_set_project_with_history_length_fail(
        self, stack_settings, stack_segmentation1, mask_property
    ):
        seg2 = stack_segmentation1._replace(
            history=[
                create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property),
                create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)
            ],
            selected_components=[1],
        )
        seg3 = stack_segmentation1._replace(
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[2],
        )
        stack_settings.set_project_info(seg2)
        assert stack_settings.chosen_components() == [1]
        stack_settings.set_project_info(seg3)
        assert stack_settings.chosen_components() == [2]
        stack_settings.keep_chosen_components = True
        with pytest.raises(HistoryProblem):
            stack_settings.set_project_info(seg2)


class TestBaseSettings:
    def test_empty_history(self, tmp_path):
        settings = BaseSettings(tmp_path)
        assert settings.history_size() == 0
        assert settings.history_redo_size() == 0

        with pytest.raises(IndexError):
            settings.history_current_element()

        with pytest.raises(IndexError):
            settings.history_next_element()

        settings.history_pop()
        assert settings.history_index == -1

    def test_modifying_history(self, tmp_path):
        settings = BaseSettings(tmp_path)
        for i in range(10):
            settings.add_history_element(i)

        assert settings.history_size() == 10
        assert settings.history_redo_size() == 0

        with pytest.raises(IndexError):
            settings.history_next_element()

        for i in range(9, 5, -1):
            assert settings.history_current_element() == i
            assert settings.history_pop() == i

        assert settings.history_current_element() == 5
        assert settings.history_size() == 6
        assert settings.history_redo_size() == 4
        settings.add_history_element(7)
        assert settings.history_size() == 7
        assert settings.history_redo_size() == 0

    def test_clean_redo(self, tmp_path):
        settings = BaseSettings(tmp_path)
        for i in range(10):
            settings.add_history_element(i)
        for _ in range(5):
            settings.history_pop()
        settings.history_current_element()
        settings.history_redo_clean()
        settings.history_current_element()


class TestPartSettings:
    def test_get_project_info(self, qtbot, tmp_path, image):
        settings = PartSettings(tmp_path)
        settings.last_executed_algorithm = "aa"
        settings.set("algorithms.aa", 1)
        settings.image = image
        pt = settings.get_project_info()
        assert pt.algorithm_parameters["algorithm_name"] == "aa"
        assert pt.algorithm_parameters["values"] == 1

    def test_pipeline_saving(self, qtbot, tmp_path, image, algorithm_parameters, mask_property):
        settings = PartSettings(tmp_path)
        settings.image = image
        settings.last_executed_algorithm = algorithm_parameters["algorithm_name"]
        algorithm = analysis_algorithm_dict[algorithm_parameters["algorithm_name"]]()
        algorithm.set_image(settings.image)
        algorithm.set_parameters(**algorithm_parameters["values"])
        result = algorithm.calculation_run(lambda x, y: None)
        settings.segmentation = result.segmentation
        settings.full_segmentation = result.full_segmentation
        settings.noise_remove_image_part = result.cleaned_channel
        settings.last_executed_algorithm = result.parameters.algorithm
        settings.set(f"algorithms.{result.parameters.algorithm}", result.parameters.values)
        project_info = settings.get_project_info()
        mask = calculate_mask(mask_property, settings.segmentation, settings.mask, settings.image_spacing)
        settings.add_history_element(create_history_element_from_project(project_info, mask_property,))
        settings.mask = mask
        algorithm_parameters["values"]["channel"] = 1
        algorithm.set_parameters(**algorithm_parameters["values"])
        algorithm.set_mask(settings.mask)
        result2 = algorithm.calculation_run(lambda x, y: None)
        assert np.max(result2.segmentation) == 2
        settings.segmentation = result2.segmentation
        settings.full_segmentation = result2.full_segmentation
        settings.noise_remove_image_part = result2.cleaned_channel
        settings.last_executed_algorithm = result.parameters.algorithm
        settings.set(f"algorithms.{result.parameters.algorithm}", result.parameters.values)
        project_info = settings.get_project_info()
        SaveProject.save(tmp_path / "project.tgz", project_info)
        assert os.path.exists(tmp_path / "project.tgz")
        loaded = LoadProject.load([tmp_path / "project.tgz"])
        assert np.all(loaded.segmentation == result2.segmentation)
        assert len(loaded.history) == 1
