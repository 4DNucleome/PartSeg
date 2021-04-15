import dataclasses
import os
import typing

import numpy as np
import pytest

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg._roi_mask.main_window import ChosenComponents
from PartSeg._roi_mask.stack_settings import StackSettings, get_mask
from PartSeg.common_backend.base_settings import BaseSettings, SwapTimeStackException, TimeAndStackException
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import analysis_algorithm_dict
from PartSegCore.analysis.io_utils import MaskInfo, create_history_element_from_project
from PartSegCore.analysis.load_functions import LoadProject
from PartSegCore.analysis.save_functions import SaveProject
from PartSegCore.io_utils import PointsInfo
from PartSegCore.mask.history_utils import create_history_element_from_segmentation_tuple
from PartSegCore.mask.io_functions import LoadStackImage
from PartSegCore.project_info import HistoryProblem, calculate_mask_from_project
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.algorithm_base import SegmentationResult
from PartSegImage import Image


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
        assert project2_res.roi_info.roi is None
        assert (
            isinstance(project2_res.selected_components, typing.Iterable) and len(project2_res.selected_components) == 0
        )
        assert project2_res.file_path == os.path.join(data_test_dir, "test_lsm.tif")

    def test_set_project(self, stack_settings, stack_segmentation1, stack_segmentation2):
        stack_settings.set_project_info(stack_segmentation1)
        segmentation1 = stack_settings.roi
        stack_settings.set_project_info(stack_segmentation2)
        segmentation2 = stack_settings.roi
        assert np.any(segmentation1 != segmentation2)

    def test_set_project_with_components(self, stack_settings, stack_segmentation1):
        seg1 = dataclasses.replace(stack_segmentation1, selected_components=[1, 2])
        seg2 = dataclasses.replace(stack_segmentation1, selected_components=[3])
        stack_settings.set_project_info(seg1)
        assert stack_settings.chosen_components() == [1, 2]
        stack_settings.set_project_info(seg2)
        assert stack_settings.chosen_components() == [3]
        stack_settings.keep_chosen_components = True
        stack_settings.set_project_info(seg1)
        assert stack_settings.chosen_components() == [1, 2, 3]

    def test_set_project_with_history(self, stack_settings, stack_segmentation1, mask_property):
        seg2 = dataclasses.replace(
            stack_segmentation1,
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[1],
        )
        seg3 = dataclasses.replace(
            stack_segmentation1,
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
        seg2 = dataclasses.replace(
            stack_segmentation1,
            history=[
                create_history_element_from_segmentation_tuple(
                    dataclasses.replace(stack_segmentation1, selected_components=[1]), mask_property
                )
            ],
            selected_components=[1],
        )
        seg3 = dataclasses.replace(
            stack_segmentation1,
            history=[
                create_history_element_from_segmentation_tuple(
                    dataclasses.replace(stack_segmentation1, selected_components=[1, 2]), mask_property
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
        seg2 = dataclasses.replace(
            stack_segmentation1,
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[1],
        )
        seg3 = dataclasses.replace(
            stack_segmentation1,
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

    def test_set_project_with_history_length_fail(self, stack_settings, stack_segmentation1, mask_property):
        seg2 = dataclasses.replace(
            stack_segmentation1,
            history=[
                create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property),
                create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property),
            ],
            selected_components=[1],
        )
        seg3 = dataclasses.replace(
            stack_segmentation1,
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

    def test_set_segmentation_result(self, stack_settings, stack_segmentation1, stack_image):
        stack_settings.set_project_info(stack_image)
        seg = SegmentationResult(
            roi=stack_segmentation1.roi_info.roi, parameters=ROIExtractionProfile("test", "test2", {})
        )
        stack_settings.set_segmentation_result(seg)
        assert stack_settings.last_executed_algorithm == "test2"
        assert np.array_equal(stack_settings.roi, stack_segmentation1.roi_info.roi)

    def test_selected_components(self, stack_settings, stack_segmentation1):
        stack_settings.set_project_info(stack_segmentation1)
        assert stack_settings.chosen_components() == [1, 3]
        assert stack_settings.component_is_chosen(1)
        assert not stack_settings.component_is_chosen(2)
        assert np.array_equal(stack_settings.components_mask(), [0, 1, 0, 1, 0])
        stack_settings.chosen_components_widget.un_check_all()
        assert stack_settings.chosen_components() == []
        stack_settings.chosen_components_widget.check_all()
        assert stack_settings.chosen_components() == list(range(1, stack_segmentation1.roi_info.roi.max() + 1))


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

    def test_add_point(self, tmp_path, qtbot):
        settings = BaseSettings(tmp_path)
        with qtbot.waitSignal(settings.points_changed):
            settings.points = [1, 2, 3]

    def test_set_roi(self, tmp_path, qtbot):
        settings = BaseSettings(tmp_path)
        roi = np.zeros((10, 10), dtype=np.uint8)
        settings.image = Image(roi, (1, 1), axes_order="XY")
        roi[1:5, 1:5] = 1
        roi[5:-1, 5:-1] = 3
        with qtbot.waitSignal(settings.roi_changed):
            settings.roi = roi
        assert len(settings.roi_info.bound_info) == 2
        assert set(settings.roi_info.bound_info) == {1, 3}
        assert settings.roi_info.alternative == {}
        assert settings.roi_info.annotations == {}

        with qtbot.waitSignal(settings.roi_clean):
            settings.roi = None
        assert settings.roi is None

        settings.image = None
        assert settings.image is not None

    def test_channels(self, tmp_path, qtbot):
        settings = BaseSettings(tmp_path)
        assert not settings.has_channels
        assert settings.channels == 0
        settings.image = Image(np.zeros((10, 10, 2), dtype=np.uint8), (1, 1), axes_order="XYC")
        assert settings.has_channels
        assert settings.channels == 2
        settings.image = Image(np.zeros((10, 10, 1), dtype=np.uint8), (1, 1), axes_order="XYC")
        assert not settings.has_channels
        assert settings.channels == 1

    def test_shape(self, tmp_path):
        settings = BaseSettings(tmp_path)
        assert settings.image_shape == ()
        settings.image = Image(np.zeros((10, 10, 2), dtype=np.uint8), (1, 1), axes_order="XYC")
        assert settings.image_shape == (1, 1, 10, 10, 2)

    def test_verify_image(self):
        assert BaseSettings.verify_image(Image(np.zeros((10, 10, 2), dtype=np.uint8), (1, 1), axes_order="XYC"))
        with pytest.raises(SwapTimeStackException):
            BaseSettings.verify_image(
                Image(np.zeros((2, 10, 10), dtype=np.uint8), (1, 1, 1), axes_order="TXY"), silent=False
            )
        im = BaseSettings.verify_image(Image(np.zeros((2, 10, 10), dtype=np.uint8), (1, 1, 1), axes_order="TXY"))
        assert not im.is_time
        assert im.times == 1
        assert im.is_stack
        assert im.layers == 2
        with pytest.raises(TimeAndStackException):
            BaseSettings.verify_image(Image(np.zeros((2, 2, 10, 10), dtype=np.uint8), (1, 1, 1), axes_order="TZXY"))


class TestPartSettings:
    def test_set_mask_info(self, qtbot, tmp_path, image):
        settings = PartSettings(tmp_path)
        settings.image = image
        mask_info = MaskInfo("", (image.get_channel(0) > 2).astype(np.uint8))
        with qtbot.wait_signal(settings.mask_changed):
            settings.set_project_info(mask_info)

    def test_project_info_set(self, qtbot, analysis_segmentation, analysis_segmentation2, tmp_path, image):
        settings = PartSettings(tmp_path)
        settings.image = image
        settings.set_project_info(analysis_segmentation)
        assert settings.mask is None
        settings.set_project_info(analysis_segmentation2)
        assert settings.mask is not None
        settings.set_project_info(analysis_segmentation)
        assert settings.mask is None
        analysis_segmentation3 = dataclasses.replace(analysis_segmentation2, roi_info=ROIInfo(None))
        settings.set_project_info(analysis_segmentation3)
        assert settings.mask is not None
        assert settings.roi is None

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
        settings.set_segmentation_result(result)
        project_info = settings.get_project_info()
        mask = calculate_mask_from_project(mask_property, settings.get_project_info())
        settings.add_history_element(
            create_history_element_from_project(
                project_info,
                mask_property,
            )
        )
        settings.mask = mask
        calculate_mask_from_project(mask_property, settings.get_project_info())
        algorithm_parameters["values"]["channel"] = 1
        algorithm.set_parameters(**algorithm_parameters["values"])
        algorithm.set_mask(settings.mask)
        result2 = algorithm.calculation_run(lambda x, y: None)
        assert np.max(result2.roi) == 2
        settings.set_segmentation_result(result2)
        project_info = settings.get_project_info()
        SaveProject.save(tmp_path / "project.tgz", project_info)
        assert os.path.exists(tmp_path / "project.tgz")
        loaded = LoadProject.load([tmp_path / "project.tgz"])
        assert np.all(loaded.roi_info.roi == result2.roi)
        assert len(loaded.history) == 1


@pytest.mark.parametrize("Settings", [PartSettings, StackSettings])
def test_set_project_info(Settings, qtbot, tmp_path, image):
    settings = Settings(tmp_path)
    settings.points = [1, 2, 3]
    assert settings.points == [1, 2, 3]
    settings.image = image
    assert settings.points is None
    settings.set_project_info(PointsInfo("a", np.arange(3)))
    assert np.all(settings.points == [0, 1, 2])


def test_get_mask(stack_segmentation1):
    res1 = get_mask(stack_segmentation1.roi_info.roi, mask=None, selected=[1, 3])
    assert isinstance(res1, np.ndarray)
    assert set(np.unique(res1)) == {0, 1}
    assert set(stack_segmentation1.roi_info.roi[res1 > 0]) == {0, 2, 4}
    mask = np.ones(stack_segmentation1.roi_info.roi.shape, dtype=np.uint8)
    res2 = get_mask(stack_segmentation1.roi_info.roi, mask=mask, selected=[1, 3])
    assert set(np.unique(res2)) == {0, 1}
    assert np.array_equal(res1, res2)
    res3 = get_mask(stack_segmentation1.roi_info.roi, mask=res1, selected=[1, 2])
    assert set(stack_segmentation1.roi_info.roi[res3 > 0]) == {0, 4}
    assert set(np.unique(res3)) == {0, 1}
    res4 = get_mask(stack_segmentation1.roi_info.roi, mask=None, selected=[1])
    assert np.array_equal(res4, stack_segmentation1.roi_info.roi != 1)

    assert get_mask(stack_segmentation1.roi_info.roi, mask=None, selected=[]) is None
    assert get_mask(None, mask=None, selected=[1, 2]) is None
    assert np.array_equal(get_mask(None, mask=res4, selected=[1, 2]), res4)
    assert np.array_equal(get_mask(stack_segmentation1.roi_info.roi, mask=res4, selected=[]), res4)
