import dataclasses
import os
import typing

import numpy as np
import pytest

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg._roi_mask.main_window import ChosenComponents
from PartSeg._roi_mask.stack_settings import StackSettings
from PartSeg.common_backend.base_settings import BaseSettings
from PartSegCore.analysis import analysis_algorithm_dict
from PartSegCore.analysis.io_utils import create_history_element_from_project
from PartSegCore.analysis.load_functions import LoadProject
from PartSegCore.analysis.save_functions import SaveProject
from PartSegCore.io_utils import HistoryProblem
from PartSegCore.mask.history_utils import create_history_element_from_segmentation_tuple
from PartSegCore.mask.io_functions import LoadStackImage
from PartSegCore.mask_create import calculate_mask_from_project


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
        assert project2_res.roi is None
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
        assert np.all(loaded.roi == result2.roi)
        assert len(loaded.history) == 1
