import os
import typing

import pytest
import numpy as np

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.segmentation_analysis.partseg_settings import PartSettings
from PartSeg.segmentation_mask.stack_gui_main import ChosenComponents
from PartSeg.segmentation_mask.stack_settings import StackSettings
from PartSegCore.analysis import analysis_algorithm_dict
from PartSegCore.analysis.io_utils import create_history_element_from_project
from PartSegCore.analysis.load_functions import LoadProject
from PartSegCore.analysis.save_functions import SaveProject
from PartSegCore.mask.io_functions import LoadSegmentationImage, LoadStackImage
from test_pipeline import image, mask_property, algorithm_parameters
from PartSegCore.mask_create import calculate_mask


class TestStackSettings:
    def test_add_project(self, tmp_path, qtbot, data_test_dir):
        settings = StackSettings(tmp_path)
        project1 = LoadSegmentationImage.load([os.path.join(data_test_dir, "test_nucleus.seg")])
        chosen_components = ChosenComponents()
        qtbot.addWidget(chosen_components)
        settings.chosen_components_widget = chosen_components
        settings.set_project_info(project1)
        project1_res = settings.get_project_info()
        assert isinstance(project1_res.chosen_components, typing.Iterable) and len(project1_res.chosen_components) == 2
        assert project1_res.file_path == os.path.join(data_test_dir, "test_nucleus.tif")
        project2 = LoadStackImage.load([os.path.join(data_test_dir, "test_lsm.tif")])
        settings.set_project_info(project2)
        project2_res = settings.get_project_info()
        assert project2_res.segmentation is None
        assert isinstance(project2_res.chosen_components, typing.Iterable) and len(project2_res.chosen_components) == 0
        assert project2_res.file_path == os.path.join(data_test_dir, "test_lsm.tif")


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
        settings.add_history_element(create_history_element_from_project(project_info, mask_property, ))
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
