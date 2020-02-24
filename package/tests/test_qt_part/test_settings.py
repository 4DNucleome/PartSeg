import os
import typing

import pytest

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.segmentation_mask.stack_gui_main import ChosenComponents
from PartSeg.segmentation_mask.stack_settings import StackSettings
from PartSegCore.mask.io_functions import LoadSegmentationImage, LoadStackImage


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
