import os

import typing

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
