import itertools
import os
from pathlib import Path

import pytest

from PartSegCore.analysis import ProjectTuple
from PartSegCore.analysis.load_functions import LoadImageMask, LoadMaskSegmentation, LoadProject, LoadStackImage
from PartSegCore.analysis.save_functions import save_dict as analysis_save_dict
from PartSegCore.mask.io_functions import LoadROIImage, LoadROIParameters
from PartSegCore.mask.io_functions import LoadStackImage as MaskLoadStackImage
from PartSegCore.mask.io_functions import (
    MaskProjectTuple,
    save_components_dict,
    save_parameters_dict,
    save_segmentation_dict,
)


class TestPartSettingsIO:
    @pytest.mark.parametrize("method", analysis_save_dict.values())
    def test_save(self, method, qtbot, part_settings_with_project, tmp_path):
        pi = part_settings_with_project.get_project_info()
        method.save(tmp_path / "data", pi, method.get_default_values())

    def test_load_segmentation(self, part_settings, data_test_dir):
        path = os.path.join(data_test_dir, "stack1_component1_1.tgz")
        pi = LoadProject.load([path])
        part_settings.set_project_info(pi)

    @pytest.mark.parametrize(
        "file_name",
        [
            "Image0003_01.oif",
            "test_czi.czi",
            "test_lsm.lsm",
            "test_lsm.tif",
            "test_lsm2.tif",
            "test_nucleus.tif",
            "test_nucleus_mask.tif",
        ],
    )
    def test_load_images(self, file_name, part_settings, data_test_dir):
        path = os.path.join(data_test_dir, file_name)
        pi = LoadStackImage.load([path])
        assert isinstance(pi, ProjectTuple)
        part_settings.set_project_info(pi)

    def test_load_images_with_mask(self, part_settings, data_test_dir):
        base_path = Path(os.path.join(data_test_dir, "stack1_components"))

        pi = LoadImageMask.load([base_path / "stack1_component1.tif", base_path / "stack1_component1_mask.tif"])
        assert isinstance(pi, ProjectTuple)
        part_settings.set_project_info(pi)

    def test_load_mask_project(self, part_settings, data_test_dir):
        path = os.path.join(data_test_dir, "test_nucleus.seg")
        pi_l = LoadMaskSegmentation.load([path])
        assert isinstance(pi_l, list)
        for pi in pi_l:
            part_settings.set_project_info(pi)


class TestStackSettingsIO:
    @pytest.mark.parametrize(
        "method",
        itertools.chain(save_segmentation_dict.values(), save_parameters_dict.values(), save_components_dict.values()),
    )
    def test_save(self, method, stack_segmentation1, stack_settings, tmp_path):
        stack_settings.set_project_info(stack_segmentation1)
        pi = stack_settings.get_project_info()
        method.save(tmp_path / "data", pi, method.get_default_values())

    @pytest.mark.parametrize(
        "file_name",
        [
            "Image0003_01.oif",
            "test_czi.czi",
            "test_lsm.lsm",
            "test_lsm.tif",
            "test_lsm2.tif",
            "test_nucleus.tif",
            "test_nucleus_mask.tif",
        ],
    )
    def test_load_images(self, file_name, stack_settings, data_test_dir):
        path = os.path.join(data_test_dir, file_name)
        pi = MaskLoadStackImage.load([path])
        assert isinstance(pi, MaskProjectTuple)
        stack_settings.set_project_info(pi)

    def test_load_segmentation_image(self, data_test_dir, stack_settings):
        pi = LoadROIImage.load([os.path.join(data_test_dir, "test_nucleus.seg")])
        assert isinstance(pi, MaskProjectTuple)
        stack_settings.set_project_info(pi)
        assert pi.image is not None

    def test_load_parameters(self, data_test_dir):
        pi = LoadROIParameters.load([os.path.join(data_test_dir, "test_nucleus.seg")])
        assert isinstance(pi, MaskProjectTuple)
        assert len(pi.roi_extraction_parameters) == 4
        assert pi.roi_extraction_parameters[1].algorithm == "Threshold"
