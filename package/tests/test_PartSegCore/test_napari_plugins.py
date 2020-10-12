import os

import pytest

from PartSegCore.napari_plugins.load_image import napari_get_reader as napari_get_reader_image
from PartSegCore.napari_plugins.load_mask_project import napari_get_reader as napari_get_reader_mask
from PartSegCore.napari_plugins.load_masked_image import napari_get_reader as napari_get_reader_mask_image
from PartSegCore.napari_plugins.load_roi_project import napari_get_reader as napari_get_reader_roi
from PartSegCore.napari_plugins.loader import project_to_layers


def test_project_to_layers_analysis(analysis_segmentation):
    res = project_to_layers(analysis_segmentation)
    assert len(res) == 2
    assert res[0][2] == "image"
    assert res[1][2] == "labels"


def test_project_to_layers_mask(stack_segmentation1):
    res = project_to_layers(stack_segmentation1)
    assert len(res) == 2
    assert res[0][2] == "image"


@pytest.fixture()
def load_data(data_test_dir):
    def _load_data(file_name, reader_hook):
        file_path = os.path.join(data_test_dir, file_name)
        res = reader_hook(file_path)
        assert res is not None
        return res(file_path)

    return _load_data


@pytest.mark.parametrize(
    "file_name",
    ["test_nucleus.tif", "test_lsm.lsm", "Image0003_01.oif", "test_czi.czi", "N2A_H2BGFP_dapi_falloidin_cycling1.oib"],
)
def test_read_images(load_data, file_name):
    data = load_data(file_name, napari_get_reader_image)
    assert isinstance(data, list)


def test_read_mask_project(load_data):
    data = load_data("test_nucleus.seg", napari_get_reader_mask)
    assert len(data) == 1
    assert data[0][2] == "labels"


def test_read_masked_image(load_data):
    data = load_data("test_nucleus.tif", napari_get_reader_mask_image)
    assert len(data) == 2
    assert data[0][2] == "image"
    assert data[1][2] == "labels"


def test_load_roi_project(load_data):
    data = load_data("stack1_component1.tgz", napari_get_reader_roi)
    assert len(data) == 4
    assert data[0][2] == "image"
    assert data[1][2] == "image"
    assert data[2][2] == "labels"
    assert data[3][2] == "labels"
