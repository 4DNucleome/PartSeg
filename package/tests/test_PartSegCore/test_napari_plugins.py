# pylint: disable=no-self-use

import os

import numpy as np
import pytest
from napari.layers import Image, Labels, Layer

from PartSegCore.analysis import ProjectTuple
from PartSegCore.mask.io_functions import LoadROIFromTIFF
from PartSegCore.napari_plugins.load_image import napari_get_reader as napari_get_reader_image
from PartSegCore.napari_plugins.load_mask_project import napari_get_reader as napari_get_reader_mask
from PartSegCore.napari_plugins.load_masked_image import napari_get_reader as napari_get_reader_mask_image
from PartSegCore.napari_plugins.load_roi_project import napari_get_reader as napari_get_reader_roi
from PartSegCore.napari_plugins.loader import project_to_layers
from PartSegCore.napari_plugins.save_mask_roi import napari_write_labels
from PartSegCore.napari_plugins.save_tiff_layer import (
    napari_write_images,
)
from PartSegCore.napari_plugins.save_tiff_layer import (
    napari_write_labels as napari_write_labels_tiff,
)
from PartSegImage import GenericImageReader
from PartSegImage import Image as PImage


def test_project_to_layers_analysis(analysis_segmentation):
    analysis_segmentation.roi_info.alternative["test"] = np.zeros(analysis_segmentation.image.shape, dtype=np.uint8)
    res = project_to_layers(analysis_segmentation)
    assert len(res) == 4
    l1 = Layer.create(*res[0])
    assert isinstance(l1, Image)
    assert l1.name == "channel 1"
    assert np.allclose(l1.scale[1:] / 1e9, analysis_segmentation.image.spacing)
    l2 = Layer.create(*res[2])
    assert isinstance(l2, Labels)
    assert l2.name == "ROI"
    assert np.allclose(l2.scale[1:] / 1e9, analysis_segmentation.image.spacing)
    l3 = Layer.create(*res[3])
    assert isinstance(l3, Labels)
    assert l3.name == "test"
    assert np.allclose(l3.scale[1:] / 1e9, analysis_segmentation.image.spacing)
    assert not l3.visible


def test_project_to_layers_roi():
    data = np.zeros((1, 1, 10, 10, 10), dtype=np.uint8)
    img = PImage(data, image_spacing=(1, 1, 1), name="ROI", axes_order="CTZYX")
    proj = ProjectTuple(file_path="", image=img)
    res = project_to_layers(proj)
    assert len(res) == 1
    assert isinstance(res[0][0], np.ndarray)
    assert res[0][2] == "labels"


def test_project_to_layers_mask(stack_segmentation1):
    res = project_to_layers(stack_segmentation1)
    assert len(res) == 3
    assert res[0][2] == "image"


@pytest.fixture
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


def test_write_labels(tmp_path):
    data = np.zeros((1, 10, 10, 10), dtype=np.uint8)
    assert napari_write_labels(str(tmp_path / "test.seg"), [], {"scale": [10, 10]}) is None
    assert not (tmp_path / "test.seg").exists()

    assert napari_write_labels(str(tmp_path / "test.seg"), data, {"scale": [10, 10]}) == str(tmp_path / "test.seg")
    assert (tmp_path / "test.seg").exists()

    assert napari_write_labels(str(tmp_path / "test.txt"), data, {"scale": [10, 10]}) is None
    assert not (tmp_path / "test.txt").exists()


def test_save_load_axis_order(tmp_path):
    data = np.zeros((1, 10, 20, 30), dtype=np.uint8)
    layer = Labels(data)
    data_path = str(tmp_path / "test.tif")
    assert napari_write_labels_tiff(data_path, *layer.as_layer_data_tuple()[:2])
    proj = LoadROIFromTIFF.load([data_path])
    assert proj.roi_info.roi.shape == data.shape
    assert napari_write_labels_tiff(str(tmp_path / "test.seg"), *layer.as_layer_data_tuple()[:2]) is None


@pytest.fixture(params=[(1, 4), (1, 3), (3, 4), (3, 3), (10, 4)])
def image_layer_tuples(request):
    res = []
    for i in range(request.param[0]):
        data = np.zeros((1, 10, 20, 30)[-request.param[1] :], dtype=np.uint8)
        data[:, 1:-1, 1:-1] = i + 1
        res.append(Image(data, scale=(1, 1, 1, 1)[-request.param[1] :]).as_layer_data_tuple())
    return res


def test_napari_write_images(image_layer_tuples, tmp_path):
    data_path = str(tmp_path / "test.tif")
    assert len(napari_write_images(data_path, image_layer_tuples)) == 1
    image = GenericImageReader.read_image(data_path)
    assert image.channels == len(image_layer_tuples)


def test_write_multichannel(tmp_path):
    data = np.zeros((1, 10, 20, 30, 5), dtype=np.uint8)
    data[:, 1:-1, 1:-1, 1:-1] = 1
    layer = Image(data)
    data_path = str(tmp_path / "test.tif")
    assert napari_write_images(data_path, [layer.as_layer_data_tuple()])
    image = GenericImageReader.read_image(data_path)
    assert image.channels == 5


def test_different_shapes(tmp_path):
    layer1 = Image(np.eye(10, dtype=np.uint8))
    layer2 = Image(np.eye(20, dtype=np.uint8))
    data_path = str(tmp_path / "test.tif")
    assert not napari_write_images(data_path, [layer1.as_layer_data_tuple(), layer2.as_layer_data_tuple()])
    assert not (tmp_path / "test.tif").exists()
