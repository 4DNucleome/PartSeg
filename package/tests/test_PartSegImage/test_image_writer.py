import numpy as np
import tifffile

from PartSegImage.image import Image
from PartSegImage.image_reader import TiffImageReader
from PartSegImage.image_writer import ImageWriter


def test_scaling(tmp_path):
    image = Image(np.zeros((10, 50, 50), dtype=np.uint8), (30, 0.1, 0.1), axes_order="ZYX")
    ImageWriter.save(image, tmp_path / "image.tif")
    read_image = TiffImageReader.read_image(tmp_path / "image.tif")
    assert np.all(np.isclose(image.spacing, read_image.spacing))


def test_save_mask(tmp_path):
    data = np.zeros((10, 40, 40), dtype=np.uint8)
    data[1:-1, 1:-1, 1:-1] = 1
    data[2:-3, 4:-4, 4:-4] = 2

    mask = np.array(data > 0).astype(np.uint8)

    image = Image(data, (0.4, 0.1, 0.1), mask=mask, axes_order="ZYX")
    ImageWriter.save_mask(image, tmp_path / "mask.tif")

    read_mask = TiffImageReader.read_image(tmp_path / "mask.tif")
    assert np.all(np.isclose(read_mask.spacing, image.spacing))


def test_ome_save(tmp_path):
    data = np.zeros((10, 20, 20, 2), dtype=np.uint8)
    image = Image(
        data, image_spacing=(30 * 10 ** -6, 10 * 10 ** -6, 10 * 10 ** -6), axes_order="ZYXC", channel_names=["a", "b"]
    )
    ImageWriter.save(image, tmp_path / "test.tif")

    with tifffile.TiffFile(tmp_path / "test.tif") as tiff:
        assert tiff.is_ome
        assert isinstance(tiff.ome_metadata, str)
        meta_data = tifffile.xml2dict(tiff.ome_metadata)["OME"]["Image"]["Pixels"]
        assert "PhysicalSizeX" in meta_data
        assert meta_data["PhysicalSizeX"] == 10
        assert "PhysicalSizeXUnit" in meta_data
        assert meta_data["PhysicalSizeXUnit"] == "um"
        assert len(meta_data["Channel"]) == 2
        assert meta_data["Channel"][0]["Name"] == "a"
        assert meta_data["Channel"][1]["Name"] == "b"
