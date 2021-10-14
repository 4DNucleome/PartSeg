import numpy as np
import pytest
import tifffile
from lxml import etree  # nosec

from PartSegImage.image import Image
from PartSegImage.image_reader import TiffImageReader
from PartSegImage.image_writer import IMAGEJImageWriter, ImageWriter


@pytest.fixture(scope="module")
def ome_xml(bundle_test_dir):
    return etree.XMLSchema(file=str(bundle_test_dir / "ome.xsd.xml"))


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


@pytest.mark.parametrize("z_size", (1, 10))
def test_ome_save(tmp_path, bundle_test_dir, ome_xml, z_size):
    data = np.zeros((z_size, 20, 20, 2), dtype=np.uint8)
    image = Image(
        data,
        image_spacing=(27 * 10 ** -6, 6 * 10 ** -6, 6 * 10 ** -6),
        axes_order="ZYXC",
        channel_names=["a", "b"],
        shift=(10, 9, 8),
        name="Test",
    )
    ImageWriter.save(image, tmp_path / "test.tif")

    with tifffile.TiffFile(tmp_path / "test.tif") as tiff:
        assert tiff.is_ome
        assert isinstance(tiff.ome_metadata, str)
        meta_data = tifffile.xml2dict(tiff.ome_metadata)["OME"]["Image"]
        assert "PhysicalSizeX" in meta_data["Pixels"]
        assert meta_data["Pixels"]["PhysicalSizeX"] == 6
        assert "PhysicalSizeXUnit" in meta_data["Pixels"]
        assert meta_data["Pixels"]["PhysicalSizeXUnit"] == "µm"
        assert len(meta_data["Pixels"]["Channel"]) == 2
        assert meta_data["Pixels"]["Channel"][0]["Name"] == "a"
        assert meta_data["Pixels"]["Channel"][1]["Name"] == "b"
        assert meta_data["Name"] == "Test"
        xml_file = etree.fromstring(tiff.ome_metadata.encode("utf8"))  # nosec
        ome_xml.assert_(xml_file)
    read_image = TiffImageReader.read_image(tmp_path / "test.tif")
    assert np.allclose(read_image.spacing, image.spacing)
    assert np.allclose(read_image.shift, image.shift)
    assert read_image.channel_names == ["a", "b"]
    assert read_image.name == "Test"


def test_scaling_imagej(tmp_path):
    image = Image(np.zeros((10, 50, 50), dtype=np.uint8), (30, 0.1, 0.1), axes_order="ZYX")
    IMAGEJImageWriter.save(image, tmp_path / "image.tif")
    read_image = TiffImageReader.read_image(tmp_path / "image.tif")
    assert np.all(np.isclose(image.spacing, read_image.spacing))


def test_save_mask_imagej(tmp_path):
    data = np.zeros((10, 40, 40), dtype=np.uint8)
    data[1:-1, 1:-1, 1:-1] = 1
    data[2:-3, 4:-4, 4:-4] = 2

    mask = np.array(data > 0).astype(np.uint8)

    image = Image(data, (0.4, 0.1, 0.1), mask=mask, axes_order="ZYX")
    IMAGEJImageWriter.save_mask(image, tmp_path / "mask.tif")

    read_mask = TiffImageReader.read_image(tmp_path / "mask.tif")
    assert np.all(np.isclose(read_mask.spacing, image.spacing))
