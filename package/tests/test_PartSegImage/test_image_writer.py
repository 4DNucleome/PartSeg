import numpy as np
import numpy.testing as npt
import pytest
import tifffile
from lxml import etree  # nosec

from PartSegImage import ChannelInfo
from PartSegImage.image import Image
from PartSegImage.image_reader import TiffImageReader
from PartSegImage.image_writer import IMAGEJImageWriter, ImageWriter


@pytest.fixture(scope="module")
def ome_xml(bundle_test_dir):
    return etree.XMLSchema(file=str(bundle_test_dir / "ome.xsd.xml"))


def test_scaling(tmp_path):
    image = Image(np.zeros((10, 50, 50), dtype=np.uint8), spacing=(30, 0.1, 0.1), axes_order="ZYX")
    ImageWriter.save(image, tmp_path / "image.tif")
    read_image = TiffImageReader.read_image(tmp_path / "image.tif")
    assert np.all(np.isclose(image.spacing, read_image.spacing))


def test_save_mask(tmp_path):
    data = np.zeros((10, 40, 40), dtype=np.uint8)
    data[1:-1, 1:-1, 1:-1] = 1
    data[2:-3, 4:-4, 4:-4] = 2

    mask = np.array(data > 0).astype(np.uint8)

    image = Image(data, spacing=(0.4, 0.1, 0.1), mask=mask, axes_order="ZYX")
    ImageWriter.save_mask(image, tmp_path / "mask.tif")

    read_mask = TiffImageReader.read_image(tmp_path / "mask.tif")
    assert np.all(np.isclose(read_mask.spacing, image.spacing))
    image.set_mask(None)
    ImageWriter.save_mask(image, tmp_path / "mask2.tif")
    assert not (tmp_path / "mask2.tif").exists()


@pytest.mark.parametrize("z_size", [1, 10])
def test_ome_save(tmp_path, bundle_test_dir, ome_xml, z_size):
    data = np.zeros((z_size, 20, 20, 2), dtype=np.uint8)
    image = Image(
        data,
        spacing=(27 * 10**-6, 6 * 10**-6, 6 * 10**-6),
        axes_order="ZYXC",
        channel_info=[ChannelInfo(name="a"), ChannelInfo(name="b")],
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
        ome_xml.assert_(xml_file)  # noqa: PT009
    read_image = TiffImageReader.read_image(tmp_path / "test.tif")
    assert np.allclose(read_image.spacing, image.spacing)
    assert np.allclose(read_image.shift, image.shift)
    assert read_image.channel_names == ["a", "b"]
    assert read_image.name == "Test"


def test_scaling_imagej(tmp_path):
    image = Image(np.zeros((10, 50, 50), dtype=np.uint8), spacing=(30, 0.1, 0.1), axes_order="ZYX")
    IMAGEJImageWriter.save(image, tmp_path / "image.tif")
    read_image = TiffImageReader.read_image(tmp_path / "image.tif")
    assert np.all(np.isclose(image.spacing, read_image.spacing))


def test_scaling_imagej_fail(tmp_path):
    image = Image(np.zeros((10, 50, 50), dtype=np.float64), spacing=(30, 0.1, 0.1), axes_order="ZYX")
    with pytest.raises(ValueError, match="Data type float64"):
        IMAGEJImageWriter.save(image, tmp_path / "image.tif")


def test_imagej_write_all_metadata(tmp_path, data_test_dir):
    image = TiffImageReader.read_image(data_test_dir / "stack1_components" / "stack1_component1.tif")
    assert "coloring: " in str(image)
    assert "coloring: None" not in str(image)
    IMAGEJImageWriter.save(image, tmp_path / "image.tif")

    image2 = TiffImageReader.read_image(tmp_path / "image.tif")

    npt.assert_array_equal(image2.default_coloring, image.get_imagej_colors())


def test_imagej_save_color(tmp_path):
    data = np.zeros((4, 20, 20), dtype=np.uint8)
    data[:, 2:-2, 2:-2] = 20
    img = Image(
        data,
        spacing=(0.4, 0.1, 0.1),
        axes_order="CYX",
        channel_info=[
            ChannelInfo(name="ch1", color_map="blue", contrast_limits=(0, 20)),
            ChannelInfo(name="ch2", color_map="#FFAA00", contrast_limits=(0, 30)),
            ChannelInfo(name="ch3", color_map="#FB1", contrast_limits=(0, 25)),
            ChannelInfo(name="ch4", color_map=(0, 180, 0), contrast_limits=(0, 22)),
        ],
    )
    assert img.get_colors()[:3] == ["blue", "#FFAA00", "#FB1"]
    assert tuple(img.get_colors()[3]) == (0, 180, 0)
    IMAGEJImageWriter.save(img, tmp_path / "image.tif")
    image2 = TiffImageReader.read_image(tmp_path / "image.tif")
    assert image2.channel_names == ["ch1", "ch2", "ch3", "ch4"]
    assert image2.ranges == [(0, 20), (0, 30), (0, 25), (0, 22)]
    assert tuple(image2.default_coloring[0][:, -1]) == (0, 0, 255)
    assert tuple(image2.default_coloring[1][:, -1]) == (255, 170, 0)
    assert tuple(image2.default_coloring[2][:, -1]) == (255, 187, 17)
    assert tuple(image2.default_coloring[3][:, -1]) == (0, 180, 0)


def test_save_mask_imagej(tmp_path):
    data = np.zeros((10, 40, 40), dtype=np.uint8)
    data[1:-1, 1:-1, 1:-1] = 1
    data[2:-3, 4:-4, 4:-4] = 2

    mask = np.array(data > 0).astype(np.uint8)

    image = Image(data, spacing=(0.4, 0.1, 0.1), mask=mask, axes_order="ZYX")
    IMAGEJImageWriter.save_mask(image, tmp_path / "mask.tif")

    read_mask = TiffImageReader.read_image(tmp_path / "mask.tif")
    assert np.all(np.isclose(read_mask.spacing, image.spacing))
