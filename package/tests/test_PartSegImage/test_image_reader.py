# pylint: disable=no-self-use
import math
import os.path
import shutil
from glob import glob
from importlib.metadata import version
from io import BytesIO

import numpy as np
import pytest
import tifffile
from packaging.version import parse as parse_version

import PartSegData
from PartSegImage import CziImageReader, GenericImageReader, Image, ObsepImageReader, OifImagReader, TiffImageReader


@pytest.fixture(autouse=True)
def _set_max_workers_czi(monkeypatch):
    # set max workers to 1 to get exception in case of problems
    monkeypatch.setattr("PartSegImage.image_reader.CZI_MAX_WORKERS", 1)


class TestImageClass:
    def test_tiff_image_read(self):
        image = TiffImageReader.read_image(PartSegData.segmentation_mask_default_image)
        assert isinstance(image, Image)
        assert np.all(np.isclose(image.spacing, (7.752248561753867e-08,) * 2))

    def test_tiff_image_read_buffer(self):
        with open(PartSegData.segmentation_mask_default_image, "rb") as f_p:
            buffer = BytesIO(f_p.read())
        image = TiffImageReader.read_image(buffer)
        assert isinstance(image, Image)
        assert np.all(np.isclose(image.spacing, (7.752248561753867e-08,) * 2))

    def test_czi_file_read(self, data_test_dir):
        image = CziImageReader.read_image(os.path.join(data_test_dir, "test_czi.czi"))
        assert np.count_nonzero(image.get_channel(0))
        assert image.channels == 4
        assert image.layers == 1

        assert image.file_path == os.path.join(data_test_dir, "test_czi.czi")

        assert np.all(np.isclose(image.spacing, (7.752248561753867e-08,) * 2))

    @pytest.mark.skipif(
        parse_version(version("czifile")) < parse_version("2019.7.2"),
        reason="There is no patch for czifile before 2019.7.2",
    )
    @pytest.mark.parametrize("file_name", ["test_czi_zstd0.czi", "test_czi_zstd1.czi", "test_czi_zstd1_hilo.czi"])
    def test_czi_file_read_compressed(self, data_test_dir, file_name):
        image = CziImageReader.read_image(os.path.join(data_test_dir, file_name))
        assert np.count_nonzero(image.get_channel(0))
        assert image.channels == 4
        assert image.layers == 1

        assert image.file_path == os.path.join(data_test_dir, file_name)

        assert np.all(np.isclose(image.spacing, (7.752248561753867e-08,) * 2))

    def test_czi_file_read_buffer(self, data_test_dir):
        with open(os.path.join(data_test_dir, "test_czi.czi"), "rb") as f_p:
            buffer = BytesIO(f_p.read())

        image = CziImageReader.read_image(buffer)
        assert np.count_nonzero(image.get_channel(0))
        assert image.channels == 4
        assert image.layers == 1
        assert image.file_path == ""

        assert np.all(np.isclose(image.spacing, (7.752248561753867e-08,) * 2))

    def test_oib_file_read(self, data_test_dir):
        image = OifImagReader.read_image(os.path.join(data_test_dir, "N2A_H2BGFP_dapi_falloidin_cycling1.oib"))
        assert image.channels == 3
        assert image.layers == 6
        assert np.all(np.isclose(image.spacing, (2.1e-07,) + (7.752248561753867e-08,) * 2))

    def test_oif_file_read(self, data_test_dir):
        image = OifImagReader.read_image(os.path.join(data_test_dir, "Image0003_01.oif"))
        assert image.channels == 1
        assert image.layers == 49
        assert np.all(np.isclose(image.spacing, (3.2e-07,) + (5.1e-08,) * 2))

    def test_read_with_mask(self, data_test_dir):
        image = TiffImageReader.read_image(
            os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif"),
            os.path.join(data_test_dir, "stack1_components", "stack1_component1_mask.tif"),
        )
        assert isinstance(image, Image)
        with pytest.raises(ValueError, match="Incompatible shape"):
            TiffImageReader.read_image(
                os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif"),
                os.path.join(data_test_dir, "stack1_components", "stack1_component2_mask.tif"),
            )

    def test_lsm_read(self, data_test_dir):
        image1 = TiffImageReader.read_image(os.path.join(data_test_dir, "test_lsm.lsm"))
        image2 = TiffImageReader.read_image(os.path.join(data_test_dir, "test_lsm.tif"))
        data = np.moveaxis(np.load(os.path.join(data_test_dir, "test_lsm.npy")), -1, 0)
        assert np.all(image1.get_data() == data)
        assert np.all(image2.get_data() == data)
        assert np.all(image1.get_data() == image2.get_data())

    def test_ome_read(self, data_test_dir):  # error in tifffile
        image1 = TiffImageReader.read_image(os.path.join(data_test_dir, "test_lsm2.tif"))
        image2 = TiffImageReader.read_image(os.path.join(data_test_dir, "test_lsm.tif"))
        data = np.moveaxis(np.load(os.path.join(data_test_dir, "test_lsm.npy")), -1, 0)
        assert np.all(image1.get_data() == data)
        assert np.all(image2.get_data() == data)
        assert np.all(image1.get_data() == image2.get_data())

    def test_generic_reader(self, data_test_dir):
        GenericImageReader.read_image(
            os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif"),
            os.path.join(data_test_dir, "stack1_components", "stack1_component1_mask.tif"),
        )
        GenericImageReader.read_image(os.path.join(data_test_dir, "test_czi.czi"))
        GenericImageReader.read_image(os.path.join(data_test_dir, "test_lsm2.tif"))
        GenericImageReader.read_image(os.path.join(data_test_dir, "test_lsm.tif"))
        GenericImageReader.read_image(os.path.join(data_test_dir, "Image0003_01.oif"))
        GenericImageReader.read_image(os.path.join(data_test_dir, "N2A_H2BGFP_dapi_falloidin_cycling1.oib"))

    def test_generic_reader_from_buffer(self, data_test_dir):
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")
        with open(file_path, "rb") as f_p:
            buffer = BytesIO(f_p.read())
        GenericImageReader().read(buffer)
        with open(file_path, "rb") as f_p:
            buffer = BytesIO(f_p.read())
        GenericImageReader().read(buffer, ext=".tif")
        with pytest.raises(NotImplementedError, match="Oif format is not supported"):
            GenericImageReader().read(buffer, ext=".oif")
        with pytest.raises(NotImplementedError, match="Obsep format is not supported"):
            GenericImageReader().read(buffer, ext=".obsep")

    def test_decode_int(self):
        assert TiffImageReader.decode_int(0) == [0, 0, 0, 0]
        assert TiffImageReader.decode_int(15) == [0, 0, 0, 15]
        assert TiffImageReader.decode_int(3 + 7 * 256 + 11 * 256**2 + 13 * 256**3) == [13, 11, 7, 3]

    def test_set_spacing(self):
        reader = TiffImageReader()
        reader.set_default_spacing((11, 12, 13))
        assert reader.default_spacing == (11, 12, 13)
        reader.set_default_spacing((5, 7))
        assert reader.default_spacing == (10**-6, 5, 7)

    def test_obsep_read(self, data_test_dir):
        image = ObsepImageReader.read_image(os.path.join(data_test_dir, "obsep", "test.obsep"))
        assert image.channels == 2
        assert np.allclose(image.spacing, (500 * 10**-9, 64 * 10**-9, 64 * 10**-9))
        assert image.channel_names == ["channel 1", "channel 2"]

    def test_obsep_deconv_read(self, data_test_dir, tmp_path):
        for el in glob(os.path.join(data_test_dir, "obsep", "*")):
            shutil.copy(os.path.join(data_test_dir, "obsep", el), tmp_path)
        image = GenericImageReader.read_image(tmp_path / "test.obsep")
        assert image.channels == 2
        assert np.allclose(image.spacing, (500 * 10**-9, 64 * 10**-9, 64 * 10**-9))
        assert image.channel_names == ["channel 1", "channel 2"]
        shutil.copy(tmp_path / "Cy5.TIF", tmp_path / "Cy5_decon2.TIF")
        image = GenericImageReader.read_image(tmp_path / "test.obsep")
        assert image.channels == 2
        shutil.copy(tmp_path / "Cy5.TIF", tmp_path / "Cy5_deconv.TIF")
        image = GenericImageReader.read_image(tmp_path / "test.obsep")
        assert image.channels == 3

    def test_double_axes_in_dim_read(self, data_test_dir):
        image = GenericImageReader.read_image(os.path.join(data_test_dir, "double_q_in_axes.tif"))
        assert image.layers == 360
        assert image.channels == 1
        assert image.stack_pos == 1
        assert image.plane_shape == (360, 32)


class CustomImage(Image):
    axis_order = "TCXYZ"


class CustomTiffReader(TiffImageReader):
    image_class = CustomImage


def test_change_class(data_test_dir):
    img = CustomTiffReader.read_image(os.path.join(data_test_dir, "test_lsm.tif"))
    assert isinstance(img, CustomImage)
    assert img.plane_shape == (1024, 1024)
    assert img.layers == 6
    assert img.channels == 3
    assert img.stack_pos == 3


def test_xml2dict():
    sample_text = """
    <level1>
        <level2>3.5322</level2>
    </level1>
    """
    data = tifffile.xml2dict(sample_text)
    assert math.isclose(data["level1"]["level2"], 3.5322)
