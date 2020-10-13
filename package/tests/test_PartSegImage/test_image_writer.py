import numpy as np

from PartSegImage.image import Image
from PartSegImage.image_reader import TiffImageReader
from PartSegImage.image_writer import ImageWriter


def test_scaling(tmp_path):
    image = Image(np.zeros((10, 50, 50), dtype=np.uint8), (30, 0.1, 0.1), axes_order="ZYX")
    ImageWriter.save(image, tmp_path / "image.tif")
    read_image = TiffImageReader.read_image(tmp_path / "image.tif")
    assert np.all(np.isclose(image.spacing, read_image.spacing))
