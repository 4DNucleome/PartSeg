import numpy as np

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
