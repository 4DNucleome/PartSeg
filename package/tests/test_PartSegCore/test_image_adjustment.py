import numpy as np

from PartSegCore.image_transforming import InterpolateImage
from PartSegImage import Image


def get_flat_image():
    data = np.zeros((1, 1, 10, 10, 1), dtype=np.uint8)
    data[:, :, 2:-2, 2:-2] = 5
    return Image(data, (5, 5))


def get_cube_image():
    data = np.zeros((1, 10, 10, 10, 1), dtype=np.uint8)
    data[:, 2:-2, 2:-2, 2:-2] = 5
    return Image(data, (10, 5, 5))


def get_flat_image_up():
    data = np.ones((1, 1, 10, 10, 1), dtype=np.uint8) * 30
    data[:, :, 2:-2, 2:-2] = 10
    return Image(data, (5, 5))


def get_cube_image_up():
    data = np.ones((1, 10, 10, 10, 1), dtype=np.uint8) * 30
    data[:, 2:-2, 2:-2, 2:-2] = 10
    return Image(data, (10, 5, 5))


class TestInterpolateImage:
    def test_simple(self):
        image = get_flat_image()
        image_res = InterpolateImage.transform(image, {"scale": 1})
        assert np.all(image.get_data() == image_res.get_data())
        image = get_cube_image()
        image_res = InterpolateImage.transform(image, {"scale": 1})
        assert np.all(image.get_data() == image_res.get_data())

    def test_nonzero_border(self):
        image = get_flat_image_up()
        image_res = InterpolateImage.transform(image, {"scale": 2})
        assert image_res.get_data().min() > 0
        assert image_res.spacing == (2.5, 2.5)
        image = get_cube_image_up()
        image_res = InterpolateImage.transform(image, {"scale": 2})
        assert image_res.get_data().min() > 0
        assert image_res.spacing == (5, 2.5, 2.5)

    def test_default_interpolate(self):
        image = get_flat_image()
        image_res = InterpolateImage.transform(image, InterpolateImage.calculate_initial(image))
        assert image_res.spacing == (5, 5)
        assert image_res.get_data().shape == (1, 1, 10, 10, 1)
        image = get_cube_image()
        image_res = InterpolateImage.transform(image, InterpolateImage.calculate_initial(image))
        assert image_res.spacing == (5, 5, 5)
        assert image_res.get_data().shape == (1, 20, 10, 10, 1)

    def test_multiple_interpolate(self):
        image = get_cube_image()
        image_res = InterpolateImage.transform(image, {"scale_x": 2, "scale_y": 3, "scale_z": 4})
        assert image_res.spacing == (2.5, 5 / 3, 2.5)
        assert image_res.get_data().shape == (1, 40, 30, 20, 1)
