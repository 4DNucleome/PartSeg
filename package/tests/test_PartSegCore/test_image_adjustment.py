# pylint: disable=no-self-use

import numpy as np

from PartSegCore.image_transforming import CombineChannels, CombineMode, InterpolateImage
from PartSegCore.image_transforming.image_projection import ImageProjection, ImageProjectionParams
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


def get_flat_image():
    data = np.zeros((1, 1, 10, 10), dtype=np.uint8)
    data[:, :, 2:-2, 2:-2] = 5
    return Image(data, (5, 5), axes_order="TZYX")


def get_cube_image():
    data = np.zeros((1, 10, 10, 10), dtype=np.uint8)
    data[:, 2:-2, 2:-2, 2:-2] = 5
    return Image(data, (10, 5, 5), axes_order="TZYX")


def get_cube_image_2ch():
    data = np.zeros((1, 2, 10, 10, 10), dtype=np.uint8)
    data[:, 0, 2:-2, 2:-2, 2:-2] = 5
    data[:, 1, 2:-2, 2:-2, 2:-2] = 10
    return Image(data, (10, 5, 5), axes_order="TCZYX")


def get_flat_image_up():
    data = np.ones((1, 1, 10, 10), dtype=np.uint8) * 30
    data[:, :, 2:-2, 2:-2] = 10
    return Image(data, (5, 5), axes_order="TZYX")


def get_cube_image_up():
    data = np.ones((1, 10, 10, 10), dtype=np.uint8) * 30
    data[:, 2:-2, 2:-2, 2:-2] = 10
    return Image(data, (10, 5, 5), axes_order="TZYX")


class TestInterpolateImage:
    def test_simple(self):
        image = get_flat_image()
        image_res, _ = InterpolateImage.transform(image, None, {"scale": 1})
        assert np.all(image.get_data() == image_res.get_data())
        image = get_cube_image()
        image_res, _ = InterpolateImage.transform(image, None, {"scale": 1})
        assert np.all(image.get_data() == image_res.get_data())

    def test_nonzero_border(self):
        image = get_flat_image_up()
        image_res, _ = InterpolateImage.transform(image, None, {"scale": 2})
        assert image_res.get_data().min() > 0
        assert image_res.spacing == (2.5, 2.5)
        image = get_cube_image_up()
        image_res, _ = InterpolateImage.transform(image, None, {"scale": 2})
        assert image_res.get_data().min() > 0
        assert image_res.spacing == (5, 2.5, 2.5)

    def test_default_interpolate(self):
        image = get_flat_image()
        image_res, _ = InterpolateImage.transform(image, None, InterpolateImage.calculate_initial(image))
        assert image_res.spacing == (5, 5)
        assert image_res.get_data().shape == (1, 1, 1, 10, 10)
        image = get_cube_image()
        image_res, _ = InterpolateImage.transform(image, None, InterpolateImage.calculate_initial(image))
        assert image_res.spacing == (5, 5, 5)
        assert image_res.get_data().shape == (1, 1, 20, 10, 10)

    def test_multiple_interpolate(self):
        image = get_cube_image()
        image_res, _ = InterpolateImage.transform(image, None, {"scale_x": 2, "scale_y": 3, "scale_z": 4})
        assert image_res.spacing == (2.5, 5 / 3, 2.5)
        assert image_res.get_data().shape == (1, 1, 40, 30, 20)


class TestImageProjection:
    def test_basic(self):
        image = get_cube_image()
        image_res, _ = ImageProjection.transform(image, ROIInfo(None), ImageProjectionParams())
        assert image_res.shape == (1, 1, 10, 10)

    def test_no_roi(self):
        image = get_cube_image()
        image_res, roi = ImageProjection.transform(image, ROIInfo(None), ImageProjectionParams(keep_roi=True))
        assert image_res.shape == (1, 1, 10, 10)
        assert roi is None

    def test_no_mask(self):
        image = get_cube_image()
        image_res, _ = ImageProjection.transform(image, ROIInfo(None), ImageProjectionParams(keep_mask=True))
        assert image_res.shape == (1, 1, 10, 10)
        assert image_res.mask is None

    def test_cast_roi(self):
        image = get_cube_image()
        roi_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        roi_arr[:, :5] = 1
        roi_arr[:, 5:] = 2
        image_res, roi_info = ImageProjection.transform(image, ROIInfo(roi_arr), ImageProjectionParams(keep_roi=True))
        assert image_res.shape == (1, 1, 10, 10)
        assert roi_info.roi.shape == (1, 1, 10, 10)
        assert np.all(roi_info.roi[0, 0] == roi_arr[0])

    def test_cast_mask(self):
        image = get_cube_image()
        mask_arr = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_arr[:, :5] = 1
        mask_arr[:, 5:] = 2
        image.set_mask(mask_arr)
        image_res, _ = ImageProjection.transform(image, ROIInfo(None), ImageProjectionParams(keep_mask=True))
        assert image_res.shape == (1, 1, 10, 10)
        assert image_res.mask.shape == (1, 1, 10, 10)


class TestCombineChannels:
    def test_simple(self):
        image = get_cube_image_2ch()
        image_res, _ = CombineChannels.transform(
            image, None, {"combine_mode": CombineMode.Max, "channel_0": True, "channel_1": False}
        )
        assert image_res.channels == 3
        assert np.all(image_res.get_channel(0) == image_res.get_channel(2))

    def test_no_channels(self):
        image = get_cube_image_2ch()
        image_res, _ = CombineChannels.transform(image, None, {"combine_mode": CombineMode.Max})
        assert image_res is image

    def test_sum(self):
        image = get_cube_image_2ch()
        image_res, _ = CombineChannels.transform(
            image, None, {"combine_mode": CombineMode.Sum, "channel_0": True, "channel_1": True}
        )
        assert image_res.channels == 3
        assert np.max(image_res.get_channel(2)) == 15

    def test_max(self):
        image = get_cube_image_2ch()
        image_res, _ = CombineChannels.transform(
            image, None, {"combine_mode": CombineMode.Max, "channel_0": True, "channel_1": True}
        )
        assert image_res.channels == 3
        assert np.max(image_res.get_channel(2)) == 10
