import numpy as np
from math import isclose
from PartSeg.tiff_image import Image
from PartSeg.utils.analysis.statistics_calculation import Diameter, PixelBrightnessSum, Volume, ComponentsNumber, \
    MaximumPixelBrightness, MinimumPixelBrightness, MeanPixelBrightness, MedianPixelBrightness, AreaType,\
    StandardDeviationOfPixelBrightness, MomentOfInertia, LongestMainAxisLength, MiddleMainAxisLength, \
    ShortestMainAxisLength, Surface


def get_cube_array():
    data = np.zeros((1, 50, 100, 100, 1), dtype=np.uint16)
    data[0, 10:40, 20:80, 20:80] = 50
    data[0, 15:35, 30:70, 30:70] = 70
    return data


def get_cube_image():
    return Image(get_cube_array(), (100, 50, 50), "")


def get_square_image():
    return Image(get_cube_array()[:, 25:26], (100, 50, 50), "")


class TestDiameter(object):
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Diameter.calculate_property(mask1, image.spacing, 1) == np.sqrt(2 * (50 * 59) ** 2 + (100 * 29) ** 2)
        assert Diameter.calculate_property(mask2, image.spacing, 1) == np.sqrt(2 * (50 * 39) ** 2 + (100 * 19) ** 2)
        assert Diameter.calculate_property(mask3, image.spacing, 1) == np.sqrt(2 * (50 * 59) ** 2 + (100 * 29) ** 2)


class TestPixelBrightnessSum(object):
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert PixelBrightnessSum.calculate_property(mask1,
                                                     image.get_channel(0)) == 30 * 60 * 60 * 50 + 20 * 40 * 40 * 20
        assert PixelBrightnessSum.calculate_property(mask2, image.get_channel(0)) == 20 * 40 * 40 * 70
        assert PixelBrightnessSum.calculate_property(mask3, image.get_channel(0)) == (30 * 60 * 60 - 20 * 40 * 40) * 50

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert PixelBrightnessSum.calculate_property(mask1, image.get_channel(0)) == 60 * 60 * 50 + 40 * 40 * 20
        assert PixelBrightnessSum.calculate_property(mask2, image.get_channel(0)) == 40 * 40 * 70
        assert PixelBrightnessSum.calculate_property(mask3, image.get_channel(0)) == (60 * 60 - 40 * 40) * 50


class TestVolume(object):
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Volume.calculate_property(mask1, image.spacing, 1) == (100 * 30) * (50 * 60) * (50 * 60)
        assert Volume.calculate_property(mask2, image.spacing, 1) == (100 * 20) * (50 * 40) * (50 * 40)
        assert Volume.calculate_property(mask3, image.spacing, 1) == \
            (100 * 30) * (50 * 60) * (50 * 60) - (100 * 20) * (50 * 40) * (50 * 40)

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Volume.calculate_property(mask1, image.spacing, 1) == (50 * 60) * (50 * 60)
        assert Volume.calculate_property(mask2, image.spacing, 1) == (50 * 40) * (50 * 40)
        assert Volume.calculate_property(mask3, image.spacing, 1) == (50 * 60) * (50 * 60) - (50 * 40) * (50 * 40)


class TestComponentsNumber(object):
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        assert ComponentsNumber.calculate_property(mask1) == 1
        assert ComponentsNumber.calculate_property(mask2) == 1
        assert ComponentsNumber.calculate_property(image.get_channel(0)) == 2

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        assert ComponentsNumber.calculate_property(mask1) == 1
        assert ComponentsNumber.calculate_property(mask2) == 1
        assert ComponentsNumber.calculate_property(image.get_channel(0)) == 2


class TestMaximumPixelBrightness:
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert MaximumPixelBrightness.calculate_property(mask1, image.get_channel(0)) == 70
        assert MaximumPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MaximumPixelBrightness.calculate_property(mask3, image.get_channel(0)) == 50

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert MaximumPixelBrightness.calculate_property(mask1, image.get_channel(0)) == 70
        assert MaximumPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MaximumPixelBrightness.calculate_property(mask3, image.get_channel(0)) == 50


class TestMinimumPixelBrightness:
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        assert MinimumPixelBrightness.calculate_property(mask1, image.get_channel(0)) == 50
        assert MinimumPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MinimumPixelBrightness.calculate_property(mask3, image.get_channel(0)) == 0

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        assert MinimumPixelBrightness.calculate_property(mask1, image.get_channel(0)) == 50
        assert MinimumPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MinimumPixelBrightness.calculate_property(mask3, image.get_channel(0)) == 0


class TestMedianPixelBrightness:
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        assert MedianPixelBrightness.calculate_property(mask1, image.get_channel(0)) == 50
        assert MedianPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MedianPixelBrightness.calculate_property(mask3, image.get_channel(0)) == 0

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        assert MedianPixelBrightness.calculate_property(mask1, image.get_channel(0)) == 50
        assert MedianPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MedianPixelBrightness.calculate_property(mask3, image.get_channel(0)) == 0


class TestMeanPixelBrightness:
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        assert MeanPixelBrightness.calculate_property(mask1, image.get_channel(0)) == \
            (30 * 60 * 60 * 50 + 20 * 40 * 40 * 20) / (30 * 60 * 60)
        assert MeanPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MeanPixelBrightness.calculate_property(mask3, image.get_channel(0)) == \
            (30 * 60 * 60 * 50 + 20 * 40 * 40 * 20) / (50 * 100 * 100)

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        assert MeanPixelBrightness.calculate_property(mask1, image.get_channel(0)) == \
            (60 * 60 * 50 + 40 * 40 * 20) / (60 * 60)
        assert MeanPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 70
        assert MeanPixelBrightness.calculate_property(mask3, image.get_channel(0)) == \
            (60 * 60 * 50 + 40 * 40 * 20) / (100 * 100)


class TestStandardDeviationOfPixelBrightness:
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        mean = (30 * 60 * 60 * 50 + 20 * 40 * 40 * 20) / (30 * 60 * 60)
        assert StandardDeviationOfPixelBrightness.calculate_property(mask1, image.get_channel(0)) == \
            np.sqrt(((30 * 60 * 60 - 20 * 40 * 40) * (50 - mean) ** 2 + ((20 * 40 * 40) * (70 - mean) ** 2)) / (
                    30 * 60 * 60))

        assert StandardDeviationOfPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 0
        mean = (30 * 60 * 60 * 50 + 20 * 40 * 40 * 20) / (50 * 100 * 100)
        assert isclose(StandardDeviationOfPixelBrightness.calculate_property(mask3, image.get_channel(0)),
                       np.sqrt(((30 * 60 * 60 - 20 * 40 * 40) * (50 - mean) ** 2 + ((20 * 40 * 40) * (70 - mean) ** 2) +
                                (50 * 100 * 100 - 30 * 60 * 60) * mean ** 2) / (50 * 100 * 100)))

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        mean = (60 * 60 * 50 + 40 * 40 * 20) / (60 * 60)
        assert StandardDeviationOfPixelBrightness.calculate_property(mask1, image.get_channel(0)) == \
            np.sqrt(((60 * 60 - 40 * 40) * (50 - mean) ** 2 + ((40 * 40) * (70 - mean) ** 2)) / (60 * 60))

        assert StandardDeviationOfPixelBrightness.calculate_property(mask2, image.get_channel(0)) == 0
        mean = (60 * 60 * 50 + 40 * 40 * 20) / (100 * 100)
        assert isclose(StandardDeviationOfPixelBrightness.calculate_property(mask3, image.get_channel(0)),
                       np.sqrt(((60 * 60 - 40 * 40) * (50 - mean) ** 2 + ((40 * 40) * (70 - mean) ** 2) +
                                (100 * 100 - 60 * 60) * mean ** 2) / (100 * 100)))


class TestMomentOfInertia:
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        in1 = MomentOfInertia.calculate_property(mask1, image.get_channel(0), image.spacing)
        in2 = MomentOfInertia.calculate_property(mask2, image.get_channel(0), image.spacing)
        in3 = MomentOfInertia.calculate_property(mask3, image.get_channel(0), image.spacing)
        assert in1 == in3
        assert in1 > in2

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = image.get_channel(0) >= 0
        in1 = MomentOfInertia.calculate_property(mask1, image.get_channel(0), image.spacing)
        in2 = MomentOfInertia.calculate_property(mask2, image.get_channel(0), image.spacing)
        in3 = MomentOfInertia.calculate_property(mask3, image.get_channel(0), image.spacing)
        assert in1 == in3
        assert in1 > in2


class TestMainAxis:
    def test_cube(self):
        array = get_cube_array()
        image = Image(array, (10, 10, 20))
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        assert LongestMainAxisLength.calculate_property(
            area_array=mask1, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 20 * 59
        assert MiddleMainAxisLength.calculate_property(
            area_array=mask1, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 10 * 59
        assert ShortestMainAxisLength.calculate_property(
            area_array=mask1, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 10 * 29
        assert LongestMainAxisLength.calculate_property(
            area_array=mask2, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 20 * 39
        assert MiddleMainAxisLength.calculate_property(
            area_array=mask2, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 10 * 39
        assert ShortestMainAxisLength.calculate_property(
            area_array=mask2, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 10 * 19

    def test_square(self):
        array = get_cube_array()
        image = Image(array[:, 25:26], (10, 10, 20))
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        assert LongestMainAxisLength.calculate_property(
            area_array=mask1, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 20 * 59
        assert MiddleMainAxisLength.calculate_property(
            area_array=mask1, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 10 * 59
        assert ShortestMainAxisLength.calculate_property(
            area_array=mask1, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 0
        assert LongestMainAxisLength.calculate_property(
            area_array=mask2, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 20 * 39
        assert MiddleMainAxisLength.calculate_property(
            area_array=mask2, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 10 * 39
        assert ShortestMainAxisLength.calculate_property(
            area_array=mask2, image=image.get_channel(0), help_dict={}, voxel_size=image.spacing, result_scalar=1,
            _area=AreaType.Mask
        ) == 0


class TestSurface:
    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Surface.calculate_property(mask1, image.spacing, 1) == 6 * (60 * 50) ** 2
        assert Surface.calculate_property(mask2, image.spacing, 1) == 6 * (40 * 50) ** 2
        assert Surface.calculate_property(mask3, image.spacing, 1) == 6 * (60 * 50) ** 2 + 6 * (40 * 50) ** 2

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Surface.calculate_property(mask1, image.spacing, 1) == 4 * (60 * 50)
        assert Surface.calculate_property(mask2, image.spacing, 1) == 4 * (40 * 50)
        assert Surface.calculate_property(mask3, image.spacing, 1) == 4 * (60 * 50) + 4 * (40 * 50)
