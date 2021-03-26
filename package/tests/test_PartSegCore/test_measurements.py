import itertools
import os
from functools import partial, reduce
from math import isclose, pi
from operator import eq, lt

import numpy as np
import pytest
from sympy import symbols

from PartSegCore.analysis import load_metadata
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, Node, PerComponent
from PartSegCore.analysis.measurement_calculation import (
    HARALIC_FEATURES,
    MEASUREMENT_DICT,
    ComponentsInfo,
    ComponentsNumber,
    Diameter,
    DistanceMaskSegmentation,
    DistancePoint,
    FirstPrincipalAxisLength,
    Haralick,
    MaximumPixelBrightness,
    MeanPixelBrightness,
    MeasurementProfile,
    MeasurementResult,
    MedianPixelBrightness,
    MinimumPixelBrightness,
    Moment,
    PixelBrightnessSum,
    RimPixelBrightnessSum,
    RimVolume,
    SecondPrincipalAxisLength,
    Sphericity,
    SplitOnPartPixelBrightnessSum,
    SplitOnPartVolume,
    StandardDeviationOfPixelBrightness,
    Surface,
    ThirdPrincipalAxisLength,
    Volume,
    Voxels,
)
from PartSegCore.autofit import density_mass_center
from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegImage import Image


def get_cube_array():
    data = np.zeros((1, 50, 100, 100, 1), dtype=np.uint16)
    data[0, 10:40, 20:80, 20:80] = 50
    data[0, 15:35, 30:70, 30:70] = 70
    return data


def get_cube_image():
    return Image(get_cube_array(), (100, 50, 50), "")


@pytest.fixture(name="cube_image")
def cube_image_fixture():
    return get_cube_image()


@pytest.fixture
def cube_mask_40(cube_image):
    return cube_image.get_channel(0)[0] > 40


@pytest.fixture
def cube_mask_60(cube_image):
    return cube_image.get_channel(0)[0] > 60


def get_square_image():
    return Image(get_cube_array()[:, 25:26], (100, 50, 50), "")


@pytest.fixture(name="square_image")
def square_image_fixture():
    return get_square_image()


def get_two_components_array():
    data = np.zeros((1, 20, 30, 60, 1), dtype=np.uint16)
    data[0, 3:-3, 2:-2, 2:19] = 60
    data[0, 3:-3, 2:-2, 22:-2] = 50
    return data


def get_two_components_image():
    return Image(get_two_components_array(), (100, 50, 50), "")


def get_two_component_mask():
    mask = np.zeros(get_two_components_image().get_channel(0).shape[1:], dtype=np.uint8)
    mask[3:-3, 2:-2, 2:-2] = 1
    return mask


class TestDiameter:
    def test_parameters(self):
        assert Diameter.get_units(3) == symbols("{}")
        assert Diameter.get_units(2) == symbols("{}")
        assert Diameter.need_channel() is False
        leaf = Diameter.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    def test_cube(self, cube_image):
        mask1 = cube_image.get_channel(0)[0] > 40
        mask2 = cube_image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        assert Diameter.calculate_property(mask1, cube_image.spacing, 1) == np.sqrt(
            2 * (50 * 59) ** 2 + (100 * 29) ** 2
        )
        assert Diameter.calculate_property(mask2, cube_image.spacing, 1) == np.sqrt(
            2 * (50 * 39) ** 2 + (100 * 19) ** 2
        )
        assert Diameter.calculate_property(mask3, cube_image.spacing, 1) == np.sqrt(
            2 * (50 * 59) ** 2 + (100 * 29) ** 2
        )

    def test_square(self, square_image):
        mask1 = square_image.get_channel(0)[0] > 40
        mask2 = square_image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        assert Diameter.calculate_property(mask1, square_image.spacing, 1) == np.sqrt(2 * (50 * 59) ** 2)
        assert Diameter.calculate_property(mask2, square_image.spacing, 1) == np.sqrt(2 * (50 * 39) ** 2)
        assert Diameter.calculate_property(mask3, square_image.spacing, 1) == np.sqrt(2 * (50 * 59) ** 2)

    def test_scale(self):
        image = get_cube_image()
        mask1 = image.get_channel(0)[0] > 40
        assert isclose(
            Diameter.calculate_property(mask1, image.spacing, 2), 2 * np.sqrt(2 * (50 * 59) ** 2 + (100 * 29) ** 2)
        )
        image = get_square_image()
        mask1 = image.get_channel(0)[0] > 40
        assert isclose(Diameter.calculate_property(mask1, image.spacing, 2), 2 * np.sqrt(2 * (50 * 59) ** 2))

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0)[0] > 80
        assert Diameter.calculate_property(mask, image.spacing, 1) == 0


class TestPixelBrightnessSum:
    def test_parameters(self):
        assert PixelBrightnessSum.get_units(3) == symbols("Pixel_brightness")
        assert PixelBrightnessSum.get_units(2) == symbols("Pixel_brightness")
        assert PixelBrightnessSum.need_channel() is True
        leaf = PixelBrightnessSum.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        assert (
            PixelBrightnessSum.calculate_property(mask1, image.get_channel(0)) == 30 * 60 * 60 * 50 + 20 * 40 * 40 * 20
        )
        assert PixelBrightnessSum.calculate_property(mask2, image.get_channel(0)) == 20 * 40 * 40 * 70
        assert PixelBrightnessSum.calculate_property(mask3, image.get_channel(0)) == (30 * 60 * 60 - 20 * 40 * 40) * 50

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        assert PixelBrightnessSum.calculate_property(mask1, image.get_channel(0)) == 60 * 60 * 50 + 40 * 40 * 20
        assert PixelBrightnessSum.calculate_property(mask2, image.get_channel(0)) == 40 * 40 * 70
        assert PixelBrightnessSum.calculate_property(mask3, image.get_channel(0)) == (60 * 60 - 40 * 40) * 50

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0) > 80
        assert PixelBrightnessSum.calculate_property(mask, image.get_channel(0)) == 0


class TestVolume:
    def test_parameters(self):
        assert Volume.get_units(3) == symbols("{}") ** 3
        assert Volume.get_units(2) == symbols("{}") ** 2
        assert Volume.need_channel() is False
        leaf = Volume.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Volume.calculate_property(mask1, image.spacing, 1) == (100 * 30) * (50 * 60) * (50 * 60)
        assert Volume.calculate_property(mask2, image.spacing, 1) == (100 * 20) * (50 * 40) * (50 * 40)
        assert Volume.calculate_property(mask3, image.spacing, 1) == (100 * 30) * (50 * 60) * (50 * 60) - (100 * 20) * (
            50 * 40
        ) * (50 * 40)

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Volume.calculate_property(mask1, image.spacing, 1) == (50 * 60) * (50 * 60)
        assert Volume.calculate_property(mask2, image.spacing, 1) == (50 * 40) * (50 * 40)
        assert Volume.calculate_property(mask3, image.spacing, 1) == (50 * 60) * (50 * 60) - (50 * 40) * (50 * 40)

    def test_scale(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        assert Volume.calculate_property(mask1, image.spacing, 2) == 2 ** 3 * (100 * 30) * (50 * 60) * (50 * 60)

        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        assert Volume.calculate_property(mask1, image.spacing, 2) == 2 ** 2 * (50 * 60) * (50 * 60)

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0) > 80
        assert Volume.calculate_property(mask, image.spacing, 1) == 0


class TestVoxels:
    def test_parameters(self):
        assert Voxels.get_units(3) == symbols("1")
        assert Voxels.get_units(2) == symbols("1")
        assert Voxels.need_channel() is False
        leaf = Voxels.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Voxels.calculate_property(mask1) == 30 * 60 * 60
        assert Voxels.calculate_property(mask2) == 20 * 40 * 40
        assert Voxels.calculate_property(mask3) == 30 * 60 * 60 - 20 * 40 * 40

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0) > 40
        mask2 = image.get_channel(0) > 60
        mask3 = mask1 * ~mask2
        assert Voxels.calculate_property(mask1) == 60 * 60
        assert Voxels.calculate_property(mask2) == 40 * 40
        assert Voxels.calculate_property(mask3) == 60 * 60 - 40 * 40

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0) > 80
        assert Voxels.calculate_property(mask) == 0


class TestComponentsNumber:
    def test_parameters(self):
        assert ComponentsNumber.get_units(3) == symbols("count")
        assert ComponentsNumber.get_units(2) == symbols("count")
        assert ComponentsNumber.need_channel() is False
        leaf = ComponentsNumber.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is PerComponent.No
        assert leaf.channel is None

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

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0) > 80
        assert ComponentsNumber.calculate_property(mask) == 0


class TestMaximumPixelBrightness:
    def test_parameters(self):
        assert MaximumPixelBrightness.get_units(3) == symbols("Pixel_brightness")
        assert MaximumPixelBrightness.get_units(2) == symbols("Pixel_brightness")
        assert MaximumPixelBrightness.need_channel() is True
        leaf = MaximumPixelBrightness.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

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

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0) > 80
        assert MaximumPixelBrightness.calculate_property(mask, image.get_channel(0)) == 0


@pytest.mark.parametrize("threshold", [80, 60, 40, 0])
@pytest.mark.parametrize("image", [get_square_image(), get_cube_image()], ids=["square", "cube"])
@pytest.mark.parametrize(
    "calc_class,np_method",
    [
        (MinimumPixelBrightness, np.min),
        (MaximumPixelBrightness, np.max),
        (MedianPixelBrightness, np.median),
        (MeanPixelBrightness, np.mean),
        (StandardDeviationOfPixelBrightness, np.std),
    ],
)
def test_pixel_brightness(image, threshold, calc_class, np_method):
    channel = image.get_channel(0)
    mask = channel > threshold
    assert calc_class.calculate_property(mask, channel) == (np_method(channel[mask]) if np.any(mask) else 0)


@pytest.mark.parametrize(
    "calc_class",
    [MinimumPixelBrightness, MaximumPixelBrightness, MeanPixelBrightness, StandardDeviationOfPixelBrightness],
)
def test_parameters_pixel_brightness(calc_class):
    assert calc_class.get_units(3) == symbols("Pixel_brightness")
    assert calc_class.get_units(2) == symbols("Pixel_brightness")
    assert calc_class.need_channel() is True
    leaf = calc_class.get_starting_leaf()
    assert isinstance(leaf, Leaf)
    assert leaf.area is None
    assert leaf.per_component is None
    assert leaf.channel is None


class TestMoment:
    def test_parameters(self):
        assert Moment.get_units(3) == symbols("{}") ** 2 * symbols("Pixel_brightness")
        assert Moment.get_units(2) == symbols("{}") ** 2 * symbols("Pixel_brightness")
        assert Moment.need_channel() is True
        leaf = Moment.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    @pytest.mark.parametrize("image", [get_cube_image(), get_square_image()], ids=["cube", "square"])
    def test_image(self, image):
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = image.get_channel(0)[0] >= 0
        in1 = Moment.calculate_property(mask1, image.get_channel(0), image.spacing)
        in2 = Moment.calculate_property(mask2, image.get_channel(0), image.spacing)
        in3 = Moment.calculate_property(mask3, image.get_channel(0), image.spacing)
        assert in1 == in3
        assert in1 > in2

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0)[0] > 80
        assert Moment.calculate_property(mask, image.get_channel(0), image.spacing) == 0

    def test_values(self):
        spacing = (10, 6, 6)
        image_array = np.zeros((10, 16, 16))
        mask = np.ones(image_array.shape)
        image_array[5, 8, 8] = 1
        assert Moment.calculate_property(mask, image_array, spacing) == 0
        image_array[5, 8, 9] = 1
        assert Moment.calculate_property(mask, image_array, spacing) == (0.5 * 6) ** 2 * 2
        image_array = np.zeros((10, 16, 16))
        image_array[5, 8, 8] = 1
        image_array[5, 10, 8] = 3
        assert Moment.calculate_property(mask, image_array, spacing) == 9 ** 2 + 3 ** 2 * 3
        image_array = np.zeros((10, 16, 16))
        image_array[5, 6, 8] = 3
        image_array[5, 10, 8] = 3
        assert Moment.calculate_property(mask, image_array, spacing) == 3 * 2 * 12 ** 2

    def test_density_mass_center(self):
        spacing = (10, 6, 6)
        image_array = np.zeros((10, 16, 16))
        image_array[5, 8, 8] = 1
        assert np.all(np.array(density_mass_center(image_array, spacing)) == np.array((50, 48, 48)))
        image_array[5, 9, 8] = 1
        assert np.all(np.array(density_mass_center(image_array, spacing)) == np.array((50, 51, 48)))
        image_array[5, 8:10, 9] = 1
        assert np.all(np.array(density_mass_center(image_array, spacing)) == np.array((50, 51, 51)))
        image_array = np.zeros((10, 16, 16))
        image_array[2, 5, 5] = 1
        image_array[8, 5, 5] = 1
        assert np.all(np.array(density_mass_center(image_array, spacing)) == np.array((50, 30, 30)))
        image_array = np.zeros((10, 16, 16))
        image_array[3:8, 4:13, 4:13] = 1
        assert np.all(np.array(density_mass_center(image_array, spacing)) == np.array((50, 48, 48)))
        image_array = np.zeros((10, 16, 16))
        image_array[5, 8, 8] = 1
        image_array[5, 10, 8] = 3
        assert np.all(np.array(density_mass_center(image_array, spacing)) == np.array((50, 57, 48)))
        assert np.all(np.array(density_mass_center(image_array[5], spacing[1:])) == np.array((57, 48)))
        assert np.all(np.array(density_mass_center(image_array[5:6], spacing)) == np.array((0, 57, 48)))


class TestMainAxis:
    @pytest.mark.parametrize("method", [FirstPrincipalAxisLength, SecondPrincipalAxisLength, ThirdPrincipalAxisLength])
    def test_parameters(self, method):
        assert method.get_units(3) == symbols("{}")
        assert method.get_units(2) == symbols("{}")
        assert method.need_channel() is True
        leaf = method.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    @pytest.mark.parametrize("image", (get_cube_image(), get_square_image()), ids=["cube", "square"])
    @pytest.mark.parametrize(
        "method,scalar,last",
        [(FirstPrincipalAxisLength, 20, 0), (SecondPrincipalAxisLength, 10, 0), (ThirdPrincipalAxisLength, 10, 1)],
    )
    @pytest.mark.parametrize("threshold,len_scalar", [(40, 59), (60, 39)])
    @pytest.mark.parametrize("result_scalar", [1, 0.5, 3])
    def test_cube(self, image, method, scalar, threshold, len_scalar, last, result_scalar):
        image = image.substitute(image_spacing=(10, 10, 20))
        channel = image.get_channel(0)
        mask = channel[0] > threshold
        len_scalar = len_scalar - last * ((100 - threshold) / 2)
        if image.is_2d and last:
            return
        assert (
            method.calculate_property(
                area_array=mask,
                channel=channel,
                help_dict={},
                voxel_size=image.spacing,
                result_scalar=result_scalar,
                _area=AreaType.Mask,
            )
            == scalar * len_scalar * result_scalar
        )

    def test_empty(self, cube_image):
        mask = cube_image.get_channel(0)[0] > 80
        assert (
            ThirdPrincipalAxisLength.calculate_property(
                area_array=mask,
                channel=cube_image.get_channel(0),
                help_dict={},
                voxel_size=cube_image.spacing,
                result_scalar=1,
                _area=AreaType.ROI,
            )
            == 0
        )

    @pytest.mark.parametrize(
        "method,result",
        [(FirstPrincipalAxisLength, 20 * 59), (SecondPrincipalAxisLength, 10 * 59), (ThirdPrincipalAxisLength, 0)],
    )
    def test_without_help_dict(self, square_image, method, result):
        square_image = square_image.substitute(image_spacing=(10, 10, 20))
        mask1 = square_image.get_channel(0)[0] > 40
        assert (
            method.calculate_property(
                area_array=mask1,
                channel=square_image.get_channel(0),
                voxel_size=square_image.spacing,
                result_scalar=1,
                _area=AreaType.Mask,
            )
            == result
        )


class TestSurface:
    def test_parameters(self):
        assert Surface.get_units(3) == symbols("{}") ** 2
        assert Surface.get_units(2) == symbols("{}") ** 2
        assert Surface.need_channel() is False
        leaf = Surface.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        assert Surface.calculate_property(mask1, image.spacing, 1) == 6 * (60 * 50) ** 2
        assert Surface.calculate_property(mask2, image.spacing, 1) == 6 * (40 * 50) ** 2
        assert Surface.calculate_property(mask3, image.spacing, 1) == 6 * (60 * 50) ** 2 + 6 * (40 * 50) ** 2

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        assert Surface.calculate_property(mask1, image.spacing, 1) == 4 * (60 * 50)
        assert Surface.calculate_property(mask2, image.spacing, 1) == 4 * (40 * 50)
        assert Surface.calculate_property(mask3, image.spacing, 1) == 4 * (60 * 50) + 4 * (40 * 50)

    def test_scale(self):
        image = get_cube_image()
        mask1 = image.get_channel(0)[0] > 40
        assert Surface.calculate_property(mask1, image.spacing, 3) == 3 ** 2 * 6 * (60 * 50) ** 2

        image = get_square_image()
        mask1 = image.get_channel(0)[0] > 40
        assert Surface.calculate_property(mask1, image.spacing, 3) == 3 * 4 * (60 * 50)

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0)[0] > 80
        assert Surface.calculate_property(mask, image.spacing, 1) == 0


class TestRimVolume:
    def test_parameters(self):
        assert RimVolume.get_units(3) == symbols("{}") ** 3
        assert RimVolume.get_units(2) == symbols("{}") ** 2
        assert RimVolume.need_channel() is False
        leaf = RimVolume.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is AreaType.Mask
        assert leaf.per_component is None
        assert leaf.channel is None

    @pytest.mark.parametrize("image", [get_cube_image(), get_square_image()], ids=["cube", "square"])
    @pytest.mark.parametrize("scale", [1, 4])
    def test_image(self, image, scale):
        image = image.substitute(image_spacing=tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))

        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        result_scale = reduce(lambda x, y: x * y, image.voxel_size)
        exp = 2 if image.is_2d else 3
        assert (
            RimVolume.calculate_property(
                area_array=mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=scale,
                distance=10 * 50,
                units=Units.nm,
            )
            == np.count_nonzero(mask3) * result_scale * scale ** exp
        )
        assert (
            RimVolume.calculate_property(
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=scale,
                distance=10 * 50,
                units=Units.nm,
            )
            == 0
        )

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0)[0] > 80
        mask1 = image.get_channel(0)[0] > 40
        assert (
            RimVolume.calculate_property(
                area_array=mask1,
                mask=mask,
                voxel_size=image.voxel_size,
                result_scalar=UNIT_SCALE[Units.nm.value],
                distance=10 * 50,
                units=Units.nm,
            )
            == 0
        )
        assert (
            RimVolume.calculate_property(
                area_array=mask,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=UNIT_SCALE[Units.nm.value],
                distance=10 * 50,
                units=Units.nm,
            )
            == 0
        )
        assert (
            RimVolume.calculate_property(
                area_array=mask,
                mask=mask,
                voxel_size=image.voxel_size,
                result_scalar=UNIT_SCALE[Units.nm.value],
                distance=10 * 50,
                units=Units.nm,
            )
            == 0
        )


class TestRimPixelBrightnessSum:
    def test_parameters(self):
        assert RimPixelBrightnessSum.get_units(3) == symbols("Pixel_brightness")
        assert RimPixelBrightnessSum.get_units(2) == symbols("Pixel_brightness")
        assert RimPixelBrightnessSum.need_channel() is True
        leaf = RimPixelBrightnessSum.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is AreaType.Mask
        assert leaf.per_component is None
        assert leaf.channel is None

    @pytest.mark.parametrize("image", [get_cube_image(), get_square_image()], ids=["cube", "square"])
    def test_image(self, image):
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        assert (
            RimPixelBrightnessSum.calculate_property(
                area_array=mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                distance=10 * 50,
                units=Units.nm,
                channel=image.get_channel(0),
            )
            == np.count_nonzero(mask3) * 50
        )
        assert (
            RimPixelBrightnessSum.calculate_property(
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                distance=10 * 50,
                units=Units.nm,
                channel=image.get_channel(0),
            )
            == 0
        )

    def test_empty(self):
        image = get_cube_image()
        mask = image.get_channel(0)[0] > 80
        mask1 = image.get_channel(0)[0] > 40
        assert (
            RimPixelBrightnessSum.calculate_property(
                area_array=mask1,
                mask=mask,
                voxel_size=image.voxel_size,
                distance=10 * 50,
                channel=image.get_channel(0),
                units=Units.nm,
            )
            == 0
        )
        assert (
            RimPixelBrightnessSum.calculate_property(
                area_array=mask,
                mask=mask1,
                voxel_size=image.voxel_size,
                distance=10 * 50,
                channel=image.get_channel(0),
                units=Units.nm,
            )
            == 0
        )
        assert (
            RimPixelBrightnessSum.calculate_property(
                area_array=mask,
                mask=mask,
                voxel_size=image.voxel_size,
                distance=10 * 50,
                channel=image.get_channel(0),
                units=Units.nm,
            )
            == 0
        )


class TestSphericity:
    def test_parameters(self):
        assert Sphericity.get_units(3) == 1
        assert Sphericity.get_units(2) == 1
        assert Sphericity.need_channel() is False
        leaf = Sphericity.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is None
        assert leaf.per_component is None
        assert leaf.channel is None

    def test_cube(self):
        image = get_cube_image()
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        mask1_radius = np.sqrt(2 * (50 * 59) ** 2 + (100 * 29) ** 2) / 2
        mask1_volume = np.count_nonzero(mask1) * reduce(lambda x, y: x * y, image.voxel_size)
        assert isclose(
            Sphericity.calculate_property(area_array=mask1, voxel_size=image.voxel_size, result_scalar=1),
            mask1_volume / (4 / 3 * pi * mask1_radius ** 3),
        )

        mask2_radius = np.sqrt(2 * (50 * 39) ** 2 + (100 * 19) ** 2) / 2
        mask2_volume = np.count_nonzero(mask2) * reduce(lambda x, y: x * y, image.voxel_size)
        assert isclose(
            Sphericity.calculate_property(area_array=mask2, voxel_size=image.voxel_size, result_scalar=1),
            mask2_volume / (4 / 3 * pi * mask2_radius ** 3),
        )

        mask3_radius = mask1_radius
        mask3_volume = np.count_nonzero(mask3) * reduce(lambda x, y: x * y, image.voxel_size)
        assert isclose(
            Sphericity.calculate_property(area_array=mask3, voxel_size=image.voxel_size, result_scalar=1),
            mask3_volume / (4 / 3 * pi * mask3_radius ** 3),
        )

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        mask3 = mask1 * ~mask2
        mask1_radius = np.sqrt(2 * (50 * 59) ** 2) / 2
        mask1_volume = np.count_nonzero(mask1) * reduce(lambda x, y: x * y, image.voxel_size)
        assert isclose(
            Sphericity.calculate_property(area_array=mask1, voxel_size=image.voxel_size, result_scalar=1),
            mask1_volume / (pi * mask1_radius ** 2),
        )

        mask2_radius = np.sqrt(2 * (50 * 39) ** 2) / 2
        mask2_volume = np.count_nonzero(mask2) * reduce(lambda x, y: x * y, image.voxel_size)
        assert isclose(
            Sphericity.calculate_property(area_array=mask2, voxel_size=image.voxel_size, result_scalar=1),
            mask2_volume / (pi * mask2_radius ** 2),
        )

        mask3_radius = mask1_radius
        mask3_volume = np.count_nonzero(mask3) * reduce(lambda x, y: x * y, image.voxel_size)
        assert isclose(
            Sphericity.calculate_property(area_array=mask3, voxel_size=image.voxel_size, result_scalar=1),
            mask3_volume / (pi * mask3_radius ** 2),
        )


@pytest.fixture
def two_comp_img():
    data = np.zeros((30, 30, 60), dtype=np.uint16)
    data[5:-5, 5:-5, 5:29] = 60
    data[5:-5, 5:-5, 31:-5] = 50
    return Image(data, (100, 100, 50), "", axes_order="ZYX")


class TestDistanceMaskSegmentation:
    def test_parameters(self):
        assert DistanceMaskSegmentation.get_units(3) == symbols("{}")
        assert DistanceMaskSegmentation.get_units(2) == symbols("{}")
        assert DistanceMaskSegmentation.need_channel() is True
        leaf = DistanceMaskSegmentation.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is AreaType.Mask
        assert leaf.per_component is None
        assert leaf.channel is None

    @pytest.mark.parametrize(
        "d_mask,d_seg", itertools.product([DistancePoint.Geometrical_center, DistancePoint.Mass_center], repeat=2)
    )
    def test_cube_zero(self, cube_image, d_mask, d_seg):
        mask1 = cube_image.get_channel(0)[0] > 40
        mask2 = cube_image.get_channel(0)[0] > 60
        assert (
            DistanceMaskSegmentation.calculate_property(
                channel=cube_image.get_channel(0),
                area_array=mask2,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
                distance_from_mask=d_mask,
                distance_to_segmentation=d_seg,
            )
            == 0
        )

    @pytest.mark.parametrize(
        "d_mask,d_seg,dist",
        [
            (DistancePoint.Border, DistancePoint.Geometrical_center, 1400),
            (DistancePoint.Geometrical_center, DistancePoint.Border, 900),
            (DistancePoint.Border, DistancePoint.Border, 500),
        ],
    )
    def test_cube(self, cube_image, d_mask, d_seg, dist):
        mask1 = cube_image.get_channel(0)[0] > 40
        mask2 = cube_image.get_channel(0)[0] > 60

        assert (
            DistanceMaskSegmentation.calculate_property(
                channel=cube_image.get_channel(0),
                area_array=mask2,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
                distance_from_mask=d_mask,
                distance_to_segmentation=d_seg,
            )
            == dist
        )

    @pytest.mark.parametrize(
        "comp1,comp2", itertools.product([DistancePoint.Geometrical_center, DistancePoint.Mass_center], repeat=2)
    )
    @pytest.mark.parametrize(
        "area_gen", [partial(eq, 50), partial(eq, 60), partial(lt, 0)], ids=["eq50", "eq60", "all"]
    )
    def test_two_components_center(self, comp1, comp2, two_comp_img, area_gen):
        channel = two_comp_img.get_channel(0)
        mask = np.zeros(two_comp_img.shape[1:-1], dtype=np.uint8)
        mask[2:-2, 2:-2, 2:-2] = 1
        area_array = area_gen(two_comp_img.get_channel(0)[0])
        if comp1 == DistancePoint.Geometrical_center:
            mask_mid = np.mean(np.nonzero(mask), axis=1)
        else:
            mask_mid = np.average(np.nonzero(mask), axis=1, weights=channel[0][mask > 0])
        if comp2 == DistancePoint.Geometrical_center:
            area_mid = np.mean(np.nonzero(area_array), axis=1)
        else:
            area_mid = np.average(np.nonzero(area_array), axis=1, weights=channel[0][area_array])
        assert isclose(
            DistanceMaskSegmentation.calculate_property(
                channel=channel,
                area_array=area_array,
                mask=mask,
                voxel_size=two_comp_img.voxel_size,
                result_scalar=1,
                distance_from_mask=comp1,
                distance_to_segmentation=comp2,
            ),
            np.sqrt(np.sum(((mask_mid - area_mid) * (100, 50, 50)) ** 2)),
        )

    def test_two_components_border(self, two_comp_img):
        mask = np.zeros(two_comp_img.shape[1:-1], dtype=np.uint8)
        mask[2:-2, 2:-2, 2:-2] = 1

        assert (
            DistanceMaskSegmentation.calculate_property(
                two_comp_img.get_channel(0),
                two_comp_img.get_channel(0)[0],
                mask,
                two_comp_img.voxel_size,
                1,
                DistancePoint.Border,
                DistancePoint.Geometrical_center,
            )
            == 1200
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                two_comp_img.get_channel(0),
                two_comp_img.get_channel(0)[0],
                mask,
                two_comp_img.voxel_size,
                1,
                DistancePoint.Geometrical_center,
                DistancePoint.Border,
            )
            == 50
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                two_comp_img.get_channel(0),
                two_comp_img.get_channel(0)[0],
                mask,
                two_comp_img.voxel_size,
                1,
                DistancePoint.Border,
                DistancePoint.Border,
            )
            == 150
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                two_comp_img.get_channel(0),
                two_comp_img.get_channel(0)[0] == 50,
                mask,
                two_comp_img.voxel_size,
                1,
                DistancePoint.Border,
                DistancePoint.Border,
            )
            == 150
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                two_comp_img.get_channel(0),
                two_comp_img.get_channel(0)[0] == 60,
                mask,
                two_comp_img.voxel_size,
                1,
                DistancePoint.Border,
                DistancePoint.Border,
            )
            == 150
        )

    def test_square(self):
        image = get_square_image()
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        assert (
            DistanceMaskSegmentation.calculate_property(
                image.get_channel(0),
                mask2,
                mask1,
                image.voxel_size,
                1,
                DistancePoint.Geometrical_center,
                DistancePoint.Geometrical_center,
            )
            == 0
        )
        mask3 = mask2.astype(np.uint8)
        mask3[:, 50:] = 2
        mask3[mask2 == 0] = 0

        assert (
            DistanceMaskSegmentation.calculate_property(
                image.get_channel(0),
                mask2,
                mask1,
                image.voxel_size,
                1,
                DistancePoint.Geometrical_center,
                DistancePoint.Geometrical_center,
            )
            == 0
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                mask3,
                mask3 == 1,
                mask1,
                image.voxel_size,
                1,
                DistancePoint.Geometrical_center,
                DistancePoint.Geometrical_center,
            )
            == 500
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                mask3,
                mask3 == 2,
                mask1,
                image.voxel_size,
                1,
                DistancePoint.Geometrical_center,
                DistancePoint.Geometrical_center,
            )
            == 500
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                mask3,
                mask3 == 1,
                mask1,
                image.voxel_size,
                1,
                DistancePoint.Geometrical_center,
                DistancePoint.Mass_center,
            )
            == 500
        )

        assert (
            DistanceMaskSegmentation.calculate_property(
                mask3,
                mask3 == 2,
                mask1,
                image.voxel_size,
                1,
                DistancePoint.Geometrical_center,
                DistancePoint.Mass_center,
            )
            == 500
        )

        assert isclose(
            DistanceMaskSegmentation.calculate_property(
                mask3, mask2, mask1, image.voxel_size, 1, DistancePoint.Geometrical_center, DistancePoint.Mass_center
            ),
            1000 * 2 / 3 - 500,
        )


class TestSplitOnPartVolume:
    def test_parameters(self):
        assert SplitOnPartVolume.get_units(3) == symbols("{}") ** 3
        assert SplitOnPartVolume.get_units(2) == symbols("{}") ** 2
        assert SplitOnPartVolume.need_channel() is False
        leaf = SplitOnPartVolume.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is AreaType.Mask
        assert leaf.per_component is None
        assert leaf.channel is None

    def test_cube_equal_radius(self, cube_image):
        cube_image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in cube_image.spacing))

        mask1 = cube_image.get_channel(0)[0] > 40
        mask2 = cube_image.get_channel(0)[0] > 60
        result_scale = reduce(lambda x, y: x * y, cube_image.voxel_size)

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask1,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == (30 * 60 * 60 - 20 * 40 * 40) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=2,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask1,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == (20 * 40 * 40 - 10 * 20 * 20) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=3,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask1,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == (10 * 20 * 20) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=4,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask1,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == 0
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask2,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == 0
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=2,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask2,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == (20 * 40 * 40 - 10 * 20 * 20) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=3,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask2,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == (10 * 20 * 20) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=4,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask2,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                result_scalar=1,
            )
            == 0
        )

    def test_result_scalar(self):
        image = get_cube_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        result_scale = reduce(lambda x, y: x * y, image.voxel_size)

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=3,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=2,
            )
            == (10 * 20 * 20) * result_scale * 8
        )

    @pytest.mark.parametrize(
        "nr,volume, diff_array",
        [
            (1, (40 * 60 * 60 - 36 * 52 * 52), False),
            (2, (36 * 52 * 52 - 30 * 40 * 40), False),
            (3, (30 * 40 * 40), False),
            (4, 0, False),
            (1, 0, True),
            (2, 0, True),
            (3, (30 * 40 * 40), True),
            (4, 0, True),
        ],
    )
    def test_cube_equal_volume_simple(self, nr, volume, diff_array):
        data = np.zeros((60, 100, 100), dtype=np.uint16)
        data[10:50, 20:80, 20:80] = 50
        data[15:45, 30:70, 30:70] = 70
        image = Image(data, (2, 1, 1), "", axes_order="ZYX")
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60
        result_scale = reduce(lambda x, y: x * y, image.voxel_size)

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=nr,
                num_of_parts=3,
                equal_volume=True,
                area_array=mask2 if diff_array else mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == volume * result_scale
        )

    def test_square_equal_radius(self):
        image = get_square_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60

        result_scale = reduce(lambda x, y: x * y, image.voxel_size)

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == (60 * 60 - 40 * 40) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=2,
                equal_volume=False,
                area_array=mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == (60 * 60 - 30 * 30) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == 0
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=2,
                equal_volume=False,
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == (40 * 40 - 30 * 30) * result_scale
        )

    def test_square_equal_volume(self):
        image = get_square_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60

        result_scale = reduce(lambda x, y: x * y, image.voxel_size)

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=3,
                equal_volume=True,
                area_array=mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == (60 * 60 - 50 * 50) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=2,
                equal_volume=True,
                area_array=mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == (60 * 60 - 44 * 44) * result_scale
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=3,
                equal_volume=True,
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == 0
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=1,
                num_of_parts=2,
                equal_volume=True,
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == 0
        )

        assert (
            SplitOnPartVolume.calculate_property(
                part_selection=2,
                num_of_parts=2,
                equal_volume=True,
                area_array=mask2,
                mask=mask1,
                voxel_size=image.voxel_size,
                result_scalar=1,
            )
            == (40 * 40) * result_scale
        )


class TestSplitOnPartPixelBrightnessSum:
    def test_parameters(self):
        assert SplitOnPartPixelBrightnessSum.get_units(3) == symbols("Pixel_brightness")
        assert SplitOnPartPixelBrightnessSum.get_units(2) == symbols("Pixel_brightness")
        assert SplitOnPartPixelBrightnessSum.need_channel() is True
        leaf = SplitOnPartPixelBrightnessSum.get_starting_leaf()
        assert isinstance(leaf, Leaf)
        assert leaf.area is AreaType.Mask
        assert leaf.per_component is None
        assert leaf.channel is None

    @pytest.mark.parametrize(
        "nr, sum_val, diff_array",
        [
            (1, (30 * 60 * 60 - 20 * 40 * 40) * 50, False),
            (2, (20 * 40 * 40 - 10 * 20 * 20) * 70, False),
            (3, (10 * 20 * 20) * 70, False),
            (4, 0, False),
            (1, 0, True),
            (2, (20 * 40 * 40 - 10 * 20 * 20) * 70, True),
            (3, (10 * 20 * 20) * 70, True),
            (4, 0, True),
        ],
    )
    def test_cube_equal_radius(self, cube_image, nr, sum_val, diff_array):
        cube_image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in cube_image.spacing))

        mask1 = cube_image.get_channel(0)[0] > 40
        mask2 = cube_image.get_channel(0)[0] > 60

        assert (
            SplitOnPartPixelBrightnessSum.calculate_property(
                part_selection=nr,
                num_of_parts=3,
                equal_volume=False,
                area_array=mask2 if diff_array else mask1,
                mask=mask1,
                voxel_size=cube_image.voxel_size,
                channel=cube_image.get_channel(0),
            )
            == sum_val
        )

    @pytest.mark.parametrize(
        "nr, sum_val, diff_array",
        [
            (1, (40 * 60 * 60 - 36 * 52 * 52) * 50, False),
            (2, (36 * 52 * 52 - 30 * 40 * 40) * 50, False),
            (3, (30 * 40 * 40) * 70, False),
            (4, 0, False),
            (1, 0, True),
            (2, 0, True),
            (3, (30 * 40 * 40) * 70, True),
            (4, 0, True),
        ],
    )
    def test_cube_equal_volume(self, nr, sum_val, diff_array):
        data = np.zeros((1, 60, 100, 100, 1), dtype=np.uint16)
        data[0, 10:50, 20:80, 20:80] = 50
        data[0, 15:45, 30:70, 30:70] = 70
        image = Image(data, (100, 50, 50), "")
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60

        assert (
            SplitOnPartPixelBrightnessSum.calculate_property(
                part_selection=nr,
                num_of_parts=3,
                equal_volume=True,
                area_array=mask2 if diff_array else mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                channel=image.get_channel(0),
            )
            == sum_val
        )

    @pytest.mark.parametrize(
        "nr, sum_val, diff_array, equal_volume",
        [
            (3, (60 * 60 - 40 * 40) * 50, False, False),
            (2, (60 * 60 - 40 * 40) * 50 + (40 * 40 - 30 * 30) * 70, False, False),
            (3, 0, True, False),
            (2, (40 * 40 - 30 * 30) * 70, True, False),
            (3, (60 * 60 - 50 * 50) * 50, False, True),
            (2, (60 * 60 - 44 * 44) * 50, False, True),
            (3, 0, True, True),
            (2, 0, True, True),
        ],
    )
    def test_square(self, nr, sum_val, diff_array, equal_volume):
        image = get_square_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        mask1 = image.get_channel(0)[0] > 40
        mask2 = image.get_channel(0)[0] > 60

        assert (
            SplitOnPartPixelBrightnessSum.calculate_property(
                part_selection=1,
                num_of_parts=nr,
                equal_volume=equal_volume,
                area_array=mask2 if diff_array else mask1,
                mask=mask1,
                voxel_size=image.voxel_size,
                channel=image.get_channel(0),
            )
            == sum_val
        )


class TestStatisticProfile:
    def test_cube_volume_area_type(self):
        image = get_cube_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        image.set_mask((image.get_channel(0)[0] > 40).astype(np.uint8))
        segmentation = (image.get_channel(0)[0] > 60).astype(np.uint8)

        statistics = [
            MeasurementEntry(
                "Mask Volume", Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No)
            ),
            MeasurementEntry(
                "Segmentation Volume",
                Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask without segmentation Volume",
                Volume.get_starting_leaf().replace_(area=AreaType.Mask_without_ROI, per_component=PerComponent.No),
            ),
        ]
        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.µm,
        )
        tot_vol, seg_vol, rim_vol = list(result.values())
        assert isclose(tot_vol[0], seg_vol[0] + rim_vol[0])
        assert result.get_units()[0] == "μm**3"

    def test_square_volume_area_type(self):
        image = get_square_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        image.set_mask((image.get_channel(0)[0] > 40).astype(np.uint8))
        segmentation = (image.get_channel(0)[0] > 60).astype(np.uint8)

        statistics = [
            MeasurementEntry(
                "Mask Volume", Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No)
            ),
            MeasurementEntry(
                "Segmentation Volume",
                Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask without segmentation Volume",
                Volume.get_starting_leaf().replace_(area=AreaType.Mask_without_ROI, per_component=PerComponent.No),
            ),
        ]
        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.µm,
        )
        tot_vol, seg_vol, rim_vol = list(result.values())
        assert isclose(tot_vol[0], seg_vol[0] + rim_vol[0])
        assert result.get_units()[0] == "μm**2"

    def test_cube_pixel_sum_area_type(self):
        image = get_cube_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        image.set_mask((image.get_channel(0)[0] > 40).astype(np.uint8))
        segmentation = (image.get_channel(0)[0] > 60).astype(np.uint8)

        statistics = [
            MeasurementEntry(
                "Mask PixelBrightnessSum",
                PixelBrightnessSum.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Segmentation PixelBrightnessSum",
                PixelBrightnessSum.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask without segmentation PixelBrightnessSum",
                PixelBrightnessSum.get_starting_leaf().replace_(
                    area=AreaType.Mask_without_ROI, per_component=PerComponent.No
                ),
            ),
        ]
        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.µm,
        )
        tot_vol, seg_vol, rim_vol = list(result.values())
        assert isclose(tot_vol[0], seg_vol[0] + rim_vol[0])

    def test_cube_surface_area_type(self):
        image = get_cube_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        image.set_mask((image.get_channel(0)[0] > 40).astype(np.uint8))
        segmentation = (image.get_channel(0)[0] > 60).astype(np.uint8)

        statistics = [
            MeasurementEntry(
                "Mask Surface", Surface.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No)
            ),
            MeasurementEntry(
                "Segmentation Surface",
                Surface.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask without segmentation Surface",
                Surface.get_starting_leaf().replace_(area=AreaType.Mask_without_ROI, per_component=PerComponent.No),
            ),
        ]
        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.µm,
        )
        tot_vol, seg_vol, rim_vol = list(result.values())
        assert isclose(tot_vol[0] + seg_vol[0], rim_vol[0])

    def test_cube_density(self):
        image = get_cube_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        image.set_mask((image.get_channel(0)[0] > 40).astype(np.uint8))
        segmentation = (image.get_channel(0)[0] > 60).astype(np.uint8)

        statistics = [
            MeasurementEntry(
                "Mask Volume", Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No)
            ),
            MeasurementEntry(
                "Segmentation Volume",
                Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask without segmentation Volume",
                Volume.get_starting_leaf().replace_(area=AreaType.Mask_without_ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask PixelBrightnessSum",
                PixelBrightnessSum.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Segmentation PixelBrightnessSum",
                PixelBrightnessSum.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask without segmentation PixelBrightnessSum",
                PixelBrightnessSum.get_starting_leaf().replace_(
                    area=AreaType.Mask_without_ROI, per_component=PerComponent.No
                ),
            ),
            MeasurementEntry(
                "Mask Volume/PixelBrightnessSum",
                Node(
                    Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No),
                    "/",
                    PixelBrightnessSum.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No),
                ),
            ),
            MeasurementEntry(
                "Segmentation Volume/PixelBrightnessSum",
                Node(
                    Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
                    "/",
                    PixelBrightnessSum.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
                ),
            ),
            MeasurementEntry(
                "Mask without segmentation Volume/PixelBrightnessSum",
                Node(
                    Volume.get_starting_leaf().replace_(area=AreaType.Mask_without_ROI, per_component=PerComponent.No),
                    "/",
                    PixelBrightnessSum.get_starting_leaf().replace_(
                        area=AreaType.Mask_without_ROI, per_component=PerComponent.No
                    ),
                ),
            ),
        ]
        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.µm,
        )
        values = list(result.values())
        for i in range(3):
            volume, brightness, density = values[i::3]
            assert isclose(volume[0] / brightness[0], density[0])

    def test_cube_volume_power(self):
        image = get_cube_image()
        image.set_spacing(tuple(x / UNIT_SCALE[Units.nm.value] for x in image.spacing))
        image.set_mask((image.get_channel(0)[0] > 40).astype(np.uint8))
        segmentation = (image.get_channel(0)[0] > 60).astype(np.uint8)

        statistics = [
            MeasurementEntry(
                "Mask Volume", Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No)
            ),
            MeasurementEntry(
                "Mask Volume power 2",
                Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No, power=2),
            ),
            MeasurementEntry(
                "Mask Volume 2",
                Node(
                    Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No, power=2),
                    "/",
                    Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No),
                ),
            ),
            MeasurementEntry(
                "Mask Volume power -1",
                Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.No, power=-1),
            ),
        ]
        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.µm,
        )
        vol1, vol2, vol3, vol4 = list(result.values())
        assert isclose(vol1[0], vol3[0])
        assert isclose(vol1[0] ** 2, vol2[0])
        assert isclose(vol1[0] * vol4[0], 1)

    def test_per_component_cache_collision(self):
        image = get_two_components_image()
        image.set_mask(get_two_component_mask())
        segmentation = np.zeros(image.mask.shape, dtype=np.uint8)
        segmentation[image.get_channel(0) == 50] = 1
        segmentation[image.get_channel(0) == 60] = 2
        statistics = [
            MeasurementEntry(
                "Volume", Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No)
            ),
            MeasurementEntry(
                "Volume per component",
                Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.Yes),
            ),
            MeasurementEntry(
                "Diameter",
                Diameter.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Diameter per component",
                Diameter.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.Yes),
            ),
            MeasurementEntry(
                "MaximumPixelBrightness",
                MaximumPixelBrightness.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "MaximumPixelBrightness per component",
                MaximumPixelBrightness.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.Yes),
            ),
            MeasurementEntry(
                "Sphericity",
                Sphericity.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Sphericity per component",
                Sphericity.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.Yes),
            ),
            MeasurementEntry(
                "LongestMainAxisLength",
                FirstPrincipalAxisLength.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "LongestMainAxisLength per component",
                FirstPrincipalAxisLength.get_starting_leaf().replace_(
                    area=AreaType.ROI, per_component=PerComponent.Yes
                ),
            ),
        ]

        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.nm,
        )
        assert result["Volume"][0] == result["Volume per component"][0][0] + result["Volume per component"][0][1]
        assert len(result["Diameter per component"][0]) == 2
        assert result["MaximumPixelBrightness"][0] == 60
        assert result["MaximumPixelBrightness per component"][0] == [50, 60]
        assert result["Sphericity per component"][0] == [
            Sphericity.calculate_property(
                area_array=segmentation[0] == 1, voxel_size=image.voxel_size, result_scalar=UNIT_SCALE[Units.nm.value]
            ),
            Sphericity.calculate_property(
                area_array=segmentation[0] == 2, voxel_size=image.voxel_size, result_scalar=UNIT_SCALE[Units.nm.value]
            ),
        ]
        assert result["LongestMainAxisLength"][0] == 55 * 50 * UNIT_SCALE[Units.nm.value]
        assert result["LongestMainAxisLength per component"][0][0] == 35 * 50 * UNIT_SCALE[Units.nm.value]
        assert result["LongestMainAxisLength per component"][0][1] == 26 * 50 * UNIT_SCALE[Units.nm.value]

    def test_all_variants(self, bundle_test_dir):
        """ This test check if all calculations finished, not values. """
        file_path = os.path.join(bundle_test_dir, "measurements_profile.json")
        assert os.path.exists(file_path)
        profile = load_metadata(file_path)["all_statistic"]
        image = get_two_components_image()
        image.set_mask(get_two_component_mask())
        segmentation = np.zeros(image.mask.shape, dtype=np.uint8)
        segmentation[image.get_channel(0) == 50] = 1
        segmentation[image.get_channel(0) == 60] = 2
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.nm,
        )
        names = {x.name for x in profile.chosen_fields}
        assert names == set(result.keys())

    def test_proportion(self):
        image = get_two_components_image()
        image.set_mask(get_two_component_mask())
        segmentation = np.zeros(image.mask.shape, dtype=np.uint8)
        segmentation[image.get_channel(0) == 50] = 1
        segmentation[image.get_channel(0) == 60] = 2
        leaf1 = Volume.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.Yes)
        leaf2 = Volume.get_starting_leaf().replace_(area=AreaType.Mask, per_component=PerComponent.Yes)
        leaf3 = Volume.get_starting_leaf().replace_(area=AreaType.Mask_without_ROI, per_component=PerComponent.Yes)
        leaf4 = PixelBrightnessSum.get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.Yes)
        statistics = [
            MeasurementEntry(
                "ROI Volume per component",
                leaf1,
            ),
            MeasurementEntry(
                "Mask Volume per component",
                leaf2,
            ),
            MeasurementEntry("ROI Volume per component/Mask Volume per component", Node(leaf1, "/", leaf2)),
            MeasurementEntry("Mask Volume per component/ROI Volume per component", Node(leaf2, "/", leaf1)),
            MeasurementEntry(
                "Mask Volume per component/Mask without ROI Volume per component", Node(leaf2, "/", leaf3)
            ),
            MeasurementEntry("Density per component", Node(leaf4, "/", leaf1)),
        ]
        profile = MeasurementProfile("statistic", statistics)
        result = profile.calculate(
            image,
            0,
            segmentation,
            result_units=Units.nm,
        )
        # TODO check values
        assert len(result["ROI Volume per component/Mask Volume per component"][0]) == 2
        assert len(result["Mask Volume per component/ROI Volume per component"][0]) == 2
        assert len(result["Mask Volume per component/Mask without ROI Volume per component"][0]) == 1
        assert len(result["Density per component"][0]) == 2


# noinspection DuplicatedCode
class TestMeasurementResult:
    def test_simple(self):
        info = ComponentsInfo(np.arange(0), np.arange(0), {})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = 5, "np", (PerComponent.No, AreaType.ROI)
        assert list(storage.keys()) == ["aa", "bb"]
        assert list(storage.values()) == [(1, ""), (5, "np")]
        assert storage.get_separated() == [[1, 5]]
        assert storage.get_labels() == ["aa", "bb"]
        storage.set_filename("test.tif")
        assert list(storage.keys()) == ["File name", "aa", "bb"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), (5, "np")]
        assert storage.get_separated() == [["test.tif", 1, 5]]
        assert storage.get_labels() == ["File name", "aa", "bb"]
        del storage["aa"]
        assert list(storage.keys()) == ["File name", "bb"]

    def test_simple2(self):
        info = ComponentsInfo(np.arange(1, 5), np.arange(1, 5), {i: [i] for i in range(1, 5)})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = 5, "np", (PerComponent.No, AreaType.ROI)
        assert list(storage.keys()) == ["aa", "bb"]
        assert list(storage.values()) == [(1, ""), (5, "np")]
        assert storage.get_separated() == [[1, 5]]
        assert storage.get_labels() == ["aa", "bb"]
        storage.set_filename("test.tif")
        assert list(storage.keys()) == ["File name", "aa", "bb"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), (5, "np")]
        assert storage.get_separated() == [["test.tif", 1, 5]]
        assert storage.get_labels() == ["File name", "aa", "bb"]

    def test_segmentation_components(self):
        info = ComponentsInfo(np.arange(1, 3), np.arange(0), {1: [], 2: []})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = [4, 5], "np", (PerComponent.Yes, AreaType.ROI)
        assert list(storage.keys()) == ["aa", "bb"]
        assert list(storage.values()) == [(1, ""), ([4, 5], "np")]
        assert storage.get_separated() == [[1, 1, 4], [2, 1, 5]]
        assert storage.get_labels() == ["Segmentation component", "aa", "bb"]
        storage.set_filename("test.tif")
        assert list(storage.keys()) == ["File name", "aa", "bb"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 4], ["test.tif", 2, 1, 5]]
        assert storage.get_labels() == ["File name", "Segmentation component", "aa", "bb"]
        storage["cc"] = [11, 3], "np", (PerComponent.Yes, AreaType.ROI)
        assert list(storage.keys()) == ["File name", "aa", "bb", "cc"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5], "np"), ([11, 3], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 4, 11], ["test.tif", 2, 1, 5, 3]]
        assert storage.get_labels() == ["File name", "Segmentation component", "aa", "bb", "cc"]
        assert storage.get_global_names() == ["File name", "aa"]

    def test_mask_components(self):
        info = ComponentsInfo(np.arange(1, 2), np.arange(1, 3), {1: [], 2: []})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = [4, 5], "np", (PerComponent.Yes, AreaType.Mask)
        assert list(storage.keys()) == ["aa", "bb"]
        assert list(storage.values()) == [(1, ""), ([4, 5], "np")]
        assert storage.get_labels() == ["Mask component", "aa", "bb"]
        assert storage.get_separated() == [[1, 1, 4], [2, 1, 5]]
        storage.set_filename("test.tif")
        assert list(storage.keys()) == ["File name", "aa", "bb"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 4], ["test.tif", 2, 1, 5]]
        assert storage.get_labels() == ["File name", "Mask component", "aa", "bb"]
        storage["cc"] = [11, 3], "np", (PerComponent.Yes, AreaType.Mask_without_ROI)
        assert list(storage.keys()) == ["File name", "aa", "bb", "cc"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5], "np"), ([11, 3], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 4, 11], ["test.tif", 2, 1, 5, 3]]
        assert storage.get_labels() == ["File name", "Mask component", "aa", "bb", "cc"]

    def test_mask_segmentation_components(self):
        info = ComponentsInfo(np.arange(1, 3), np.arange(1, 3), {1: [1], 2: [2]})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = [4, 5], "np", (PerComponent.Yes, AreaType.ROI)
        assert list(storage.keys()) == ["aa", "bb"]
        assert list(storage.values()) == [(1, ""), ([4, 5], "np")]
        assert storage.get_separated() == [[1, 1, 4], [2, 1, 5]]
        assert storage.get_labels() == ["Segmentation component", "aa", "bb"]
        storage.set_filename("test.tif")
        assert list(storage.keys()) == ["File name", "aa", "bb"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 4], ["test.tif", 2, 1, 5]]
        assert storage.get_labels() == ["File name", "Segmentation component", "aa", "bb"]
        storage["cc"] = [11, 3], "np", (PerComponent.Yes, AreaType.Mask)
        assert list(storage.keys()) == ["File name", "aa", "bb", "cc"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5], "np"), ([11, 3], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 1, 4, 11], ["test.tif", 2, 2, 1, 5, 3]]
        assert storage.get_labels() == ["File name", "Segmentation component", "Mask component", "aa", "bb", "cc"]

    def test_mask_segmentation_components2(self):
        info = ComponentsInfo(np.arange(1, 4), np.arange(1, 3), {1: [1], 2: [2], 3: [1]})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = [4, 5, 6], "np", (PerComponent.Yes, AreaType.ROI)
        assert list(storage.keys()) == ["aa", "bb"]
        assert list(storage.values()) == [(1, ""), ([4, 5, 6], "np")]
        assert storage.get_separated() == [[1, 1, 4], [2, 1, 5], [3, 1, 6]]
        assert storage.get_labels() == ["Segmentation component", "aa", "bb"]
        storage.set_filename("test.tif")
        assert list(storage.keys()) == ["File name", "aa", "bb"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5, 6], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 4], ["test.tif", 2, 1, 5], ["test.tif", 3, 1, 6]]
        assert storage.get_labels() == ["File name", "Segmentation component", "aa", "bb"]
        storage["cc"] = [11, 3], "np", (PerComponent.Yes, AreaType.Mask)
        assert list(storage.keys()) == ["File name", "aa", "bb", "cc"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5, 6], "np"), ([11, 3], "np")]
        assert storage.get_separated() == [
            ["test.tif", 1, 1, 1, 4, 11],
            ["test.tif", 2, 2, 1, 5, 3],
            ["test.tif", 3, 1, 1, 6, 11],
        ]
        assert storage.get_labels() == ["File name", "Segmentation component", "Mask component", "aa", "bb", "cc"]

    def test_mask_segmentation_components3(self):
        info = ComponentsInfo(np.arange(1, 4), np.arange(1, 3), {1: [1], 2: [2], 3: [1, 2]})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = [4, 5, 6], "np", (PerComponent.Yes, AreaType.ROI)
        assert list(storage.keys()) == ["aa", "bb"]
        assert list(storage.values()) == [(1, ""), ([4, 5, 6], "np")]
        assert storage.get_separated() == [[1, 1, 4], [2, 1, 5], [3, 1, 6]]
        assert storage.get_labels() == ["Segmentation component", "aa", "bb"]
        storage.set_filename("test.tif")
        assert list(storage.keys()) == ["File name", "aa", "bb"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5, 6], "np")]
        assert storage.get_separated() == [["test.tif", 1, 1, 4], ["test.tif", 2, 1, 5], ["test.tif", 3, 1, 6]]
        assert storage.get_labels() == ["File name", "Segmentation component", "aa", "bb"]
        storage["cc"] = [11, 3], "np", (PerComponent.Yes, AreaType.Mask)
        assert list(storage.keys()) == ["File name", "aa", "bb", "cc"]
        assert list(storage.values()) == [("test.tif", ""), (1, ""), ([4, 5, 6], "np"), ([11, 3], "np")]
        assert storage.get_separated() == [
            ["test.tif", 1, 1, 1, 4, 11],
            ["test.tif", 2, 2, 1, 5, 3],
            ["test.tif", 3, 1, 1, 6, 11],
            ["test.tif", 3, 2, 1, 6, 3],
        ]
        assert storage.get_labels() == ["File name", "Segmentation component", "Mask component", "aa", "bb", "cc"]


class TestHaralick:
    def test_base(self):
        data = np.zeros((10, 20, 20), dtype=np.uint8)
        data[1:-1, 3:-3, 3:-3] = 2
        data[1:-1, 4:-4, 4:-4] = 3
        mask = data > 0
        res = Haralick.calculate_property(mask, data, distance=1, feature=HARALIC_FEATURES[0])
        assert res.size == 1

    def test_4d_base(self):
        data = np.zeros((1, 10, 20, 20), dtype=np.uint8)
        data[0, :-1, 3:-3, 3:-3] = 2
        data[0, 1:-1, 4:-4, 4:-4] = 3
        mask = data > 0
        res = Haralick.calculate_property(mask, data, distance=1, feature=HARALIC_FEATURES[0])
        assert res.size == 1

    @pytest.mark.parametrize("feature", HARALIC_FEATURES)
    @pytest.mark.parametrize("distance", range(1, 5))
    def test_variants(self, feature, distance):
        data = np.zeros((10, 20, 20), dtype=np.uint8)
        data[1:-1, 3:-3, 3:-3] = 2
        data[1:-1, 4:-4, 4:-4] = 3
        mask = data > 0
        Haralick.calculate_property(mask, data, distance=distance, feature=feature)


@pytest.mark.parametrize("method", MEASUREMENT_DICT.values())
@pytest.mark.parametrize("dtype", [float, int, np.uint8, np.uint16, np.uint32, np.float16, np.float32])
def ttest_all_methods(method, dtype):
    data = np.zeros((10, 20, 20), dtype=dtype)
    data[1:-1, 3:-3, 3:-3] = 2
    data[1:-1, 4:-4, 4:-4] = 3
    roi = (data > 0).astype(np.uint8)
    mask = (data > 2).astype(np.uint8)

    res = method.calculate_property(
        area_array=roi,
        mask=mask,
        channel=data,
        channel_num=0,
        voxel_size=(1, 1, 1),
        result_scalar=1,
        roi_alternative={},
        roi_annotation={},
        **method.get_default_values(),
    )
    float(res)
