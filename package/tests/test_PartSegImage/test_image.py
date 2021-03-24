import os

import numpy as np
import pytest

from PartSegImage import Image, ImageWriter, TiffImageReader
from PartSegImage.image import FRAME_THICKNESS


class TestImageBase:
    image_class = Image

    def needed_shape(self, shape, axes: str, drop: str):
        new_axes = self.image_class.axis_order
        for el in drop:
            new_axes = new_axes.replace(el, "")
        res_shape = [1] * len(new_axes)
        for size, name in zip(shape, axes):
            res_shape[new_axes.index(name)] = size
        return tuple(res_shape)

    def image_shape(self, shape, axes):
        return self.needed_shape(shape, axes, "")

    def mask_shape(self, shape, axes):
        return self.needed_shape(shape, axes, "C")

    def reorder_axes_letter(self, letters: str):
        res = "".join(x for x in self.image_class.axis_order if x in letters)
        assert len(res) == len(letters)
        return res

    def prepare_mask_shape(self, shape):
        base_axes = set("TZYX")
        refer_axes = self.image_class.axis_order.replace("C", "")
        i, j = 0, 0
        new_shape = [1] * len(refer_axes)
        for i, val in enumerate(refer_axes):
            if val in base_axes:
                new_shape[i] = shape[j]
                j += 1
        return new_shape

    def prepare_image_initial_shape(self, shape, channel):
        new_shape = self.prepare_mask_shape(shape)
        new_shape.insert(self.image_class.axis_order.index("C"), channel)
        return new_shape

    def test_fit_mask_simple(self):
        initial_shape = self.prepare_image_initial_shape([1, 10, 20, 20], 1)
        data = np.zeros(initial_shape, np.uint8)
        image = self.image_class(data, (1, 1, 1), "")
        mask = np.zeros((1, 10, 20, 20), np.uint8)
        mask[0, 2:-2, 4:-4, 4:-4] = 5
        image.fit_mask_to_image(mask)

    def test_fit_mask_mapping_val(self):
        initial_shape = self.prepare_image_initial_shape([1, 10, 20, 20], 1)
        data = np.zeros(initial_shape, np.uint8)
        image = self.image_class(data, (1, 1, 1), "")
        mask = np.zeros((1, 10, 20, 20), np.uint16)
        mask[0, 2:-2, 4:-4, 4:10] = 5
        mask[0, 2:-2, 4:-4, 11:-4] = 7
        mask2 = image.fit_mask_to_image(mask)
        assert np.all(np.unique(mask2) == [0, 1, 2])
        assert np.all(np.unique(mask) == [0, 5, 7])
        map_arr = np.array([0, 0, 0, 0, 0, 1, 0, 2])
        assert np.all(map_arr[mask] == mask2)
        assert mask2.dtype == np.uint8

    def test_fit_mask_to_image_change_type(self):
        initial_shape = self.prepare_image_initial_shape([1, 30, 50, 50], 1)
        data = np.zeros(initial_shape, np.uint8)
        image = self.image_class(data, (1, 1, 1), "")
        mask_base = np.zeros(30 * 50 * 50, dtype=np.uint32)
        mask_base[:50] = np.arange(50, dtype=np.uint32)
        image.set_mask(np.reshape(mask_base, (1, 30, 50, 50)))
        assert image.mask.dtype == np.uint8

        mask_base[:50] = np.arange(50, dtype=np.uint32) + 5
        image.set_mask(np.reshape(mask_base, (1, 30, 50, 50)))
        assert image.mask.dtype == np.uint8

        mask_base[:350] = np.arange(350, dtype=np.uint32)
        image.set_mask(np.reshape(mask_base, (1, 30, 50, 50)))
        assert image.mask.dtype == np.uint16

        mask_base[:350] = np.arange(350, dtype=np.uint32) + 5
        image.set_mask(np.reshape(mask_base, (1, 30, 50, 50)))
        assert image.mask.dtype == np.uint16

        mask_base[: 2 ** 16 + 5] = np.arange(2 ** 16 + 5, dtype=np.uint32)
        image.set_mask(np.reshape(mask_base, (1, 30, 50, 50)))
        assert image.mask.dtype == np.uint32

        mask_base[: 2 ** 16 + 5] = np.arange(2 ** 16 + 5, dtype=np.uint32) + 5
        image.set_mask(np.reshape(mask_base, (1, 30, 50, 50)))
        assert image.mask.dtype == np.uint32

    def test_image_mask(self):
        initial_shape = self.prepare_image_initial_shape([1, 10, 50, 50], 4)
        self.image_class(np.zeros(initial_shape), (5, 5, 5), mask=np.zeros((10, 50, 50)))
        self.image_class(np.zeros(initial_shape), (5, 5, 5), mask=np.zeros((1, 10, 50, 50)))
        with pytest.raises(ValueError):
            self.image_class(np.zeros((1, 10, 50, 50, 4)), (5, 5, 5), mask=np.zeros((1, 10, 50, 40)))
        with pytest.raises(ValueError):
            self.image_class(np.zeros((1, 10, 50, 50, 4)), (5, 5, 5), mask=np.zeros((1, 10, 50, 50, 4)))
        mask = np.zeros((1, 10, 50, 50))
        mask[0, 2:-2] = 1
        im = self.image_class(np.zeros(initial_shape), (5, 5, 5), mask=mask)
        assert np.all(im.mask == mask)

    def test_reorder_axes(self):
        fixed_array = self.image_class.reorder_axes(np.zeros((10, 20)), axes="XY")
        assert fixed_array.shape == self.image_shape((10, 20), "XY")
        fixed_image = self.image_class(np.zeros((10, 20)), image_spacing=(1, 1, 1), axes_order="XY")
        assert fixed_image.shape == self.image_shape((10, 20), "XY")
        with pytest.raises(ValueError):
            Image.reorder_axes(np.zeros((10, 20)), axes="XYZ")

    def test_reorder_axes_with_mask(self):
        im = self.image_class(np.zeros((10, 50, 50, 4)), (5, 5, 5), mask=np.zeros((10, 50, 50)), axes_order="ZYXC")
        assert im.shape == self.image_shape((10, 50, 50, 4), "ZYXC")  # (1, 10, 50, 50, 4)
        im = self.image_class(np.zeros((50, 10, 50, 4)), (5, 5, 5), mask=np.zeros((50, 10, 50)), axes_order="YZXC")
        assert im.shape == self.image_shape((50, 10, 50, 4), "YZXC")  # (1, 10, 50, 50, 4)

    def test_wrong_dim_create(self):
        with pytest.raises(ValueError) as exception_info:
            self.image_class(np.zeros((10, 20)), (1, 1, 1), axes_order="XYZ")
        assert "Data should" in str(exception_info.value)

    def test_channel_pos(self):
        initial_shape = self.prepare_image_initial_shape([1, 10, 20, 20], 1)
        image = self.image_class(np.zeros(initial_shape), (1, 1, 1), "")
        assert image.channel_pos == image.axis_order.index("C")

    def test_get_dimension_number(self):
        assert (
            self.image_class(
                np.zeros((1, 10, 20, 20, 1), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
            ).get_dimension_number()
            == 3
        )
        assert (
            self.image_class(
                np.zeros((1, 1, 20, 20, 1), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
            ).get_dimension_number()
            == 2
        )
        assert (
            self.image_class(
                np.zeros((10, 1, 20, 20, 1), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
            ).get_dimension_number()
            == 3
        )
        assert (
            self.image_class(
                np.zeros((1, 1, 20, 20, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
            ).get_dimension_number()
            == 3
        )
        assert (
            self.image_class(
                np.zeros((10, 1, 20, 20, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
            ).get_dimension_number()
            == 4
        )
        assert (
            self.image_class(
                np.zeros((10, 3, 20, 20, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
            ).get_dimension_number()
            == 5
        )

    def test_get_dimension_letters(self):
        assert self.image_class(
            np.zeros((1, 10, 20, 20, 1), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
        ).get_dimension_letters() == self.reorder_axes_letter("ZYX")
        assert self.image_class(
            np.zeros((1, 1, 20, 20, 1), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
        ).get_dimension_letters() == self.reorder_axes_letter("YX")
        assert self.image_class(
            np.zeros((10, 1, 20, 20, 1), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
        ).get_dimension_letters() == self.reorder_axes_letter("TYX")
        assert self.image_class(
            np.zeros((1, 1, 20, 20, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
        ).get_dimension_letters() == self.reorder_axes_letter("YXC")
        assert self.image_class(
            np.zeros((10, 1, 20, 20, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
        ).get_dimension_letters() == self.reorder_axes_letter("TYXC")
        assert self.image_class(
            np.zeros((10, 3, 20, 20, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC"
        ).get_dimension_letters() == self.reorder_axes_letter("TZYXC")

    def test_set_mask(self):
        initial_shape = self.prepare_image_initial_shape([1, 10, 20, 30], 1)
        image = self.image_class(np.zeros(initial_shape, np.uint8), (1, 1, 1), "")
        assert image.mask is None
        assert not image.has_mask
        image.set_mask(np.ones((10, 20, 30), np.uint8))
        assert image.mask.shape == tuple(self.prepare_mask_shape((1, 10, 20, 30)))
        assert image.has_mask
        assert np.all(image.mask == 1)
        image.set_mask(np.full((10, 20, 30), 5, np.uint8))
        assert image.mask.shape == tuple(self.prepare_mask_shape((1, 10, 20, 30)))
        assert np.all(image.mask == 1)
        mask = np.full((10, 20, 30), 5, np.uint8)
        mask[0, 0, 0] = 1
        image.set_mask(mask)
        assert image.mask.shape == tuple(self.prepare_mask_shape((1, 10, 20, 30)))
        assert np.all(np.bincount(image.mask.flat) == (0, 1, 10 * 20 * 30 - 1))
        image.set_mask(None)
        assert image.mask is None

    def test_set_mask_reorder(self):
        image = self.image_class(np.zeros((1, 10, 20, 30, 1), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        image.set_mask(np.ones((30, 20, 10), np.uint8), "XYZ")
        assert image.mask.shape == self.mask_shape((1, 10, 20, 30), "TZYX")

    def test_get_image_for_save(self):
        image = self.image_class(np.zeros((1, 10, 3, 20, 30), np.uint8), (1, 1, 1), "", axes_order="TZCYX")
        assert image.get_image_for_save().shape == (1, 10, 3, 20, 30)
        image = self.image_class(np.zeros((1, 10, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        assert image.get_image_for_save().shape == (1, 10, 3, 20, 30)

    def test_get_mask_for_save(self):
        image = self.image_class(np.zeros((1, 10, 3, 20, 30), np.uint8), (1, 1, 1), "", axes_order="TZCYX")
        assert image.get_mask_for_save() is None
        image.set_mask(np.zeros((1, 10, 20, 30), np.uint8), axes="TZYX")
        assert image.get_mask_for_save().shape == (1, 10, 1, 20, 30)

    def test_image_properties(self):
        image = self.image_class(np.zeros((1, 10, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        assert not image.has_mask
        assert not image.is_time
        assert image.is_stack
        assert not image.is_2d
        assert image.channels == 3
        assert image.layers == 10
        assert image.times == 1
        assert image.plane_shape == (20, 30)

    def test_swap_time_and_stack(self):
        image = self.image_class(np.zeros((1, 10, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        image2 = image.swap_time_and_stack()
        assert image.times == 1
        assert image.layers == 10
        assert image2.times == 10
        assert image2.layers == 1

    def test_get_channel(self):
        image = self.image_class(np.zeros((1, 10, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        channel = image.get_channel(1)
        assert channel.shape == self.mask_shape((1, 10, 20, 30), "TZYX")
        with pytest.raises(IndexError):
            image.get_channel(4)

    def test_get_layer(self):
        image = self.image_class(np.zeros((1, 10, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        layer = image.get_layer(0, 5)
        assert layer.shape == self.needed_shape((20, 30, 3), "YXC", "TZ")

    def test_spacing(self):
        image = self.image_class(np.zeros((1, 10, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        assert image.spacing == (1, 1, 1)
        assert image.voxel_size == (1, 1, 1)
        image.set_spacing((1, 2, 3))
        assert image.spacing == (1, 2, 3)
        assert image.voxel_size == (1, 2, 3)
        with pytest.raises(ValueError):
            image.set_spacing((1, 2, 3, 4))
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            image.set_spacing(1)
        image.set_spacing((1, 0, 4))
        assert image.spacing == (1, 2, 3)
        assert image.voxel_size == (1, 2, 3)

    def test_spacing_2d(self):
        image = self.image_class(np.zeros((1, 1, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        assert image.spacing == (1, 1)
        assert image.voxel_size == (1, 1)
        image.set_spacing((1, 2))
        assert image.spacing == (1, 2)
        assert image.voxel_size == (1, 2)

    def test_cut_image(self):
        # TODO add tests for more irregular shape
        image = self.image_class(np.zeros((1, 10, 20, 30, 3), np.uint8), (1, 1, 1), "", axes_order="TZYXC")
        mask1 = np.zeros((1, 10, 20, 30), np.uint8)
        mask1[0, 2:-2, 2:9, 2:-2] = 1
        mask1[0, 2:-2, 11:-2, 2:-2] = 2
        image.set_mask(mask1, "TZYX")
        mask = image.mask
        image.set_mask(np.zeros((1, 10, 20, 30), np.uint8), "TZYX")
        im = image.cut_image(mask == 1, replace_mask=False)
        assert np.all(im.mask == 0)
        assert im.shape == self.image_shape((1, 10, 11, 30, 3), axes="TZYXC")
        im = image.cut_image(mask == 2, replace_mask=True)
        assert np.all(
            im.mask[
                :, FRAME_THICKNESS:-FRAME_THICKNESS, FRAME_THICKNESS:-FRAME_THICKNESS, FRAME_THICKNESS:-FRAME_THICKNESS
            ]
            == 1
        )
        assert im.shape == self.image_shape((1, 10, 11, 30, 3), axes="TZYXC")

        # Test cutting with list of slices
        points = np.nonzero(mask == 2)
        lower_bound = np.min(points, axis=1)
        upper_bound = np.max(points, axis=1)
        cut_list = [slice(x, y + 1) for x, y in zip(lower_bound, upper_bound)]
        res = image.cut_image(cut_list)
        shape = [y - x + +1 for x, y in zip(lower_bound, upper_bound)]
        shape.insert(image.channel_pos, 3)
        shape[image.x_pos] += 2 * FRAME_THICKNESS
        shape[image.y_pos] += 2 * FRAME_THICKNESS
        shape[image.stack_pos] += 2 * FRAME_THICKNESS
        assert res.shape == tuple(shape)

    def test_get_ranges(self):
        data = np.zeros((1, 10, 20, 30, 3), np.uint8)
        data[..., :10, 0] = 2
        data[..., :10, 1] = 20
        data[..., :10, 2] = 9
        image = self.image_class(data, (1, 1, 1), "", axes_order="TZYXC")
        assert len(image.get_ranges()) == 3
        assert image.get_ranges() == [(0, 2), (0, 20), (0, 9)]

    def test_get_um_spacing(self):
        image = self.image_class(
            np.zeros((1, 10, 20, 30, 3), np.uint8), (10 ** -6, 10 ** -6, 10 ** -6), "", axes_order="TZYXC"
        )
        assert image.get_um_spacing() == (1, 1, 1)
        image = self.image_class(
            np.zeros((1, 1, 20, 30, 3), np.uint8), (10 ** -6, 10 ** -6, 10 ** -6), "", axes_order="TZYXC"
        )
        assert image.get_um_spacing() == (1, 1)

    def test_save(self, tmp_path):
        data = np.zeros((1, 10, 20, 30, 3), np.uint8)
        data[..., :10, 0] = 2
        data[..., :10, 1] = 20
        data[..., :10, 2] = 9
        image = self.image_class(data, (10 ** -6, 10 ** -6, 10 ** -6), "", axes_order="TZYXC")
        mask = np.zeros((10, 20, 30), np.uint8)
        mask[..., 2:12] = 1
        image.set_mask(mask, "ZYX")
        ImageWriter.save(image, os.path.join(tmp_path, "img.tif"))
        ImageWriter.save_mask(image, os.path.join(tmp_path, "img_mask.tif"))
        read_image: Image = TiffImageReader.read_image(
            os.path.join(tmp_path, "img.tif"), os.path.join(tmp_path, "img_mask.tif")
        )
        assert read_image.get_um_spacing() == (1, 1, 1)
        assert len(read_image.get_ranges()) == 3
        assert read_image.get_ranges() == [(0, 2), (0, 20), (0, 9)]

    def test_axes_pos(self):
        data = np.zeros((10, 10), np.uint8)
        image = self.image_class(data, (1, 1), axes_order="XY")
        assert image.x_pos == image.axis_order.index("X")
        assert image.y_pos == image.axis_order.index("Y")
        assert image.time_pos == image.axis_order.index("T")
        assert image.stack_pos == image.axis_order.index("Z")
        assert image.channel_pos == image.axis_order.index("C")


class ChangeChannelPosImage(Image):
    axis_order = "TZCYX"


class TestInheritanceImageChannelPos(TestImageBase):
    image_class = ChangeChannelPosImage


class ChangeTimePosImage(Image):
    axis_order = "ZTYXC"


class TestInheritanceImageTimePos(TestImageBase):
    image_class = ChangeTimePosImage


class WeirdOrderImage(Image):
    axis_order = "XYZTC"


class TestInheritanceWeirdOrderImage(TestImageBase):
    image_class = WeirdOrderImage


class AdditionalAxesImage(Image):
    axis_order = "VTZYXC"

    def get_image_for_save(self, index=0) -> np.ndarray:
        return super().get_image_for_save()[..., index]


class TestInheritanceAdditionalAxesImage(TestImageBase):
    image_class = AdditionalAxesImage


class TestMergeImage:
    @pytest.mark.parametrize("chanel_mark", ["C", Image.axis_order.index("C")])
    @pytest.mark.parametrize("check_dtype", [np.uint8, np.uint16, np.uint32, np.float16, np.float32, np.float64])
    def test_merge_chanel(self, chanel_mark, check_dtype):
        image1 = Image(data=np.zeros((3, 10, 10), dtype=np.uint8), axes_order="ZXY", image_spacing=(1, 1, 1))
        image2 = Image(data=np.ones((3, 10, 10), dtype=check_dtype), axes_order="ZXY", image_spacing=(1, 1, 1))
        res_image = image1.merge(image2, chanel_mark)
        assert res_image.channels == 2
        assert np.all(res_image.get_channel(0) == 0)
        assert np.all(res_image.get_channel(1) == 1)
        assert res_image.dtype == check_dtype

    def test_merge_fail(self):
        image1 = Image(data=np.zeros((4, 10, 10), dtype=np.uint8), axes_order="ZXY", image_spacing=(1, 1, 1))
        image2 = Image(data=np.zeros((3, 10, 10), dtype=np.uint8), axes_order="ZXY", image_spacing=(1, 1, 1))
        with pytest.raises(ValueError):
            image1.merge(image2, "C")

    def test_different_axes_order(self):
        image1 = Image(data=np.zeros((3, 10, 10), dtype=np.uint8), axes_order="ZXY", image_spacing=(1, 1, 1))
        image2 = ChangeChannelPosImage(
            data=np.zeros((3, 10, 10), dtype=np.uint8), axes_order="ZXY", image_spacing=(1, 1, 1)
        )
        res_image = image1.merge(image2, "C")
        assert res_image.channels == 2
        assert isinstance(res_image, Image)
        assert isinstance(image2.merge(image1, "C"), ChangeChannelPosImage)
