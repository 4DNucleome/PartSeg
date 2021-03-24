import re
import typing
import warnings
from collections.abc import Iterable
from itertools import zip_longest

import numpy as np

Spacing = typing.Tuple[typing.Union[float, int], ...]

_DEF = object()
FRAME_THICKNESS = 2

NAPARI_SCALING = 10 ** 9


def minimal_dtype(val: int):
    """
    Calculate minimal type to handle value in array

    :param val:
    :return: minimal dtype to handle given value
    :rtype:
    """
    if val < 250:
        return np.uint8
    if val < 2 ** 16 - 5:
        return np.uint16
    return np.uint32


def reduce_array(
    array: np.ndarray,
    components: typing.Optional[typing.Iterable[int]] = None,
    max_val: typing.Optional[int] = None,
    dtype=None,
) -> np.ndarray:
    """
    Relabel components from 1 to components_num with keeping order

    :param array: array to relabel, deed to be integer type
    :param components: components to be keep, if None then all will be keep
    :param max_val: number of maximum component in array, if absent then will be calculated
        (to reduce whole array processing)
    :param dtype: type of returned array if no then minimal type is calculated
    :return: relabeled array in minimum type
    """
    # this function minimal dtype is np.uint8 so there is no need to do calculation.
    if components is None:
        components = np.unique(array.flat)
        if max_val is None:
            max_val = np.max(components)

    if max_val is None:
        max_val = np.max(array)

    translate = np.zeros(max_val + 1, dtype=dtype or minimal_dtype(len(components) + 1))

    for i, val in enumerate(sorted(components), start=0 if 0 in components else 1):
        translate[val] = i

    return translate[array]


class Image:
    """
    Base class for Images used in PartSeg

    :param data: 5-dim array with order: time, z, y, x, channel
    :param image_spacing: spacing for z, y, x
    :param file_path: path to image on disc
    :param mask: mask array in shape z,y,x
    :param default_coloring: default colormap - not used yet
    :param ranges: default ranges for channels
    :param channel_names: labels for channels
    :param axes_order: allow to create Image object form data with different axes order, or missed axes

    :cvar str ~.axis_order: internal order of axes

    It is prepared for subclassing with changed internal order. Eg:

    >>> class ImageJImage(Image):
    >>>     axis_order = "TZCYX"

    """

    _image_spacing: Spacing
    axis_order = "TZYXC"

    def __new__(cls, *args, **kwargs):
        if hasattr(cls, "return_order"):  # pragma: no cover
            warnings.warn("Using return_order is deprecated since PartSeg 0.11.0", DeprecationWarning)
            cls.axis_order = cls.return_order
        return super().__new__(cls)

    def __init__(
        self,
        data: np.ndarray,
        image_spacing: Spacing,
        file_path=None,
        mask: typing.Union[None, np.ndarray] = None,
        default_coloring=None,
        ranges=None,
        channel_names=None,
        axes_order: typing.Optional[str] = None,
    ):
        # TODO add time distance to image spacing
        if axes_order is None:
            axes_order = self.axis_order
        if data.ndim != len(axes_order):
            raise ValueError(
                "Data should have same number of dimensions " f"like length of axes_order (current :{len(axes_order)}"
            )
        if not isinstance(image_spacing, tuple):
            image_spacing = tuple(image_spacing)
        self._image_array = self.reorder_axes(data, axes_order)
        self._image_spacing = (1.0,) * (3 - len(image_spacing)) + image_spacing
        self._image_spacing = tuple(el if el > 0 else 10 ** -6 for el in self._image_spacing)

        self.file_path = file_path
        self.default_coloring = default_coloring
        if self.default_coloring is not None:
            self.default_coloring = [np.array(x) for x in default_coloring]
        default_channel_names = [f"channel {i+1}" for i in range(self.channels)]
        if isinstance(channel_names, Iterable):
            self._channel_names = [
                x if x is not None else y for x, y in zip_longest(channel_names, default_channel_names, fillvalue=None)
            ]
        else:
            self._channel_names = default_channel_names
        self._channel_names = self._channel_names[: self.channels]
        if ranges is None:
            axis = list(range(len(self.axis_order)))
            axis.remove(self.axis_order.index("C"))
            axis = tuple(axis)
            self.ranges = list(zip(np.min(self._image_array, axis=axis), np.max(self._image_array, axis=axis)))
        else:
            self.ranges = ranges
        if mask is not None:
            data_shape = list(data.shape)
            try:
                data_shape.pop(axes_order.index("C"))
            except ValueError:  # pragma: no cover
                pass
            mask = self._fit_array_to_image(data_shape, mask)
            mask = np.take(self.reorder_axes(mask, axes_order.replace("C", "")), 0, self.channel_pos)
            self._mask_array = self.fit_mask_to_image(mask)
        else:
            self._mask_array = None

    def merge(self, image: "Image", axis: typing.Union[str, int]) -> "Image":
        """
        Produce new image merging image data along given axis. All metadata
        are obtained from self.

        :param Image image: Image to be merged
        :param typing.Union[str, int] axis:
        :return: New image produced from merge
        :rtype: Image
        """
        if isinstance(axis, str):
            axis = self.axis_order.index(axis)
        data = self.reorder_axes(image.get_data(), image.axis_order)
        data = np.concatenate((self.get_data(), data), axis=axis)
        channel_names = self.channel_names
        reg = re.compile(r"channel \d+")
        for name in image.channel_names:
            match = reg.match(name)
            new_name = name
            if match:
                name = "channel"
                new_name = f"channel {len(channel_names) + 1}"
            i = 1
            while new_name in channel_names:
                new_name = f"{name} ({i})"
                i += 1
                if i > 10000:  # pragma: no cover
                    raise ValueError("fail when try to fix channel name")
            channel_names.append(new_name)

        return self.substitute(data=data, ranges=self.ranges + image.ranges, channel_names=channel_names)

    @property
    def channel_names(self) -> typing.List[str]:
        return self._channel_names[:]

    @property
    def channel_pos(self) -> int:
        """Channel axis. Need to have 'C' in :py:attr:`axis_order`"""
        return self.axis_order.index("C")

    @property
    def x_pos(self):
        return self.axis_order.index("X")

    @property
    def y_pos(self):
        return self.axis_order.index("Y")

    @property
    def time_pos(self):
        """Time axis. Need to have 'T' in :py:attr:`axis_order`"""
        return self.axis_order.index("T")

    @property
    def stack_pos(self) -> int:
        """Stack axis. Need to have 'Z' in :py:attr:`axis_order`"""
        return self.axis_order.index("Z")

    @property
    def array_axis_order(self):
        return self.axis_order.replace("C", "")

    @property
    def dtype(self) -> np.dtype:
        """dtype of image array"""
        return self._image_array.dtype

    @staticmethod
    def _reorder_axes(array: np.ndarray, input_axes: str, return_axes) -> np.ndarray:
        if array.ndim != len(input_axes):
            raise ValueError(f"array.ndim ({array.ndim}) need to be equal to length of axes ('{input_axes}')")
        mapping_dict = {v: i for i, v in enumerate(return_axes)}
        if array.ndim < len(return_axes):
            array = array.reshape(array.shape + (1,) * (len(return_axes) - array.ndim))
        new_positions = [mapping_dict[x] for x in input_axes if x in mapping_dict]
        axes_to_map = [i for i, x in enumerate(input_axes) if x in mapping_dict]
        return np.moveaxis(array, axes_to_map, new_positions)

    @classmethod
    def reorder_axes(cls, array: np.ndarray, axes: str) -> np.ndarray:
        """
        reorder axes to internal storage format

        :param np.ndarray array: array to have changed order of axes
        :param str axes: axes order
        :return: array with correct order of axes
        """
        return cls._reorder_axes(array, axes, cls.axis_order)

    def get_dimension_number(self) -> int:
        """return number of nontrivial dimensions"""
        return np.squeeze(self._image_array).ndim

    def get_dimension_letters(self) -> str:
        """
        :return: letters which indicates non trivial dimensions
        """
        return "".join(key for val, key in zip(self._image_array.shape, self.axis_order) if val > 1)

    def substitute(
        self,
        data=None,
        image_spacing=None,
        file_path=None,
        mask=_DEF,
        default_coloring=None,
        ranges=None,
        channel_names=None,
    ):
        """Create copy of image with substitution of not None elements"""
        data = self._image_array if data is None else data
        image_spacing = self._image_spacing if image_spacing is None else image_spacing
        file_path = self.file_path if file_path is None else file_path
        mask = self._mask_array if mask is _DEF else mask
        default_coloring = self.default_coloring if default_coloring is None else default_coloring
        ranges = self.ranges if ranges is None else ranges
        channel_names = self.channel_names if channel_names is None else channel_names
        return self.__class__(
            data=data,
            image_spacing=image_spacing,
            file_path=file_path,
            mask=mask,
            default_coloring=default_coloring,
            ranges=ranges,
            channel_names=channel_names,
        )

    def set_mask(self, mask: typing.Optional[np.ndarray], axes: typing.Optional[str] = None):
        """
        Set mask for image, check if it has proper shape.

        :param mask: mask in same shape like image. May not contains 1 dim axes.
        :param axes: order of axes in mask, use if different than :py:attr:`return_order`
        :raise ValueError: on wrong shape
        """
        if mask is None:
            self._mask_array = None
        elif axes is not None:
            self._mask_array = self.fit_mask_to_image(np.take(self.reorder_axes(mask, axes), 0, self.channel_pos))
        else:
            self._mask_array = self.fit_mask_to_image(mask)

    def get_data(self) -> np.ndarray:
        return self._image_array[:]

    @property
    def mask(self) -> typing.Optional[np.ndarray]:
        return self._mask_array[:] if self.has_mask else None

    @staticmethod
    def _fit_array_to_image(base_shape, array: np.ndarray) -> np.ndarray:
        """change shape of array with inserting singe dimensional entries"""
        shape = list(array.shape)
        for i, el in enumerate(base_shape):
            if el == 1 and el != shape[i]:
                shape.insert(i, 1)
            elif el != shape[i]:
                raise ValueError(f"Wrong array shape {shape} for {base_shape}")
        if len(shape) != len(base_shape):
            raise ValueError(f"Wrong array shape {shape} for {base_shape}")
        return np.reshape(array, shape)

    def fit_array_to_image(self, array: np.ndarray) -> np.ndarray:
        """
        Change shape of array with inserting singe dimensional entries

        :param np.ndarray array: array to be fitted

        :return: reshaped array witha added missing 1 in shape

        :raises ValueError: if cannot fit array
        """
        base_shape = list(self._image_array.shape)
        base_shape.pop(self.channel_pos)
        return self._fit_array_to_image(base_shape, array)

    # noinspection DuplicatedCode
    def fit_mask_to_image(self, array: np.ndarray) -> np.ndarray:
        """
        call :py:meth:`fit_array_to_image` and
        then relabel and change type to minimal
        which fit all information
        """
        array = self.fit_array_to_image(array)
        unique = np.unique(array)
        if unique.size == 2 and unique[1] == 1:
            return array.astype(np.uint8)
        if unique.size == 1:
            if unique[0] != 0:
                return np.ones(array.shape, dtype=np.uint8)
            return array.astype(np.uint8)
        max_val = np.max(unique)
        return reduce_array(array, unique, max_val)

    def get_image_for_save(self) -> np.ndarray:
        """
        :return: numpy array in imagej tiff order axes
        """
        return self._reorder_axes(self._image_array, self.axis_order, "TZCYX")

    def get_mask_for_save(self) -> typing.Optional[np.ndarray]:
        """
        :return: if image has mask then return mask with axes in proper order
        """
        if not self.has_mask:
            return None
        axes_order = list(self.axis_order)
        axes_order.pop(self.channel_pos)
        return self._reorder_axes(self._mask_array, "".join(axes_order), "TZCYX")

    @property
    def has_mask(self) -> bool:
        """check if image is masked"""
        return self._mask_array is not None

    @property
    def is_time(self) -> bool:
        """check if image contains time data"""
        return self.times > 1

    @property
    def is_stack(self) -> bool:
        """check if image contain 3d data"""
        return self.layers > 1

    @property
    def channels(self) -> int:
        """number of image channels"""
        return self._image_array.shape[self.channel_pos]

    @property
    def layers(self) -> int:
        """z-dim of image"""
        return self._image_array.shape[self.stack_pos]

    @property
    def times(self) -> int:
        """number of time frames"""
        return self._image_array.shape[self.time_pos]

    @property
    def plane_shape(self) -> (int, int):
        """y,x size of image"""
        x_index = self.axis_order.index("X")
        y_index = self.axis_order.index("Y")
        return self._image_array.shape[y_index], self._image_array.shape[x_index]

    @property
    def shape(self):
        """Whole image shape. order of axes my change. Current order is in :py:attr:`return_order`"""
        return self._image_array.shape

    def swap_time_and_stack(self):
        """
        Swap time and stack axes.
        For example my be used to convert time image in 3d image.
        """
        image_array = np.swapaxes(self._image_array, self.time_pos, self.stack_pos)
        return self.substitute(data=image_array)

    def get_axis_positions(self) -> typing.Dict[str, int]:
        """
        :return: dict with mapping axis to its position
        :rtype: dict
        """
        return {l: i for i, l in enumerate(self.axis_order)}

    def get_array_axis_positions(self) -> typing.Dict[str, int]:
        """
        :return: dict with mapping axis to its position for array fitted to image
        :rtype: dict
        """
        return {l: i for i, l in enumerate(self.axis_order.replace("C", ""))}

    def get_data_by_axis(self, **kwargs) -> np.ndarray:
        """
        Get part of data extracted by sub axis. Axis is selected by single letter from :py:attr:`axis_order`

        :param kwargs: axis list with
        :return:
        :rtype:
        """
        slices: typing.List[typing.Union[int, slice]] = [slice(None) for _ in range(len(self.axis_order))]
        axis_pos = self.get_axis_positions()
        for name in kwargs:
            if name.upper() in axis_pos:
                slices[axis_pos[name.upper()]] = kwargs[name]
        return self._image_array[tuple(slices)]

    def clip_array(self, array, **kwargs):
        array = self.fit_array_to_image(array)
        slices: typing.List[typing.Union[int, slice]] = [slice(None) for _ in range(len(self.array_axis_order))]
        axis_pos = self.get_array_axis_positions()
        for name in kwargs:
            if name.upper() in axis_pos:
                slices[axis_pos[name.upper()]] = kwargs[name]
        return array[tuple(slices)]

    def get_channel(self, num) -> np.ndarray:
        """
        Alias for :py:func:`get_sub_data` with argument ``c=num``

        :param int num: channel num to be extracted
        :return: given channel array
        :rtype: numpy.ndarray
        """
        return self.get_data_by_axis(c=num)

    def get_layer(self, time: int, stack: int) -> np.ndarray:
        """
        return single layer contains data for all channel

        :param time: time coordinate. For images with not time use 0.
        :param stack: "z coordinate. For time data use 0.
        :return:
        """
        elem_num = max(self.stack_pos, self.time_pos) + 1
        indices: typing.List[typing.Union[int, slice]] = [0] * elem_num
        for i in range(elem_num):
            if i == self.time_pos:
                indices[i] = time
            elif i == self.stack_pos:
                indices[i] = stack
            else:
                indices[i] = slice(None)
        return self._image_array[tuple(indices)]

    @property
    def is_2d(self) -> bool:
        """
        Check if image z and time dimension are equal to 1.
        Equivalent to:
        `image.layers == 1 and image.times == 1`
        """
        return self.layers == 1 and self.times == 1

    @property
    def spacing(self) -> Spacing:
        """image spacing"""
        if self.is_2d:
            return tuple(self._image_spacing[1:])
        return self._image_spacing

    def normalized_scaling(self, factor=NAPARI_SCALING) -> Spacing:
        if self.is_2d:
            return (1, 1) + tuple(np.multiply(self.spacing, factor))
        return (1,) + tuple(np.multiply(self.spacing, factor))

    @property
    def voxel_size(self) -> Spacing:
        """alias for spacing"""
        return self.spacing

    def set_spacing(self, value: Spacing):
        """set image spacing"""
        if 0 in value:
            return
        if self.is_2d and len(value) + 1 == len(self._image_spacing):
            value = (1.0,) + tuple(value)
        if len(value) != len(self._image_spacing):  # pragma: no cover
            raise ValueError("Correction of spacing fail.")
        self._image_spacing = tuple(value)

    @staticmethod
    def _frame_array(array: typing.Optional[np.ndarray], index_to_add: typing.List[int]):
        if array is None:  # pragma: no cover
            return array
        result_shape = list(array.shape)
        image_pos = [slice(None) for _ in range(array.ndim)]

        for index in index_to_add:
            result_shape[index] += FRAME_THICKNESS * 2
            image_pos[index] = slice(FRAME_THICKNESS, result_shape[index] - FRAME_THICKNESS)

        data = np.zeros(shape=result_shape, dtype=array.dtype)
        data[tuple(image_pos)] = array
        return data

    @staticmethod
    def calc_index_to_frame(array_axis, important_axis):
        """
        calculate in which axis frame should be added

        :param str array_axis: list of image axis
        :param str important_axis: list of framed axis
        :return:
        """
        return [array_axis.index(letter) for letter in important_axis]

    def cut_image(
        self, cut_area: typing.Union[np.ndarray, typing.List[slice], typing.Tuple[slice]], replace_mask=False
    ) -> "Image":
        """
        Create new image base on mask or list of slices
        :param replace_mask: if cut area is represented by mask array,
        then in result image the mask is set base on cut_area
        :param cut_area: area to cut. Defined with slices or mask
        :return: Image
        """
        new_mask = None
        if isinstance(cut_area, (list, tuple)):
            cut_area2 = cut_area[:]
            cut_area2.insert(self.channel_pos, slice(None))
            new_image = self._image_array[tuple(cut_area2)]
            if self._mask_array is not None:
                new_mask = self._mask_array[tuple(cut_area)]
        else:
            cut_area = self.fit_array_to_image(cut_area)
            points = np.nonzero(cut_area)
            lower_bound = np.min(points, axis=1)
            upper_bound = np.max(points, axis=1)
            new_cut = [slice(x, y + 1) for x, y in zip(lower_bound, upper_bound)]
            catted_cut_area = cut_area[tuple(new_cut)]
            image_cut = new_cut[:]
            image_cut.insert(self.channel_pos, slice(None))
            new_image = np.copy(self._image_array[tuple(image_cut)])
            if self.channel_pos == len(self.axis_order) - 1:
                new_image[catted_cut_area == 0] = 0
            else:
                for i in range(self.channels):
                    np.take(new_image, i, self.channel_pos)[catted_cut_area == 0] = 0
            if replace_mask:
                new_mask = catted_cut_area
            elif self._mask_array is not None:
                new_mask = self._mask_array[tuple(new_cut)]
                new_mask[catted_cut_area == 0] = 0
        important_axis = "XY" if self.is_2d else "XYZ"

        return self.__class__(
            self._frame_array(new_image, self.calc_index_to_frame(self.axis_order, important_axis)),
            self._image_spacing,
            None,
            self._frame_array(new_mask, self.calc_index_to_frame(self.array_axis_order, important_axis)),
            self.default_coloring,
            self.ranges,
            self.channel_names,
        )

    def get_imagej_colors(self):
        # TODO review
        if self.default_coloring is None:
            return None
        try:
            if len(self.default_coloring) != self.channels:
                return None
        except TypeError:
            return None
        res = []
        for color in self.default_coloring:
            if color.ndim == 1:
                res.append(np.array([np.linspace(0, x, num=256) for x in color]))
            else:
                if color.shape[1] != 256:
                    res.append(
                        np.array(
                            [
                                np.interp(np.linspace(0, 255, num=256), np.linspace(0, color.shape[1], num=256), x)
                                for x in color
                            ]
                        )
                    )
                res.append(color)
        return res

    def get_colors(self):
        # TODO review
        if self.default_coloring is None:
            return None
        res = []
        for color in self.default_coloring:
            if color.ndim == 2:
                res.append(list(color[:, -1]))
            else:
                res.append(list(color))
        return res

    def get_um_spacing(self) -> Spacing:
        """image spacing in micrometers"""
        return tuple(float(x * 10 ** 6) for x in self.spacing)

    def get_ranges(self) -> typing.List[typing.Tuple[float, float]]:
        """image brightness ranges for each channel"""
        return self.ranges[:]

    def __str__(self):
        return (
            f"{self.__class__} Shape {self._image_array.shape}, dtype: {self._image_array.dtype}, "
            f"labels: {self.channel_names}, coloring: {self.get_colors()} mask: {self.has_mask}"
        )

    def __repr__(self):
        mask_info = f"mask=True, mask_dtype={self._mask_array.dtype}" if self.mask is not None else "mask=False"
        return (
            f"Image(shape={self._image_array.shape} dtype={self._image_array.dtype}, spacing={self.spacing}, "
            f"labels={self.channel_names}, channels={self.channels}, axes={repr(self.axis_order)}, {mask_info})"
        )
