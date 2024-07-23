from __future__ import annotations

import re
import typing
import warnings
from collections.abc import Iterable
from contextlib import suppress

import numpy as np

from PartSegImage import Channel

Spacing = typing.Tuple[typing.Union[float, int], ...]
_IMAGE_DATA = typing.Union[typing.List[np.ndarray], np.ndarray]

_DEF = object()
FRAME_THICKNESS = 2

DEFAULT_SCALE_FACTOR = 10**9


def minimal_dtype(val: int):
    """
    Calculate minimal type to handle value in array

    :param val:
    :return: minimal dtype to handle given value
    :rtype:
    """
    if val < 250:
        return np.uint8
    return np.uint16 if val < 2**16 - 5 else np.uint32


def reduce_array(
    array: np.ndarray,
    components: typing.Collection[int] | None = None,
    max_val: int | None = None,
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
    axis_order = "CTZYX"
    array_axis_order: str

    def __new__(cls, *args, **kwargs):
        if hasattr(cls, "return_order"):  # pragma: no cover
            warnings.warn("Using return_order is deprecated since PartSeg 0.11.0", DeprecationWarning, stacklevel=2)
            cls.axis_order = cls.return_order
        cls.array_axis_order = cls.axis_order.replace("C", "")
        return super().__new__(cls)

    def __init__(
        self,
        data: _IMAGE_DATA,
        image_spacing: Spacing,
        file_path=None,
        mask: None | np.ndarray = None,
        default_coloring=None,
        ranges=None,
        channel_names=None,
        axes_order: str | None = None,
        shift: Spacing | None = None,
        name: str = "",
        metadata: dict | None = None,
    ):
        # TODO add time distance to image spacing
        if axes_order is None:  # pragma: no cover
            warnings.warn(
                f"axes_order should be provided, Currently it uses {self.__class__}.axis_order",
                category=DeprecationWarning,
                stacklevel=2,
            )
            axes_order = self.axis_order
        self._check_data_dimensionality(data, axes_order)
        if not isinstance(image_spacing, tuple):
            image_spacing = tuple(image_spacing)
        self._channel_arrays = self._split_data_on_channels(data, axes_order)
        self._image_spacing = (1.0,) * (3 - len(image_spacing)) + image_spacing
        self._image_spacing = tuple(el if el > 0 else 10**-6 for el in self._image_spacing)

        self._shift = tuple(shift) if shift is not None else (0,) * len(self._image_spacing)
        self.name = name

        self.file_path = file_path
        self.default_coloring = default_coloring
        if self.default_coloring is not None:
            self.default_coloring = [np.array(x) for x in default_coloring]

        self._channel_names = self._prepare_channel_names(channel_names, self.channels)

        self.ranges = self._adjust_ranges(ranges, self._channel_arrays)
        self._mask_array = self._fit_mask(mask, data, axes_order)
        self.metadata = dict(metadata) if metadata is not None else {}

    @staticmethod
    def _check_data_dimensionality(data, axes_order):
        if (isinstance(data, list) and any(x.ndim + 1 != len(axes_order) for x in data)) or (
            not isinstance(data, list) and data.ndim != len(axes_order)
        ):
            if isinstance(data, list):
                ndim = ", ".join([f"{x.ndim} + 1" for x in data])
            else:
                ndim = str(data.ndim)
            raise ValueError(
                "Data should have same number of dimensions "
                f"like length of axes_order (axis :{len(axes_order)}, ndim: {ndim}"
            )

    @staticmethod
    def _adjust_ranges(
        ranges: list[tuple[float, float]] | None, channel_arrays: list[np.ndarray]
    ) -> list[tuple[float, float]]:
        if ranges is None:
            ranges = list(zip((np.min(c) for c in channel_arrays), (np.max(c) for c in channel_arrays)))
        return [(min_val, max_val) if (min_val != max_val) else (min_val, min_val + 1) for (min_val, max_val) in ranges]

    def _fit_mask(self, mask, data, axes_order):
        mask_array = self._prepare_mask(mask, data, axes_order)
        if mask_array is not None:
            mask_array = self.fit_mask_to_image(mask_array)
        return mask_array

    @classmethod
    def _prepare_mask(cls, mask, data, axes_order) -> np.ndarray | None:
        if mask is None:
            return None

        if isinstance(data, list):
            data_shape = list(data[0].shape)
        else:
            data_shape = list(data.shape)
            with suppress(ValueError):
                data_shape.pop(axes_order.index("C"))

        mask = cls._fit_array_to_image(data_shape, mask)
        return cls.reorder_axes(mask, axes_order.replace("C", ""))

    @staticmethod
    def _prepare_channel_names(channel_names, channels_num) -> list[str]:
        default_channel_names = [f"channel {i + 1}" for i in range(channels_num)]
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        if isinstance(channel_names, Iterable):
            channel_names_list = [str(x) for x in channel_names]
            channel_names_list = channel_names_list[:channels_num] + default_channel_names[len(channel_names_list) :]
        else:
            channel_names_list = default_channel_names
        return channel_names_list[:channels_num]

    @classmethod
    def _split_data_on_channels(cls, data: np.ndarray | list[np.ndarray], axes_order: str) -> list[np.ndarray]:
        if isinstance(data, list) and not axes_order.startswith("C"):  # pragma: no cover
            raise ValueError("When passing data as list of numpy arrays then Channel must be first axis.")
        if "C" not in axes_order:
            if not isinstance(data, np.ndarray):  # pragma: no cover
                raise TypeError("If `axes_order` does not contain `C` then data must be numpy array.")
            return [cls.reorder_axes(data, axes_order)]
        if axes_order.startswith("C"):
            if isinstance(data, list):
                dtype = np.result_type(*data)
                return [cls.reorder_axes(x, axes_order[1:]).astype(dtype) for x in data]
            return [cls.reorder_axes(x, axes_order[1:]) for x in data]

        if not isinstance(data, np.ndarray):
            raise TypeError("If `data` is list of arrays then `axes_order` must start with `C`")  # pragma: no cover
        pos: list[slice | int] = [slice(None) for _ in range(data.ndim)]
        c_pos = axes_order.index("C")
        res = []
        for i in range(data.shape[c_pos]):
            pos[c_pos] = i
            res.append(cls.reorder_axes(data[tuple(pos)], axes_order.replace("C", "")))
        return res

    @staticmethod
    def _merge_channel_names(base_channel_names: list[str], new_channel_names: list[str]) -> list[str]:
        base_channel_names = base_channel_names[:]
        reg = re.compile(r"channel \d+")
        for name in new_channel_names:
            match = reg.match(name)
            new_name = name
            base_name = name
            if match and base_name in base_channel_names:
                new_name = f"channel {len(base_channel_names) + 1}"
            i = 1
            while new_name in base_channel_names:
                new_name = f"{base_name} ({i})"
                i += 1
                if i > 10000:  # pragma: no cover
                    raise ValueError("fail when try to fix channel name")
            base_channel_names.append(new_name)
        return base_channel_names

    def merge(self, image: Image, axis: str) -> Image:
        """
        Produce new image merging image data along given axis. All metadata
        are obtained from self.

        :param Image image: Image to be merged
        :param str axis:
        :return: New image produced from merge
        :rtype: Image
        """
        if axis == "C":
            data = self._image_data_normalize(
                self._channel_arrays + [self.reorder_axes(x, image.array_axis_order) for x in image._channel_arrays]
            )
            channel_names = self._merge_channel_names(self.channel_names, image.channel_names)
        else:
            index = self.array_axis_order.index(axis)
            data = self._image_data_normalize(
                [
                    np.concatenate((y, self.reorder_axes(y, image.array_axis_order)), axis=index)
                    for x, y in zip(self._channel_arrays, image._channel_arrays)
                ]
            )
            channel_names = self.channel_names

        return self.substitute(data=data, ranges=self.ranges + image.ranges, channel_names=channel_names)

    @property
    def channel_names(self) -> list[str]:
        return self._channel_names[:]

    @property
    def channel_pos(self) -> int:  # pragma: no cover
        """Channel axis. Need to have 'C' in :py:attr:`axis_order`"""
        warnings.warn(
            "channel_pos is deprecated and code its using may not work properly", category=FutureWarning, stacklevel=2
        )
        return self.axis_order.index("C")

    @property
    def x_pos(self):
        return self.array_axis_order.index("X")

    @property
    def y_pos(self):
        return self.array_axis_order.index("Y")

    @property
    def time_pos(self):
        """Time axis. Need to have 'T' in :py:attr:`axis_order`"""
        return self.array_axis_order.index("T")

    @property
    def stack_pos(self) -> int:
        """Stack axis. Need to have 'Z' in :py:attr:`axis_order`"""
        return self.array_axis_order.index("Z")

    @property
    def dtype(self) -> np.dtype:
        """dtype of image array"""
        return self._channel_arrays[0].dtype

    @staticmethod
    def _reorder_axes(array: np.ndarray, input_axes: str, return_axes) -> np.ndarray:
        if array.ndim != len(input_axes):
            raise ValueError(f"array.ndim ({array.ndim}) need to be equal to length of axes ('{input_axes}')")
        if input_axes == return_axes:
            return array
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
        return cls._reorder_axes(array, axes, cls.array_axis_order)

    def get_dimension_number(self) -> int:
        """return number of nontrivial dimensions"""
        return np.squeeze(self._channel_arrays[0]).ndim

    def get_dimension_letters(self) -> str:
        """
        :return: letters which indicates non trivial dimensions
        """
        return "".join(key for val, key in zip(self._channel_arrays[0].shape, self.array_axis_order) if val > 1)

    def substitute(
        self,
        data=None,
        image_spacing=None,
        file_path=None,
        mask=_DEF,
        default_coloring=None,
        ranges=None,
        channel_names=None,
    ) -> Image:
        """Create copy of image with substitution of not None elements"""
        data = self._channel_arrays if data is None else data
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
            axes_order=self.axis_order,
            metadata=self.metadata,
        )

    def set_mask(self, mask: np.ndarray | None, axes: str | None = None):
        """
        Set mask for image, check if it has proper shape.

        :param mask: mask in same shape like image. May not contains 1 dim axes.
        :param axes: order of axes in mask, use if different than :py:attr:`return_order`
        :raise ValueError: on wrong shape
        """
        if mask is None:
            self._mask_array = None
        elif axes is not None:
            self._mask_array = self.fit_mask_to_image(self.reorder_axes(mask, axes))
        else:
            self._mask_array = self.fit_mask_to_image(mask)

    def get_data(self) -> np.ndarray:
        if "C" in self.axis_order:
            return np.stack(self._channel_arrays, axis=self.axis_order.index("C"))
        return self._channel_arrays[0]

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask_array[:] if self._mask_array is not None else None

    @staticmethod
    def _fit_array_to_image(base_shape, array: np.ndarray) -> np.ndarray:
        """change shape of array with inserting single dimensional entries"""
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
        Change shape of array with inserting single dimensional entries

        :param np.ndarray array: array to be fitted

        :return: reshaped array with added missing 1 in shape

        :raises ValueError: if cannot fit array
        """
        return self._fit_array_to_image(self._channel_arrays[0].shape, array)

    # noinspection DuplicatedCode
    def fit_mask_to_image(self, array: np.ndarray) -> np.ndarray:
        """
        call :py:meth:`fit_array_to_image` and
        then relabel and change type to minimal
        which fit all information
        """
        array = self.fit_array_to_image(array)
        if np.max(array) == 1:
            return array.astype(np.uint8)
        unique = np.unique(array)
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
        if "C" in self.axis_order:
            return self._reorder_axes(
                np.stack(self._channel_arrays, axis=self.axis_order.index("C")), self.axis_order, "TZCYX"
            )
        return self._reorder_axes(self._channel_arrays[0], self.axis_order, "TZCYX")

    def get_mask_for_save(self) -> np.ndarray | None:
        """
        :return: if image has mask then return mask with axes in proper order
        """
        if self._mask_array is None:
            return None
        return self._reorder_axes(self._mask_array, "".join(self.array_axis_order), "TZCYX")

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
        return len(self._channel_arrays)

    @property
    def layers(self) -> int:
        """z-dim of image"""
        return self._channel_arrays[0].shape[self.stack_pos]

    @property
    def times(self) -> int:
        """number of time frames"""
        return self._channel_arrays[0].shape[self.time_pos]

    @property
    def plane_shape(self) -> tuple[int, int]:
        """y,x size of image"""
        return self._channel_arrays[0].shape[self.y_pos], self._channel_arrays[0].shape[self.x_pos]

    @property
    def shape(self):
        """Whole image shape. order of axes my change. Current order is in :py:attr:`return_order`"""
        return self._channel_arrays[0].shape

    def swap_time_and_stack(self):
        """
        Swap time and stack axes.
        For example my be used to convert time image in 3d image.
        """
        image_array_list = [np.swapaxes(x, self.time_pos, self.stack_pos) for x in self._channel_arrays]
        return self.substitute(data=self._image_data_normalize(image_array_list))

    @classmethod
    def get_axis_positions(cls) -> dict[str, int]:
        """
        :return: dict with mapping axis to its position
        :rtype: dict
        """
        return {letter: i for i, letter in enumerate(cls.axis_order)}

    @classmethod
    def get_array_axis_positions(cls) -> dict[str, int]:
        """
        :return: dict with mapping axis to its position for array fitted to image
        :rtype: dict
        """
        return {letter: i for i, letter in enumerate(cls.array_axis_order)}

    def get_data_by_axis(self, **kwargs) -> np.ndarray:
        """
        Get part of data extracted by sub axis. Axis is selected by single letter from :py:attr:`axis_order`

        :param kwargs: axis list with
        :return:
        :rtype:
        """
        slices: list[int | slice] = [slice(None) for _ in range(len(self.array_axis_order))]
        axis_pos = self.get_array_axis_positions()
        if "c" in kwargs:
            kwargs["C"] = kwargs.pop("c")
        if "C" in kwargs:
            if isinstance(kwargs["C"], Channel):
                kwargs["C"] = kwargs["C"].value
            if isinstance(kwargs["C"], str):
                kwargs["C"] = self.channel_names.index(kwargs["C"])

        channel = kwargs.pop("C", slice(None) if "C" in self.axis_order else 0)
        if isinstance(channel, Channel):
            channel = channel.value

        axis_order = self.axis_order
        for name, value in kwargs.items():
            if name.upper() in axis_pos:
                slices[axis_pos[name.upper()]] = value
                if isinstance(value, int):
                    axis_order = axis_order.replace(name.upper(), "")

        slices_t = tuple(slices)
        if isinstance(channel, int):
            return self._channel_arrays[channel][slices_t]
        return np.stack([x[slices_t] for x in self._channel_arrays[channel]], axis=axis_order.index("C"))

    def clip_array(self, array: np.ndarray, **kwargs: int | slice) -> np.ndarray:
        """
        Clip array by axis. Axis is selected by single letter from :py:attr:`axis_order`

        :param array: array to clip
        :param kwargs: mapping from axis to position or slice on this axis
        :return: clipped array
        """
        array = self.fit_array_to_image(array)
        slices: list[int | slice] = [slice(None) for _ in range(len(self.array_axis_order))]
        axis_pos = self.get_array_axis_positions()
        for name in kwargs:
            if (n := name.upper()) in axis_pos:
                slices[axis_pos[n]] = kwargs[name]
        return array[tuple(slices)]

    def get_channel(self, num: int | str | Channel) -> np.ndarray:
        """
        Alias for :py:func:`get_sub_data` with argument ``c=num``

        :param int | str | Channel num: channel num or name to be extracted
        :return: given channel array
        :rtype: numpy.ndarray
        """
        return self.get_data_by_axis(c=num)

    def has_channel(self, num: int | str | Channel) -> bool:
        if isinstance(num, Channel):
            num = num.value

        if isinstance(num, str):
            return num in self.channel_names
        return 0 <= num < self.channels

    def get_layer(self, time: int, stack: int) -> np.ndarray:
        """
        return single layer contains data for all channel

        :param time: time coordinate. For images with not time use 0.
        :param stack: "z coordinate. For time data use 0.
        :return:
        """
        warnings.warn(
            "Image.get_layer is deprecated. Use get_data_by_axis instead", category=DeprecationWarning, stacklevel=2
        )
        return self.get_data_by_axis(T=time, Z=stack)

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
        return tuple(self._image_spacing[1:]) if self.is_2d else self._image_spacing

    def normalized_scaling(self, factor=DEFAULT_SCALE_FACTOR) -> Spacing:
        if self.is_2d:
            return (1, 1, *tuple(np.multiply(self.spacing, factor)))
        return (1, *tuple(np.multiply(self.spacing, factor)))

    @property
    def shift(self):
        return self._shift[1:] if self.is_2d else self._shift

    @property
    def voxel_size(self) -> Spacing:
        """alias for spacing"""
        return self.spacing

    def set_spacing(self, value: Spacing):
        """set image spacing"""
        if 0 in value:
            return
        if self.is_2d and len(value) + 1 == len(self._image_spacing):
            value = (1.0, *tuple(value))
        if len(value) != len(self._image_spacing):  # pragma: no cover
            raise ValueError("Correction of spacing fail.")
        self._image_spacing = tuple(value)

    @staticmethod
    def _frame_array(array: np.ndarray | None, index_to_add: list[int], frame=FRAME_THICKNESS):
        if array is None:  # pragma: no cover
            return array
        result_shape = list(array.shape)
        image_pos = [slice(None) for _ in range(array.ndim)]

        for index in index_to_add:
            result_shape[index] += frame * 2
            image_pos[index] = slice(frame, result_shape[index] - frame)

        data = np.zeros(shape=result_shape, dtype=array.dtype)
        data[tuple(image_pos)] = array
        return data

    @staticmethod
    def calc_index_to_frame(array_axis: str, important_axis: str) -> list[int]:
        """
        calculate in which axis frame should be added

        :param str array_axis: list of image axis
        :param str important_axis: list of framed axis
        :return: list of indices to add frame.
        """
        return [array_axis.index(letter) for letter in important_axis]

    def _frame_cut_area(self, cut_area: typing.Iterable[slice], frame: int):
        cut_area = list(cut_area)
        important_axis = "XY" if self.is_2d else "XYZ"
        for ind in self.calc_index_to_frame(self.array_axis_order, important_axis):
            sl = cut_area[ind]
            cut_area[ind] = slice(
                max(sl.start - frame, 0) if sl.start is not None else None,
                sl.stop + frame if sl.stop is not None else None,
                sl.step,
            )
        return cut_area

    def _cut_image_slices(
        self, cut_area: typing.Iterable[slice], frame: int
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        new_mask = None
        cut_area = self._frame_cut_area(cut_area, frame)
        new_image = [x[tuple(cut_area)] for x in self._channel_arrays]
        if self._mask_array is not None:
            new_mask = self._mask_array[tuple(cut_area)]
        return new_image, new_mask

    def _roi_to_slices(self, roi: np.ndarray) -> list[slice]:
        cut_area = self.fit_array_to_image(roi)
        points = np.nonzero(cut_area)
        lower_bound = np.min(points, axis=1)
        upper_bound = np.max(points, axis=1)
        return [slice(x, y + 1) for x, y in zip(lower_bound, upper_bound)]

    def _cut_with_roi(self, cut_area: np.ndarray, replace_mask: bool, frame: int):
        new_mask = None
        cut_area = self.fit_array_to_image(cut_area)
        new_cut = tuple(self._roi_to_slices(cut_area))
        catted_cut_area = cut_area[new_cut]
        new_image = [np.copy(x[new_cut]) for x in self._channel_arrays]
        for el in new_image:
            el[catted_cut_area == 0] = 0
        if replace_mask:
            new_mask = catted_cut_area
        elif self._mask_array is not None:
            new_mask = self._mask_array[new_cut]
            new_mask[catted_cut_area == 0] = 0
        important_axis = "XY" if self.is_2d else "XYZ"
        new_image = [
            self._frame_array(x, self.calc_index_to_frame(self.array_axis_order, important_axis), frame)
            for x in new_image
        ]

        new_mask = self._frame_array(new_mask, self.calc_index_to_frame(self.array_axis_order, important_axis), frame)
        return new_image, new_mask

    def cut_image(
        self,
        cut_area: np.ndarray | typing.Iterable[slice],
        replace_mask=False,
        frame: int = FRAME_THICKNESS,
        zero_out_cut_area: bool = True,
    ) -> Image:
        """
        Create new image base on mask or list of slices
        :param bool replace_mask: if cut area is represented by mask array,
        then in result image the mask is set base on cut_area if cur_area is np.ndarray
        :param typing.Union[np.ndarray, typing.Iterable[slice]] cut_area: area to cut. Defined with slices or mask
        :param int frame: additional frame around cut_area
        :param bool zero_out_cut_area:
        :return: Image
        """
        if isinstance(cut_area, np.ndarray):
            if zero_out_cut_area:
                new_image, new_mask = self._cut_with_roi(cut_area, replace_mask, frame)
            else:
                new_cut = self._roi_to_slices(cut_area)
                new_image, new_mask = self._cut_image_slices(new_cut, frame)
                if replace_mask:
                    new_mask = cut_area[tuple(self._frame_cut_area(new_cut, frame))]
        else:
            new_image, new_mask = self._cut_image_slices(cut_area, frame)

        return self.__class__(
            data=self._image_data_normalize(new_image),
            image_spacing=self._image_spacing,
            file_path=None,
            mask=new_mask,
            default_coloring=self.default_coloring,
            ranges=self.ranges,
            channel_names=self.channel_names,
            axes_order=self.axis_order,
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
        return tuple(float(x * 10**6) for x in self.spacing)

    def get_um_shift(self) -> Spacing:
        """image spacing in micrometers"""
        return tuple(float(x * 10**6) for x in self.shift)

    def get_ranges(self) -> list[tuple[float, float]]:
        """image brightness ranges for each channel"""
        return self.ranges[:]

    def __str__(self):
        return (
            f"{self.__class__} Shape {self._channel_arrays[0].shape}, dtype: {self._channel_arrays[0].dtype}, "
            f"labels: {self.channel_names}, coloring: {self.get_colors()} mask: {self.has_mask}"
        )

    def __repr__(self):
        mask_info = f"mask=True, mask_dtype={self._mask_array.dtype}" if self.mask is not None else "mask=False"
        return (
            f"Image(shape={self._channel_arrays[0].shape} dtype={self._channel_arrays[0].dtype}, spacing={self.spacing}"
            f", labels={self.channel_names}, channels={self.channels}, axes={self.axis_order!r}, {mask_info})"
        )

    @classmethod
    def _image_data_normalize(cls, data: _IMAGE_DATA) -> _IMAGE_DATA:
        if isinstance(data, np.ndarray):
            return data
        if cls.axis_order.startswith("C"):
            shape = data[0].shape
            if any(x.shape != shape for x in data):
                raise ValueError(f"Shape of arrays are different {[x.shape for x in data]}")
            return data
        if "C" not in cls.axis_order:
            return data[0]
        return np.stack(data, axis=cls.axis_order.index("C"))
