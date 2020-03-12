import numpy as np
import typing

from tifffile import lazyattr

Spacing = typing.Tuple[typing.Union[float, int], ...]


def minimal_dtype(val: int):
    """
    Calculate minimal type to handle value in array

    :param val:
    :return: minimal dtype to handle given value
    :rtype:
    """
    if val < 250:
        return np.uint8
    elif val < 2 ** 16 - 5:
        return np.uint16
    else:
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
    if components is None:
        components = np.unique(array.flat)
        if max_val is None:
            max_val = np.max(components)

    if max_val is None:
        max_val = np.max(array)

    translate = np.zeros(max_val + 1, dtype=dtype if dtype else minimal_dtype(len(components) + 1))
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
    :param labels: labels for channels
    :param axes_order: allow to create Image object form data with different axes order, or missed axes

    :cvar str ~.return_order: internal order of axes

    It is prepared for subclassing with changed internal order. Eg:

    >>> class ImageJImage(Image):
    >>>     return_order = "TZCYX"

    """

    _image_spacing: Spacing
    return_order = "TZYXC"

    def __init__(
        self,
        data: np.ndarray,
        image_spacing: Spacing,
        file_path=None,
        mask: typing.Union[None, np.ndarray] = None,
        default_coloring=None,
        ranges=None,
        labels=None,
        axes_order: typing.Optional[str] = None,
    ):
        # TODO add time distance to image spacing
        if axes_order is None:
            axes_order = self.return_order
        if data.ndim != len(axes_order):
            raise ValueError(
                "Data should have same number of dimensions " f"like length of axes_order (current :{len(axes_order)}"
            )
        if not isinstance(image_spacing, tuple):
            image_spacing = tuple(image_spacing)
        self._image_array = self.reorder_axes(data, axes_order)
        self._image_spacing = (1.0,) * (3 - len(image_spacing)) + image_spacing
        self._image_spacing = tuple([el if el > 0 else 10 ** -6 for el in self._image_spacing])
        self.file_path = file_path
        self.default_coloring = default_coloring
        if self.default_coloring is not None:
            self.default_coloring = [np.array(x) for x in default_coloring]
        self.labels = labels
        if isinstance(self.labels, (tuple, list)):
            self.labels = self.labels[: self.channels]
        if ranges is None:
            axis = list(range(len(self.return_order)))
            axis.remove(self.return_order.index("C"))
            axis = tuple(axis)
            self.ranges = list(zip(np.min(self._image_array, axis=axis), np.max(self._image_array, axis=axis)))
        else:
            self.ranges = ranges
        if mask is not None:
            data_shape = list(data.shape)
            try:
                data_shape.pop(axes_order.index("C"))
            except ValueError:
                pass
            mask = self._fit_array_to_image(data_shape, mask)
            mask = np.take(self.reorder_axes(mask, axes_order.replace("C", "")), 0, self.channel_pos)
            self._mask_array = self.fit_mask_to_image(mask)
        else:
            self._mask_array = None

    @lazyattr
    def channel_pos(self):
        return self.return_order.index("C")

    @lazyattr
    def time_pos(self):
        return self.return_order.index("T")

    @lazyattr
    def stack_pos(self):
        return self.return_order.index("Z")

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
        return cls._reorder_axes(array, axes, cls.return_order)

    def get_dimension_number(self) -> int:
        """return number of nontrivial dimensions"""
        return np.squeeze(self._image_array).ndim

    def get_dimension_letters(self) -> str:
        """
        :return: letters which indicates non trivial dimensions
        """
        return "".join([key for val, key in zip(self._image_array.shape, self.return_order) if val > 1])

    def substitute(
        self, data=None, image_spacing=None, file_path=None, mask=None, default_coloring=None, ranges=None, labels=None
    ):
        """Create copy of image with substitution of not None elements"""
        data = self._image_array if data is None else data
        image_spacing = self._image_spacing if image_spacing is None else image_spacing
        file_path = self.file_path if file_path is None else file_path
        mask = self._mask_array if mask is None else mask
        default_coloring = self.default_coloring if default_coloring is None else default_coloring
        ranges = self.ranges if ranges is None else ranges
        labels = self.labels if labels is None else labels
        return self.__class__(
            data=data,
            image_spacing=image_spacing,
            file_path=file_path,
            mask=mask,
            default_coloring=default_coloring,
            ranges=ranges,
            labels=labels,
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
                raise ValueError("Wrong array shape")
        if len(shape) != len(base_shape):
            raise ValueError("Wrong array shape")
        return np.reshape(array, shape)

    def fit_array_to_image(self, array: np.ndarray) -> np.ndarray:
        """change shape of array with inserting singe dimensional entries"""
        base_shape = list(self._image_array.shape)
        base_shape.pop(self.channel_pos)
        return self._fit_array_to_image(base_shape, array)

    @staticmethod
    def _change_array_type_to_minimal(array, max_val):
        return array.astype(minimal_dtype(max_val))

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
        return self._reorder_axes(self._image_array, self.return_order, "TZCYX")

    def get_mask_for_save(self) -> typing.Optional[np.ndarray]:
        """
        :return: if image has mask then return mask with axes in proper order
        """
        if not self.has_mask:
            return None
        axes_order = list(self.return_order)
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
        x_index = self.return_order.index("X")
        y_index = self.return_order.index("Y")
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

    def get_channel(self, num) -> np.ndarray:
        """"""
        return np.take(self._image_array, num, self.channel_pos)

    def get_layer(self, time: int, stack: int) -> np.ndarray:
        """
        return single layer contains data for all channel

        :param time: time coordinate. For images with not time use 0.
        :param stack: "z coordinate. For time data use 0.
        :return:
        """
        elem_num = max(self.stack_pos, self.time_pos) + 1
        indices = [0] * elem_num
        for i in range(elem_num):
            if i == self.time_pos:
                indices[i] = time
            elif i == self.stack_pos:
                indices[i] = stack
            else:
                indices[i] = slice(None)
        indices = tuple(indices)
        return self._image_array[indices]

    def get_mask_layer(self, num) -> np.ndarray:
        if self._mask_array is None:
            raise ValueError("No mask")
        return self._mask_array[0, num]

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

    @property
    def voxel_size(self) -> Spacing:
        """alias for spacing"""
        return self.spacing

    def set_spacing(self, value: Spacing):
        """set image spacing"""
        if self.is_2d and len(value) + 1 == len(self._image_spacing):
            value = (1.0,) + tuple(value)
        if len(value) != len(self._image_spacing):
            raise ValueError("Correction of spacing fail.")
        self._image_spacing = tuple(value)

    def cut_image(
        self, cut_area: typing.Union[np.ndarray, typing.List[slice], typing.Tuple[slice]], replace_mask=False
    ):
        """
        Create new image base on mask or list of slices
        :param replace_mask: if cut area is represented by mask array,
        then in result image the mask is set base on cut_area
        :param cut_area: area to cut. Defined with slices or mask
        :return: Image
        """
        new_mask = None
        if isinstance(cut_area, (list, tuple)):
            new_image = self._image_array[cut_area]
            if self._mask_array is not None:
                new_mask = self._mask_array[cut_area]
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
            if self.channel_pos == len(self.return_order) - 1:
                new_image[catted_cut_area == 0] = 0
            else:
                for i in range(self.channels):
                    np.take(new_image, i, self.channel_pos)[catted_cut_area == 0] = 0
            if replace_mask:
                new_mask = catted_cut_area
            elif self._mask_array is not None:
                new_mask = self._mask_array[tuple(new_cut)]
                new_mask[catted_cut_area == 0] = 0
        return self.__class__(
            new_image, self._image_spacing, None, new_mask, self.default_coloring, self.ranges, self.labels
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
        return tuple([float(x * 10 ** 6) for x in self.spacing])

    def get_ranges(self) -> typing.Collection[typing.Tuple[float, float]]:
        """image brightness ranges for each channel"""
        return self.ranges[:]

    def __str__(self):
        return (
            f"{self.__class__} Shape {self._image_array.shape}, dtype: {self._image_array.dtype}, "
            f"labels: {self.labels}, coloring: {self.get_colors()} mask: {self.has_mask}"
        )
