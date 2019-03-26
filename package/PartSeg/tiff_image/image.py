import numpy as np
import typing


class Image(object):
    """
    Base object for storing image.
    default order is T, Z, Y, X, C

    """
    return_order = "TZYXC"

    def __init__(self, data: np.ndarray, image_spacing, file_path=None, mask: typing.Union[None, np.ndarray] = None,
                 default_coloring=None, ranges=None, labels=None):
        """
        Base class for Images used in PartSeg
        :param data: 5-dim array with order: time, z, y, x, channel
        :param image_spacing: spacing for z, y, x (TODO add time distance)
        :param file_path: path to image on disc
        :param mask: mask array in shape z,y,x
        :param default_coloring: default colormap - not used yet
        :param ranges: default ranges for channels
        :param labels: labels for channels
        """
        assert len(data.shape) == 5
        self._image_array = data
        self._image_spacing = tuple([1] * (3 - len(image_spacing)) + list(image_spacing))
        self.file_path = file_path
        self.default_coloring = default_coloring
        if self.default_coloring is not None:
            self.default_coloring = [np.array(x) for x in default_coloring]
        self.labels = labels
        if isinstance(self.labels, (tuple, list)):
            self.labels = self.labels[:self.channels]
        if ranges is None:
            self.ranges = list(
                zip(np.min(self._image_array, axis=(0, 1, 2, 3)), np.max(self._image_array, axis=(0, 1, 2, 3))))
        else:
            self.ranges = ranges
        self._mask_array = self.fit_array_to_image(mask) if mask is not None else None

    def get_dimension_number(self):
        return np.squeeze(self._image_spacing).ndim

    def get_dimension_letters(self):
        return [key for val, key in zip(self._image_array.shape, "tzyx") if val > 1]



    def substitute(self, data=None, image_spacing=None, file_path=None, mask=None, default_coloring=None, ranges=None,
                   labels=None):
        """Create copy of image with substitution of not None elements"""
        data = self._image_array if data is None else data
        image_spacing = self._image_spacing if image_spacing is None else image_spacing
        file_path = self.file_path if file_path is None else file_path
        mask = self._mask_array if mask is None else mask
        default_coloring = self.default_coloring if default_coloring is None else default_coloring
        ranges = self.ranges if ranges is None else ranges
        labels = self.labels if labels is None else labels
        return self.__class__(data=data, image_spacing=image_spacing, file_path=file_path, mask=mask,
                              default_coloring=default_coloring, ranges=ranges, labels=labels)

    def set_mask(self, mask: np.ndarray):
        if mask is None:
            self._mask_array = None
        else:
            self._mask_array = self.fit_array_to_image(mask)

    def get_data(self):
        return self._image_array

    @property
    def mask(self):
        return self._mask_array

    def fit_array_to_image(self, array: np.ndarray):
        shape = list(array.shape)
        for i, el in enumerate(self._image_array.shape[:-1]):
            if el == 1 and el != shape[i]:
                shape.insert(i, 1)
            elif el != shape[i]:
                raise ValueError("Wrong array shape")
        if len(shape) != len(self._image_array.shape[:-1]):
            raise ValueError("Wrong array shape")
        return np.reshape(array, shape)

    def get_image_for_save(self):
        array = np.moveaxis(self._image_array, 4, 2)
        return np.reshape(array, array.shape)

    def get_mask_for_save(self):
        if not self.has_mask:
            return None
        return np.reshape(self._mask_array, self._mask_array.shape[:2] + (1,) + self._mask_array.shape[2:])

    @property
    def has_mask(self):
        return self._mask_array is not None

    @property
    def is_time(self):
        return self._image_array.shape[0] > 1

    @property
    def is_stack(self):
        return self._image_array.shape[1] > 1

    @property
    def channels(self):
        return self._image_array.shape[-1]

    @property
    def layers(self):
        return self._image_array.shape[1]

    @property
    def times(self):
        return self._image_array.shape[0]

    @property
    def plane_shape(self):
        return self._image_array.shape[2:4]

    @property
    def shape(self):
        return self._image_array.shape

    def swap_time_and_stack(self):
        image_array = np.swapaxes(self._image_array, 0, 1)
        return self.substitute(data=image_array)

    def __getitem__(self, item):
        # TODO not good solution, improve it
        li = []
        if self.is_time:
            li.append(slice(None))
        else:
            li.append(0)
        if not self.is_stack:
            li.append(0)
        li = tuple(li)
        return self._image_array[li][item]

    def get_channel(self, num) -> np.ndarray:
        if self.is_time:
            li = slice(None)
        else:
            li = 0
        return self._image_array[li, ..., num]

    def get_layer(self, time, stack) -> np.ndarray:
        return self._image_array[time, stack]

    def get_mask_layer(self, num) -> np.ndarray:
        if self._mask_array is None:
            raise ValueError("No mask")
        return self._mask_array[0, num]

    @property
    def is_2d(self):
        return self._image_array.shape[1] == 1 and self._image_array.shape[0] == 1

    @property
    def spacing(self):
        if self.is_2d:
            return tuple(self._image_spacing[1:])
        return self._image_spacing

    @property
    def voxel_size(self):
        return self.spacing

    def set_spacing(self, value):
        if self.is_2d and len(value) + 1 == len(self._image_spacing):
            value = (1,) + tuple(value)
        assert len(value) == len(self._image_spacing)
        self._image_spacing = tuple(value)

    def cut_image(self, cut_area: typing.Union[np.ndarray, typing.List[slice], typing.Tuple[slice]], replace_mask=False):
        """
        Create new image base on mask or list of slices
        :param replace_mask: if cut area is represented by mask array,
        then in result image the mask is set base on cut_area
        :param cut_area: area to cut. Defined with slices or mask
        :return: Image
        """
        new_mask = None
        if isinstance(cut_area, (list,tuple)):
            new_image = self._image_array[cut_area]
            if self._mask_array is not None:
                new_mask = self._mask_array[cut_area]
        else:
            cut_area = self.fit_array_to_image(cut_area)
            points = np.nonzero(cut_area)
            lower_bound = np.min(points, axis=1)
            upper_bound = np.max(points, axis=1)
            new_cut = tuple([slice(x, y + 1) for x, y in zip(lower_bound, upper_bound)])
            new_image = np.copy(self._image_array[new_cut])
            catted_cut_area = cut_area[new_cut]
            new_image[catted_cut_area == 0] = 0
            if replace_mask:
                new_mask = catted_cut_area
            elif self._mask_array is not None:
                new_mask = self._mask_array[new_cut]
                new_mask[catted_cut_area == 0] = 0
        return self.__class__(new_image, self._image_spacing, None, new_mask,
                              self.default_coloring, self.ranges, self.labels)

    def get_imagej_colors(self):
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
                    res.append(np.array(
                        [np.interp(np.linspace(0, 255, num=256), np.linspace(0, color.shape[1], num=256), x)
                         for x in color])
                    )
                res.append(color)
        return res

    def get_colors(self):
        if self.default_coloring is None:
            return None
        res = []
        for color in self.default_coloring:
            if color.ndim == 2:
                res.append(list(color[:, -1]))
            else:
                res.append(list(color))
        return res

    def get_um_spacing(self):
        return [x * 10 ** 6 for x in self.spacing]

    def get_ranges(self):
        return self.ranges

    def __str__(self):
        return f"{self.__class__} Shape {self._image_array.shape}, dtype: {self._image_array.dtype}, " \
               f"labels: {self.labels}, coloring: {self.get_colors()} mask: {self.has_mask}"
