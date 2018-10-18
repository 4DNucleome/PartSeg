import numpy as np
import typing


class Image(object):
    """
    Base object for storing image.
    default order is T, Z, Y, X, C

    """
    return_order = "TZCYX"

    def __init__(self, data: np.ndarray, image_spacing, mask: typing.Union[None, np.ndarray] = None,
                 default_coloring=None, ranges=None, labels=None):
        self._image_array = data
        self._mask_array = mask
        self._image_spacing = image_spacing
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
    def plane_shape(self):
        return self._image_array.shape[2:4]

    def swap_time_and_stack(self):
        self._image_array = np.swapaxes(self._image_array, 0, 1)

    def __getitem__(self, item):
        # TODO not good solution, improve it
        li = []
        if self.is_time():
            li.append(slice(None))
        else:
            li.append(0)
        if not self.is_stack():
            li.append(0)
        return self._image_array[li][item]

    def get_channel(self, num):
        return self._image_array[..., num]

    def get_layer(self, num) -> np.ndarray:
        return self._image_array[0, num]

    @property
    def is_2d(self):
        return False

    @property
    def spacing(self):
        return self._image_spacing

    def cut_image(self, cut_area: typing.Union[np.ndarray, typing.List[slice]]):
        """
        Create new image base on mask or list of slices
        :param cut_area: area to cut. Defined with slices or mask
        :return: Image
        """
        new_mask = None
        if isinstance(cut_area, list):
            new_image = self._image_array[cut_area]
            if self._mask_array is not None:
                new_mask = self._mask_array[cut_area]
        else:
            points = np.nonzero(cut_area)
            lower_bound = np.min(points, axis=1)
            upper_bound = np.max(points, axis=1)
            new_cut = [slice(x, y) for x, y in zip(lower_bound, upper_bound)]
            new_image = self._image_array[new_cut]
            catted_cut_area = cut_area[new_cut]
            new_image[catted_cut_area == 0] = 0
            if self._mask_array is not None:
                new_mask = self._mask_array[cut_area]
                new_mask[catted_cut_area == 0] = 0
        return self.__class__(new_image, self._image_spacing, new_mask)

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
        return [x * 10 **6 for x in self.spacing]

    def get_ranges(self):
        return self.ranges

    def __str__(self):
        return f"{self.__class__} Shape {self._image_array.shape}, dtype: {self._image_array.dtype}, " \
               f"labels: {self.labels}, coloring: {self.get_colors()} mask: {self.has_mask}"
