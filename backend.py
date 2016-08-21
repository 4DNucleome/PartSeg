from __future__ import print_function, division
import matplotlib
from matplotlib import pyplot
import numpy as np
import SimpleITK as sitk


def gaussian(image, radius):
    """
    :param image: image to apply gausian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(layer), radius))
    return res


def bisect(arr, val, comp):
    l = -1
    r = len(arr)
    while r - l > 1:
        e = (l + r) >> 1
        if comp(arr[e], val): l = e
        else: r = e
    return r


class Profile:
    def __init__(self, name, threshold, threshold_list, threshold_type, minimum_size):
        """
        :param name: str,
        :param threshold: int
        :param threshold_list: list[int]
        :param threshold_type: str
        :param minimum_size: int
        """
        self.name = name
        self.threshold = threshold
        self.threshold_list = threshold_list
        self.threshold_type = threshold_type
        self.minimum_size = minimum_size


class Settings(object):
    """
    :type profiles: dict[str, Profile]
    :type threshold: int
    :type threshold_list: list[int]
    :type threshold_type: str
    :type minimum_size: int
    :type image: np.ndarray
    :type image_change_callback_: list[() -> None]
    """
    def __init__(self, setings_path):
        # TODO Reading setings from file
        self.color_map_name = "cubehelix"
        self.color_map = matplotlib.cm.get_cmap(self.color_map_name)
        self.callback_color_map = []
        self.chosen_colormap = pyplot.colormaps()
        self.profiles = {}
        self.use_gauss = False
        self.threshold = 33000
        self.threshold_list = []
        self.threshold_type = "Upper"
        self.threshold_layer_separate = False
        self.minimum_size = 100
        self.overlay = 0.7
        self.image = None
        self.mask = None
        self.image_change_callback = []
        self.threshold_change_callback = []
        self.minimum_size_change_callback = []
        self.layer_num = 0
        self.open_directory = None
        self.save_directory = None
        self.spacing = (1, 1, 6)

    def change_colormap(self, new_color_map):
        """
        :type new_color_map: str
        :param new_color_map: name of new colormap
        :return:
        """
        self.color_map_name = new_color_map
        self.color_map = matplotlib.cm.get_cmap(new_color_map)
        for fun in self.callback_color_map:
            fun()

    def add_colormap_callback(self, callback):
        self.callback_color_map.append(callback)

    def remove_colormap_callback(self, callback):
        self.callback_color_map.remove(callback)

    def create_new_profile(self, name, overwrite=False):
        """
        :type name: str
        :type overwrite: bool
        :param name: Profile name
        :param overwrite: Overwrite existing profile
        :return:
        """
        if not overwrite and name in self.profiles:
            raise ValueError("Profile with this name already exists")
        self.profiles[name] = Profile(name, self.threshold, self.threshold_list, self.threshold_type, self.minimum_size)

    @property
    def colormap_list(self):
        return self.chosen_colormap

    @property
    def avaliable_colormaps_list(self):
        return pyplot.colormaps()

    def add_image(self, image, mask=None):
        self.image = image
        self.mask = mask
        for fun in self.image_change_callback:
            fun(image)

    def add_image_callback(self, callback):
        self.image_change_callback.append(callback)

    def change_threshold(self, new_threshold):
        if self.threshold_layer_separate:
            self.threshold_list[self.layer_num] = new_threshold
        else:
            self.threshold = new_threshold
        for fun in self.threshold_change_callback:
            fun()

    def change_threshold_type(self, new_type):
        print(new_type)
        if new_type == "Upper threshold:":
            self.threshold_type = "Upper"
        else:
            self.threshold_type = "Lower"
        for fun in self.threshold_change_callback:
            fun()

    def change_gauss(self, use_gauss):
        self.use_gauss = bool(use_gauss)
        for fun in self.threshold_change_callback:
            fun()

    def add_threshold_callback(self, callback):
        self.threshold_change_callback.append(callback)

    def change_min_size(self, new_min_size):
        self.minimum_size = new_min_size
        for fun in self.minimum_size_change_callback:
            fun()

    def add_min_size_callback(self, callback):
        self.minimum_size_change_callback.append(callback)

    def get_profile_list(self):
        return self.profiles.keys()


class Segment(object):
    """:type _segmented_image: np.ndarray"""
    def __init__(self, settings):
        """
        :type settings: Settings
        """
        self._settings = settings
        self._image = None
        self._gauss_image = None
        self._threshold_image = None
        self._segmented_image = None
        self._finally_segment = None
        self._sizes_array = []
        self.segmentation_change_callback = []
        self._segmentation_changed = True
        self._settings.add_threshold_callback(self.threshold_updated)
        self._settings.add_min_size_callback(self.min_size_updated)

    def set_image(self, image):
        self._image = image
        self._gauss_image = gaussian(self._image, 1)  # TODO radius in settings
        self._segmentation_changed = True
        self.threshold_updated()

    def threshold_updated(self):
        self._threshold_image = np.zeros(self._image.shape, dtype=np.uint8)
        if self._settings.use_gauss:
            image_to_threshold = self._gauss_image
        else:
            image_to_threshold = self._image
        # Define wich threshold use
        if self._settings.threshold_type == "Upper":
            def get_mask(image, threshold):
                return image <= threshold
        else:
            def get_mask(image, threshold):
                return image >= threshold

        if self._settings.threshold_layer_separate:
            for i in range(self._image.shape[0]):
                self._threshold_image[i][get_mask(image_to_threshold[i], self._settings.threshold_list[i])] = 1
        else:
            self._threshold_image[get_mask(image_to_threshold, self._settings.threshold)] = 1
        if self._settings.mask is not None:
            self._threshold_image *= (self._settings.mask > 0)
        connect = sitk.ConnectedComponent(sitk.GetImageFromArray(self._threshold_image))
        self._segmented_image = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
        self._sizes_array = np.bincount(self._segmented_image.flat)
        self.min_size_updated()

    def min_size_updated(self):
        ind = bisect(self._sizes_array[1:], self._settings.minimum_size, lambda x, y: x > y)
        # print(ind, self._sizes_array, self._settings.minimum_size)
        self._finally_segment = np.copy(self._segmented_image)
        self._finally_segment[self._finally_segment > ind] = 0
        self._segmentation_changed = True
        for fun in self.segmentation_change_callback:
            if isinstance(fun, tuple):
                fun[0](self._sizes_array[1:ind])
                continue
            fun()

    @property
    def segmentation_changed(self):
        """:rtype: bool"""
        return self._segmentation_changed

    def get_segmentation(self):
        self._segmentation_changed = False
        return self._finally_segment

    def add_segmentation_callback(self, callback):
        self.segmentation_change_callback.append(callback)