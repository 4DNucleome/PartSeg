from __future__ import print_function, division
import matplotlib
from matplotlib import pyplot
import numpy as np
import SimpleITK as sitk

UPPER = "Upper"

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
    def __init__(self, name, threshold, threshold_list, threshold_type, minimum_size, use_gauss):
        """
        :param name: str,
        :param threshold: int
        :param threshold_list: list[int]
        :param threshold_type: str
        :param minimum_size: int
        :param use_gauss: bool
        """
        self.name = name
        self.threshold = threshold
        self.threshold_list = threshold_list
        self.threshold_type = threshold_type
        self.minimum_size = minimum_size
        self.use_gauss = use_gauss

    def __str__(self):
        text = self.name + "\n"

        return text


class Settings(object):
    """
    :type profiles: dict[str, Profile]
    :type threshold: int
    :type threshold_list: list[int]
    :type threshold_type: str
    :type minimum_size: int
    :type image: np.ndarray
    :type image_change_callback: list[() -> None]
    """
    def __init__(self, setings_path):
        # TODO Reading setings from file
        self.color_map_name = "cubehelix"
        self.color_map = matplotlib.cm.get_cmap(self.color_map_name)
        self.callback_colormap = []
        self.callback_colormap_list = []
        self.callback_change_layer = []
        self.chosen_colormap = pyplot.colormaps()
        self.profiles = dict()
        self.use_gauss = False
        self.use_draw_result = False
        self.draw_callback = []
        self.threshold = 33000
        self.threshold_list = []
        self.threshold_type = UPPER
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
        self.spacing = [1, 1, 3]
        self.voxel_size = [70, 70, 210]
        self.size_unit = "nm"
        self.advanced_menu_geometry = None
        self.file_path = ""
        self.protect = False

    def change_colormap(self, new_color_map):
        """
        :type new_color_map: str
        :param new_color_map: name of new colormap
        :return:
        """
        self.color_map_name = new_color_map
        self.color_map = matplotlib.cm.get_cmap(new_color_map)
        for fun in self.callback_colormap:
            fun()

    def add_colormap_list_callback(self, callback):
        self.callback_colormap_list.append(callback)

    def add_colormap_callback(self, callback):
        self.callback_colormap.append(callback)

    def remove_colormap_callback(self, callback):
        self.callback_colormap.remove(callback)

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
    def available_colormap_list(self):
        return pyplot.colormaps()

    def add_image(self, image, file_path, mask=None,):
        self.image = image
        self.mask = mask
        self.file_path = file_path
        self.threshold_list = []
        self.threshold_layer_separate = False
        for fun in self.image_change_callback:
            if isinstance(fun, tuple) and fun[1] == str:
                fun[0](file_path)
                continue
            fun(image)

    def add_image_callback(self, callback):
        self.image_change_callback.append(callback)

    def change_threshold(self, new_threshold):
        if self.protect:
            return
        if self.threshold_layer_separate:
            if self.threshold_list[self.layer_num] == new_threshold:
                return
            self.threshold_list[self.layer_num] = new_threshold
        else:
            if self.threshold == new_threshold:
                return
            self.threshold = new_threshold
        for fun in self.threshold_change_callback:
            fun()

    def add_change_layer_callback(self, callback):
        self.callback_change_layer.append(callback)

    def change_layer(self, val):
        self.layer_num = val
        self.protect = True
        if self.threshold_layer_separate:
            for fun in self.callback_change_layer:
                fun(self.threshold_list[val])
        else:
            for fun in self.callback_change_layer:
                fun(self.threshold)
        # for fun in self.threshold_change_callback:
        #     fun()
        self.protect = False

    def change_threshold_type(self, new_type):
        print(new_type)
        if new_type == "Upper threshold:":
            self.threshold_type = "Upper"
        else:
            self.threshold_type = "Lower"
        for fun in self.threshold_change_callback:
            fun()

    def change_layer_threshold(self, layer_threshold):
        self.threshold_layer_separate = layer_threshold
        if layer_threshold and self.threshold_list == []:
            self.threshold_list = [self.threshold] * self.image.shape[0]
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

    def set_available_colormap(self, cmap_list):
        self.chosen_colormap = cmap_list
        for fun in self.callback_colormap_list:
            fun()

    def change_draw_use(self, use_draw):
        self.use_draw_result = use_draw
        for fun in self.draw_callback:
            fun()

    def add_draw_callback(self, callback):
        self.draw_callback.append(callback)


class Segment(object):
    """:type _segmented_image: np.ndarray"""
    def __init__(self, settings):
        """
        :type settings: Settings
        """
        self._settings = settings
        self._image = None
        self.draw_canvas = None
        self._gauss_image = None
        self._threshold_image = None
        self._segmented_image = None
        self._finally_segment = None
        self._sizes_array = []
        self.segmentation_change_callback = []
        self._segmentation_changed = True
        self.protect = False
        self._settings.add_threshold_callback(self.threshold_updated)
        self._settings.add_min_size_callback(self.min_size_updated)
        self._settings.add_draw_callback(self.draw_update)

    def set_image(self, image):
        self._image = image
        self._gauss_image = gaussian(self._image, 1)  # TODO radius in settings
        self._segmentation_changed = True
        self.threshold_updated()

    def threshold_updated(self):
        if self.protect:
            return
        self._threshold_image = np.zeros(self._image.shape, dtype=np.uint8)
        if self._settings.use_gauss:
            image_to_threshold = self._gauss_image
        else:
            image_to_threshold = self._image
        # Define wich threshold use
        if self._settings.threshold_type == UPPER:
            def get_mask(image, threshold):
                return image <= threshold
        else:
            def get_mask(image, threshold):
                return image >= threshold

        if self._settings.threshold_layer_separate:
            print("Layer separate")
            for i in range(self._image.shape[0]):
                self._threshold_image[i][get_mask(image_to_threshold[i], self._settings.threshold_list[i])] = 1
        else:
            print("normal")
            self._threshold_image[get_mask(image_to_threshold, self._settings.threshold)] = 1
        if self._settings.mask is not None:
            self._threshold_image *= (self._settings.mask > 0)
        self.draw_update()

    def draw_update(self, canvas=None):
        if self.protect:
            return
        if canvas is not None:
            self.draw_canvas[...] = canvas[...]
            return
        if self._settings.use_draw_result:
            threshold_image = np.copy(self._threshold_image)
            threshold_image[self.draw_canvas == 1] = 1
            threshold_image[self.draw_canvas == 2] = 0
        else:
            threshold_image = self._threshold_image
        connect = sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_image))
        self._segmented_image = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
        self._sizes_array = np.bincount(self._segmented_image.flat)
        self.min_size_updated()

    def min_size_updated(self):
        if self.protect:
            return
        ind = bisect(self._sizes_array[1:], self._settings.minimum_size, lambda x, y: x > y)
        # print(ind, self._sizes_array, self._settings.minimum_size)
        self._finally_segment = np.copy(self._segmented_image)
        self._finally_segment[self._finally_segment > ind] = 0
        self._segmentation_changed = True
        for fun in self.segmentation_change_callback:
            if isinstance(fun, tuple):
                fun[0](self._sizes_array[1:ind+1])
                continue
            fun()

    @property
    def segmentation_changed(self):
        """:rtype: bool"""
        return self._segmentation_changed

    def get_segmentation(self):
        self._segmentation_changed = False
        return self._finally_segment

    def get_full_segmentation(self):
        return self._segmented_image

    def add_segmentation_callback(self, callback):
        self.segmentation_change_callback.append(callback)