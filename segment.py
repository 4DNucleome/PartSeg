import numpy as np
from utils import class_to_dict
from image_operations import gaussian, DrawType
import SimpleITK as sitk
from enum import Enum


class ThresholdType(Enum):
    upper = 1
    lower = 2

UPPER = ThresholdType.upper


class SegmentationProfile(object):
    PARAMETERS = ("name", "threshold", "threshold_list", "threshold_type", "minimum_size", "use_gauss", "gauss_radius",
                  "threshold_layer_separate")
    SEGMENTATION_PARAMETERS = ("threshold", "threshold_list", "threshold_type", "minimum_size", "use_gauss",
                               "gauss_radius", "threshold_layer_separate")

    def __init__(self, name, threshold, threshold_list, threshold_type, minimum_size, use_gauss, gauss_radius,
                 threshold_layer_separate):
        """
        :param name: str,
        :param threshold: int
        :param threshold_list: list[int]
        :param threshold_type: str
        :param minimum_size: int
        :param use_gauss: bool
        """
        self.name = name
        if threshold_layer_separate:
            self.threshold = np.median(threshold_list)
            self.threshold_list = threshold_list
        else:
            self.threshold = threshold
            self.threshold_list = []
        self.threshold_type = threshold_type
        self.minimum_size = minimum_size
        self.use_gauss = use_gauss
        self.gauss_radius = gauss_radius
        self.threshold_layer_separate = threshold_layer_separate

    def __str__(self):
        if self.name != "":
            text = "Name: {}\n".format(self.name)
        else:
            text = ""
        text += "{} threshold: ".format(self.threshold_type)
        if self.threshold_layer_separate:
            text += str(self.threshold_list)
        else:
            text += str(self.threshold)
            text += "\n"
        text += "Minimum object size: {}\n".format(self.minimum_size)
        text += "Use gauss [{}]\n".format("x" if self.use_gauss else " ")
        if self.use_gauss:
            text += "Gauss radius: {}".format(self.gauss_radius)
        return text

    def get_parameters(self):
        return class_to_dict(self, *self.PARAMETERS)


class DummySegment(object):
    def __init__(self):
        self._image = None
        self._full_segmentation = None
        self._segmentation = None

    def calculate_segmentation(self, profile, image, mask):
        """
        :type profile: SegmentationProfile
        :type image: np.array
        :type mask: np.array | None
        :param profile:
        :param image:
        :return:
        """
        self._image = image
        if profile.use_gauss:
            image = gaussian(image, profile.gauss_radius)
        if profile.threshold_type == UPPER:
            def get_mask(img, threshold):
                return np.array(img <= threshold)
        else:
            def get_mask(img, threshold):
                return np.array(img >= threshold)
        if profile.threshold_layer_separate:
            threshold_image = np.zeros(image.shape, dtype=np.uint8)
            for i in range(image.shape[0]):
                threshold_image[i][get_mask(image[i], profile.threshold_list[i])] = 1
        else:
            threshold_image = get_mask(image, profile.threshold).astype(np.uint8)
        if mask is not None:
            threshold_image *= (mask > 0)
        components = sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_image))
        self._full_segmentation = sitk.GetArrayFromImage(threshold_image)
        self._segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(components, profile.minimum_size))

    def get_segmentation(self):
        return self.segmentation

    @property
    def segmentation(self):
        return self._segmentation

    @property
    def draw_canvas(self):
        return np.zeros(self._image.shape, dtype=np.uint8)


class Segment(object):
    """
    :type _segmented_image: np.ndarray
    :type segmentation_change_callback: list[() -> None | (list[int] -> None]
    """

    def __init__(self, settings, callback=True):
        """
        :type settings: Settings
        """
        self._settings = settings
        self._image = None
        self.draw_canvas = None
        self.draw_counter = 0
        self.gauss_image = None
        self._threshold_image = None
        self._segmented_image = None
        self._finally_segment = None
        self._sizes_array = []
        self.segmentation_change_callback = []
        self._segmentation_changed = True
        self.protect = False
        if callback:
            self._settings.add_threshold_callback(self.threshold_updated)
            self._settings.add_min_size_callback(self.min_size_updated)
            self._settings.add_draw_callback(self.draw_update)

    def recalculate(self):
        self.threshold_updated()

    def set_image(self):
        self._segmentation_changed = True
        self._finally_segment = np.zeros(self._settings.image.shape, dtype=np.uint8)
        self.threshold_updated()

    def threshold_updated(self):
        if self.protect:
            return
        self._threshold_image = np.zeros(self._settings.image.shape, dtype=np.uint8)
        if self._settings.use_gauss:
            image_to_threshold = self._settings.gauss_image
        else:
            image_to_threshold = self._settings.image
        # Define which threshold use
        if self._settings.threshold_type == UPPER:
            def get_mask(image, threshold):
                return image <= threshold
        else:
            def get_mask(image, threshold):
                return image >= threshold

        if self._settings.threshold_layer_separate:
            # print("Layer separate")
            for i in range(self._image.shape[0]):
                self._threshold_image[i][get_mask(image_to_threshold[i], self._settings.threshold_list[i])] = 1
        else:
            # print("normal")
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
        hide_set = set(np.unique(self._segmented_image[self.draw_canvas == DrawType.force_hide.value]))
        show_set = set(np.unique(self._segmented_image[self.draw_canvas == DrawType.force_show.value]))
        hide_set -= show_set
        show_set.discard(0)
        hide_set.discard(0)
        finally_segment = np.copy(self._segmented_image)
        finally_segment[finally_segment > ind] = 0
        for val in show_set:
            finally_segment[self._segmented_image == val] = val
        for val in hide_set:
            finally_segment[self._segmented_image == val] = 0
        if len(show_set) > 0 or len(hide_set) > 0:
            self._finally_segment = np.zeros(finally_segment.shape, dtype=finally_segment.dtype)
            for i, val in enumerate(np.unique(finally_segment)[1:], 1):
                self._finally_segment[finally_segment == val] = i
        else:
            self._finally_segment = finally_segment

        self._segmentation_changed = True
        for fun in self.segmentation_change_callback:
            if isinstance(fun, tuple):
                fun[0](self._sizes_array[1:ind+1])
                continue
            if callable(fun):
                fun()

    @property
    def segmentation_changed(self):
        """:rtype: bool"""
        return self._segmentation_changed

    def get_segmentation(self):
        self._segmentation_changed = False
        return self._finally_segment

    def get_size_array(self):
        return self._sizes_array

    def get_full_segmentation(self):
        return self._segmented_image

    def add_segmentation_callback(self, callback):
        self.segmentation_change_callback.append(callback)


def bisect(arr, val, comp):
    l = -1
    r = len(arr)
    while r - l > 1:
        e = (l + r) >> 1
        if comp(arr[e], val):
            l = e
        else:
            r = e
    return r
