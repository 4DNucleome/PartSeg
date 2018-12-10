import inspect
import logging
from enum import Enum

import SimpleITK as sitk
import numpy as np

from partseg_utils import bisect
from partseg_utils import class_to_dict
from partseg_utils.global_settings import develop
from partseg_utils.image_operations import gaussian, DrawType


class ThresholdType(Enum):
    upper = 1
    lower = 2

UPPER = "Upper"


class SegmentationProfile(object):
    PARAMETERS = ("name", "threshold", "threshold_list", "threshold_type", "minimum_size", "use_gauss", "gauss_radius",
                  "threshold_layer_separate", "leave_biggest")
    SEGMENTATION_PARAMETERS = ("threshold", "threshold_list", "threshold_type", "minimum_size", "use_gauss",
                               "gauss_radius", "threshold_layer_separate", "leave_biggest")

    def __init__(self, name, threshold, threshold_list, threshold_type, minimum_size, use_gauss, gauss_radius,
                 threshold_layer_separate, leave_biggest=False):
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
        self.leave_biggest = leave_biggest
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

    def leave_biggest_swap(self):
        self.leave_biggest = not self.leave_biggest


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
    :type _settings: Settings
    """

    def __init__(self, settings, callback=True):
        """
        :type settings: Settings
        """
        self._settings = settings
        self.draw_canvas = None
        self.draw_counter = 0
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

    @property
    def gauss_image(self):
        if develop:
            cur_frame = inspect.currentframe()
            cal_frame = inspect.getouterframes(cur_frame, 2)
            logging.warning('caller name:', cal_frame[1][3])
        return self._settings.gauss_image

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
            for i in range(self._settings.image.shape[0]):
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
            if self.draw_canvas is None:
                return
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
        if self._settings.leave_biggest:
            self._finally_segment = np.copy(self._segmented_image)
            self._finally_segment[self._finally_segment > 1] = 0
            self._segmentation_changed = True
            for fun in self.segmentation_change_callback:
                if isinstance(fun, tuple):
                    fun[0](self._sizes_array[1:2])
                    continue
                if callable(fun):
                    fun()
        ind = bisect(self._sizes_array[1:], self._settings.minimum_size, lambda x, y: x > y)
        # print(ind, self._sizes_array, self._settings.minimum_size)
        if self.draw_canvas is not None:
            hide_set = set(np.unique(self._segmented_image[self.draw_canvas == DrawType.force_hide.value]))
            show_set = set(np.unique(self._segmented_image[self.draw_canvas == DrawType.force_show.value]))
            hide_set -= show_set
            show_set.discard(0)
        else:
            hide_set = set()
            show_set = set()
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
        """:rtype: np.ndarray"""
        self._segmentation_changed = False
        return self._finally_segment

    def get_size_array(self):
        return self._sizes_array

    def get_full_segmentation(self):
        return self._segmented_image

    def add_segmentation_callback(self, callback):
        self.segmentation_change_callback.append(callback)


def fill_holes_in_mask(mask):
    """:rtype: np.ndarray"""
    holes_mask = (mask == 0).astype(np.uint8)
    component_mask = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(holes_mask)))
    border_set = set()
    for dim_num in range(component_mask.ndim):
        border_set.update(np.unique(np.take(component_mask, [0, -1], axis=dim_num)))
    for i in range(1, np.max(component_mask)+1):
        if i not in border_set:
            component_mask[component_mask == i] = 0
    return component_mask == 0


def fill_2d_holes_in_mask(mask):
    """
    :type mask: np.ndarray
    :rtype: np.ndarray
    """
    mask = np.copy(mask)
    if mask.ndim == 2:
        return fill_holes_in_mask(mask)
    for i in range(mask.shape[0]):
        mask[i] = fill_holes_in_mask(mask[i])
    return mask



