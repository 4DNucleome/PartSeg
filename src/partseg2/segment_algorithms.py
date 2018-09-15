from collections import defaultdict

from PyQt5.QtCore import pyqtSignal

from project_utils.algorithm_base import SegmentationAlgorithm
from project_utils.distance_in_structure.find_split import distance_sprawl, path_minimum_sprawl, path_maximum_sprawl
from project_utils.image_operations import gaussian
import numpy as np
import SimpleITK as sitk
from project_utils import bisect


class RestartableAlgorithm(SegmentationAlgorithm):
    execution_done_extend = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self.parameters = defaultdict(lambda: None)
        self.new_parameters = {}

    def set_image(self, image):
        self.image = image
        self.parameters.clear()


class ThresholdBaseAlgorithm(RestartableAlgorithm):
    """
    :type segmentation: np.ndarray
    """

    def __init__(self):
        super(ThresholdBaseAlgorithm, self).__init__()
        self.mask = None
        self.gauss_image = None
        self.threshold_image = None
        self._sizes_array = []
        self.components_num = 0

    def run(self):
        finally_segment = self.calculation_run()
        if finally_segment is not None:
            self.execution_done.emit(finally_segment)
            self.execution_done_extend.emit(finally_segment, self.segmentation)
            self.parameters.update(self.new_parameters)

    def calculation_run(self):
        restarted = False
        if self.new_parameters["use_gauss"]:
            if self.parameters["gauss_radius"] != self.new_parameters["gauss_radius"] or \
                    self.new_parameters["use_gauss"] != self.parameters["use_gauss"]:
                self.gauss_image = gaussian(self.image, self.new_parameters["gauss_radius"])
                restarted = True
        elif self.new_parameters["use_gauss"] != self.parameters["use_gauss"]:
            self.gauss_image = self.image
            restarted = True
        if restarted or self.new_parameters["threshold"] != self.parameters["threshold"]:
            threshold_image = self._threshold(self.gauss_image)
            if self.mask is not None:
                threshold_image *= (self.mask > 0)
            connect = sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_image))
            self.segmentation = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
            self._sizes_array = np.bincount(self.segmentation.flat)
            restarted = True
        if restarted or self.new_parameters["minimum_size"] != self.parameters["minimum_size"]:
            ind = bisect(self._sizes_array[1:], self.new_parameters["minimum_size"], lambda x, y: x > y)
            finally_segment = np.copy(self.segmentation)
            finally_segment[finally_segment > ind] = 0
            self.components_num = ind
            return finally_segment

    def _clean(self):
        super()._clean()
        self.parameters = defaultdict(lambda: None)
        self.gauss_image = None
        self.mask = None

    def _threshold(self, image, thr=None):
        raise NotImplementedError()

    def set_image(self, image):
        self.image = image
        self.parameters["gauss_radius"] = None

    def set_mask(self, mask):
        self.mask = mask
        self.new_parameters["threshold"] = self.parameters["threshold"]
        self.parameters["threshold"] = None


class OneTHresholdAlgorithm(ThresholdBaseAlgorithm):
    def set_parameters(self, threshold, minimum_size, use_gauss, gauss_radius):
        self.new_parameters["threshold"] = threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius


class LowerThresholdAlgorithm(OneTHresholdAlgorithm):
    def _threshold(self, image, thr=None):
        return (image > self.new_parameters["threshold"]).astype(np.uint8)


class UpperThresholdAlgorithm(OneTHresholdAlgorithm):
    def _threshold(self, image, thr=None):
        return (image < self.new_parameters["threshold"]).astype(np.uint8)


class RangeThresholdAlgorithm(ThresholdBaseAlgorithm):
    def set_parameters(self, lower_threshold, upper_threshold, minimum_size, use_gauss, gauss_radius):
        self.new_parameters["threshold"] = lower_threshold, upper_threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius

    def _threshold(self, image, thr=None):
        return ((image > self.new_parameters["threshold"][0]) * (image < self.new_parameters["threshold"][1])).astype(
            np.uint8)


class BaseThresholdFlowAlgorithm(ThresholdBaseAlgorithm):
    def path_sprawl(self, base_image, object_image):
        raise NotImplementedError()

    def __init__(self):
        super().__init__()
        self.finally_segment = None

    def set_parameters(self, threshold, minimum_size, use_gauss, gauss_radius, base_threshold):
        self.new_parameters["threshold"] = threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius
        self.new_parameters["base_threshold"] = base_threshold

    def run(self):
        finally_segment = self.calculation_run()
        if finally_segment is None:
            restarted = False
            finally_segment =np.copy(self.finally_segment)
        else:
            self.finally_segment = finally_segment
            restarted = True

        if restarted or self.new_parameters["base_threshold"] != self.parameters["base_threshold"]:
            threshold_image = self._threshold(self.gauss_image, self.new_parameters["base_threshold"])
            print(f"Sizes {np.count_nonzero(threshold_image)}, {np.count_nonzero(finally_segment)}", self.__class__)
            if self.mask is not None:
                threshold_image *= (self.mask > 0)
            new_segment = self.path_sprawl(threshold_image, finally_segment)
            self.execution_done.emit(new_segment)
            self.execution_done_extend.emit(new_segment, threshold_image)
            self.parameters.update(self.new_parameters)


class LowerThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    def _threshold(self, image, thr=None):
        if thr is None:
            thr = self.new_parameters["threshold"]
        print(f"Threshold {thr}")
        return (image > thr).astype(np.uint8)


class UpperThresholdFlowAlgorithm(BaseThresholdFlowAlgorithm):
    def _threshold(self, image, thr=None):
        if thr is None:
            thr = self.new_parameters["threshold"]
        return (image < thr).astype(np.uint8)


class LowerThresholdDistanceFlowAlgorithm(LowerThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        return distance_sprawl(base_image, object_image, self.components_num)


class UpperThresholdDistanceFlowAlgorithm(UpperThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        return distance_sprawl(base_image, object_image, self.components_num)


class LowerThresholdPathFlowAlgorithm(LowerThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        image = self.image.astype(np.float64)
        image[base_image == 0] = 0
        mid = path_maximum_sprawl(image, object_image, self.components_num)
        return path_maximum_sprawl(image, mid, self.components_num)


class UpperThresholdPathFlowAlgorithm(UpperThresholdFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        image = self.image.astype(np.float64)
        image[base_image == 0] = 0
        mid = path_minimum_sprawl(image, object_image, self.components_num)
        return path_minimum_sprawl(image, mid, self.components_num)


class LowerThresholdPathDistanceFlowAlgorithm(LowerThresholdPathFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        mid = super().path_sprawl(base_image, object_image)
        return distance_sprawl(base_image, mid, self.components_num)


class UpperThresholdPathDistanceFlowAlgorithm(UpperThresholdPathFlowAlgorithm):
    def path_sprawl(self, base_image, object_image):
        mid = super().path_sprawl(base_image, object_image)
        return distance_sprawl(base_image, mid, self.components_num)
