from collections import defaultdict

from project_utils.algorithm_base import SegmentationAlgorithm
from project_utils.image_operations import gaussian
import numpy as np
import SimpleITK as sitk
from project_utils import bisect


class RestartableAlgorithm(SegmentationAlgorithm):
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


    def run(self):
        restarted = False
        print(self.new_parameters["use_gauss"])
        print(self.parameters)
        print(self.parameters["use_gauss"])
        if self.new_parameters["use_gauss"]:
            if self.parameters["gauss_radius"] != self.new_parameters["gauss_radius"]:
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
            self.execution_done.emit(finally_segment)

    def clean(self):
        super().clean()
        self.gauss_image = None


    def _threshold(self, image):
        raise NotImplementedError()

    def set_image(self, image):
        self.image = image
        self.parameters["gauss_radius"] = None

    def set_mask(self, mask):
        self.mask = mask
        self.new_parameters["threshold"] = self.parameters["threshold"]
        self.parameters["threshold"] = None


    def set_parameters(self, threshold, minimum_size,  use_gauss, gauss_radius):
        self.new_parameters["threshold"] = threshold
        self.new_parameters["minimum_size"] = minimum_size
        self.new_parameters["use_gauss"] = use_gauss
        self.new_parameters["gauss_radius"] = gauss_radius


class LowerThresholdAlgorithm(ThresholdBaseAlgorithm):
    def _threshold(self, image):
        return (image > self.new_parameters["threshold"]).astype(np.uint8)

class UpperThresholdAlgorithm(ThresholdBaseAlgorithm):
    def _threshold(self, image):
        return (image < self.new_parameters["threshold"]).astype(np.uint8)