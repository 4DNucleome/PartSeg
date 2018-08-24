import SimpleITK as sitk
import numpy as np

from project_utils import bisect
from project_utils.algorithm_base import SegmentationAlgorithm
from project_utils.image_operations import gaussian
from .segment import close_small_holes, opening





class ThresholdPreview(SegmentationAlgorithm):
    def __init__(self):
        super(ThresholdPreview, self).__init__()
        self.use_gauss = False
        self.gauss_radius = 0
        self.threshold = 0
        self.exclude_mask = None
        self.image = None

    def run(self):
        image = self.image
        if self.use_gauss:
            image = gaussian(image, 1)
        if self.exclude_mask is None:
            res = (image > self.threshold).astype(np.uint8)
        else:
            mask = self.exclude_mask > 0
            res = (image > self.threshold).astype(self.exclude_mask.dtype)
            res[mask] = 0
            res[res > 0] = image.max() + 1
            res[mask] = self.exclude_mask[mask]
        self.image = None
        self.exclude_mask = None
        self.execution_done.emit(res)

    def set_parameters(self, image, threshold, exclude_mask, use_gauss, gauss_radius):
        self.image = image
        self.threshold = threshold
        self.exclude_mask = exclude_mask
        self.use_gauss = use_gauss
        self.gauss_radius = gauss_radius


class ThresholdAlgorithm(SegmentationAlgorithm):
    """
    :type segmentation: np.ndarray
    """
    def __init__(self):
        super(ThresholdAlgorithm, self).__init__()
        self.image = None
        self.threshold = None
        self.minimum_size = None
        self.segmentation = None
        self.sizes = None
        self.use_gauss = False
        self.gauss_radius = 0
        self.exclude_mask = None
        self.close_holes = False
        self.close_holes_size = 0
        self.smooth_border = False
        self.smooth_border_radius = 0

    def run(self):
        if self.use_gauss:
            image = gaussian(self.image, self.gauss_radius)
            self.progress_signal.emit("Gauss done", 0)
        else:
            image = self.image
        mask = (image > self.threshold).astype(np.uint8)
        if self.exclude_mask is not None:
            self.progress_signal.emit("Components exclusion apply", 1)
            mask[self.exclude_mask > 0] = 0
        if self.close_holes:
            self.progress_signal.emit("Holes closing", 2)
            mask = close_small_holes(mask, self.close_holes_size)
        self.progress_signal.emit("Components calculating", 3)
        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(mask)
                ), 20
            )
        )
        if self.smooth_border:
            self.progress_signal.emit("Smoothing borders", 4)
            self.segmentation = opening(self.segmentation, self.smooth_border_radius, 20)

        self.sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.sizes[1:], self.minimum_size, lambda x, y: x > y)
        self.image = image
        self.threshold = self.threshold
        self.minimum_size = self.minimum_size
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0
        if self.exclude_mask is not None:
            resp[resp > 0] += self.exclude_mask.max()
            resp[self.exclude_mask > 0] = self.exclude_mask[self.exclude_mask > 0]
        self.progress_signal.emit("Calculation done", 5)
        self.execution_done.emit(resp)

    def set_parameters(self, image, threshold, minimum_size, exclude_mask, close_holes, smooth_border, use_gauss,
                       close_holes_size, smooth_border_radius, gauss_radius):
        self.image = image
        self.threshold = threshold
        self.minimum_size = minimum_size
        self.exclude_mask = exclude_mask
        self.close_holes = close_holes
        self.smooth_border = smooth_border
        self.use_gauss = use_gauss
        self.close_holes_size = close_holes_size
        self.smooth_border_radius = smooth_border_radius
        self.gauss_radius = gauss_radius


class AutoThresholdAlgorithm(SegmentationAlgorithm):
    """
    :type segmentation: np.ndarray
    """
    def __init__(self):
        super(AutoThresholdAlgorithm, self).__init__()
        self.image = None
        self.threshold = None
        self.minimum_size = None
        self.segmentation = None
        self.sizes = None
        self.use_gauss = False
        self.gauss_radius = 0
        self.exclude_mask = None
        self.close_holes = False
        self.close_holes_size = 0
        self.smooth_border = False
        self.smooth_border_radius = 0
        self.suggested_size = 0

    def run(self):
        if self.use_gauss:
            image = gaussian(self.image, self.gauss_radius)
            self.progress_signal.emit("Gauss done", 0)
        else:
            image = np.copy(self.image)
        if self.exclude_mask is not None:
            self.progress_signal.emit("Components exclusion apply", 1)
            image[self.exclude_mask > 0] = 0
        self.progress_signal.emit("Threshold calculation", 1)
        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.ThresholdMaximumConnectedComponents(sitk_image, self.suggested_size)
        mask = sitk.GetArrayFromImage(sitk_mask)
        min_val = np.min(image[mask > 0])
        if self.threshold < min_val:
            self.threshold = min_val
        self.info_signal.emit("Threshold: {}".format(self.threshold))
        mask = (image > self.threshold).astype(np.uint8)
        if self.close_holes:
            self.progress_signal.emit("Holes closing", 2)
            mask = close_small_holes(mask, self.close_holes_size)
        self.progress_signal.emit("Components calculating", 3)

        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(mask)
                ), 20
            )
        )
        if self.smooth_border:
            self.progress_signal.emit("Smoothing borders", 4)
            self.segmentation = opening(self.segmentation, self.smooth_border_radius, 20)

        self.sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.sizes[1:], self.minimum_size, lambda x, y: x > y)
        self.image = image
        self.threshold = self.threshold
        self.minimum_size = self.minimum_size
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0
        if self.exclude_mask is not None:
            resp[resp > 0] += self.exclude_mask.max()
            resp[self.exclude_mask > 0] = self.exclude_mask[self.exclude_mask > 0]
        self.progress_signal.emit("Calculation done", 5)
        self.execution_done.emit(resp)

    def set_parameters(self, image, suggested_size, threshold, minimum_size, exclude_mask, close_holes, smooth_border,
                       use_gauss, close_holes_size, smooth_border_radius, gauss_radius):
        self.image = image
        self.threshold = threshold
        self.minimum_size = minimum_size
        self.exclude_mask = exclude_mask
        self.close_holes = close_holes
        self.smooth_border = smooth_border
        self.use_gauss = use_gauss
        self.close_holes_size = close_holes_size
        self.smooth_border_radius = smooth_border_radius
        self.gauss_radius = gauss_radius
        self.suggested_size = suggested_size



