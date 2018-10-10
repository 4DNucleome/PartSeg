from abc import ABC

import SimpleITK as sitk
import numpy as np

from project_utils import bisect
from project_utils.algorithm_base import SegmentationAlgorithm
from .segment import close_small_holes, opening


class StackAlgorithm(SegmentationAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.exclude_mask = None

    def _clean(self):
        super()._clean()
        self.exclude_mask = None


class ThresholdPreview(StackAlgorithm):

    def __init__(self):
        super(ThresholdPreview, self).__init__()
        self.use_gauss = "No"
        self.gauss_radius = 0
        self.threshold = 0

    def calculation_run(self, _report_fun):
        image = self.get_gauss(self.use_gauss, self.gauss_radius)
        if self.exclude_mask is None:
            res = (image > self.threshold).astype(np.uint8)
        else:
            mask = self.exclude_mask > 0
            if self.exclude_mask is not None:
                result_data_type = self.exclude_mask.dtype
            else:
                result_data_type = np.uint8
            res = (image > self.threshold).astype(result_data_type)
            res[mask] = 0
            if self.exclude_mask is not None:
                res[res > 0] = self.exclude_mask.max() + 1
            res[mask] = self.exclude_mask[mask]
        self.image = None
        self.exclude_mask = None
        return res

    def set_parameters(self, image, threshold, exclude_mask, use_gauss, gauss_radius):
        self.image = image
        self.threshold = threshold
        self.exclude_mask = exclude_mask
        self.use_gauss = use_gauss
        self.gauss_radius = gauss_radius

    def get_info_text(self):
        return ""


class ThresholdAlgorithm(StackAlgorithm):
    """
    :type segmentation: np.ndarray
    """
    def __init__(self):
        super(ThresholdAlgorithm, self).__init__()
        self.threshold = None
        self.minimum_size = None
        self.sizes = None
        self.use_gauss = False
        self.gauss_radius = 0
        self.close_holes = False
        self.close_holes_size = 0
        self.smooth_border = False
        self.smooth_border_radius = 0
        self.gauss_2d = False
        self.edge_connection = True

    def calculation_run(self, report_fun):
        image = self.get_gauss(self.use_gauss, self.gauss_radius)
        mask = np.array(image > self.threshold).astype(np.uint8)
        if self.exclude_mask is not None:
            report_fun("Components exclusion apply", 1)
            mask[self.exclude_mask > 0] = 0
        if self.close_holes:
            report_fun("Holes closing", 2)
            mask = close_small_holes(mask, self.close_holes_size)
        report_fun("Components calculating", 3)
        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(mask), self.edge_connection
                ), 20
            )
        )
        if self.smooth_border:
            report_fun("Smoothing borders", 4)
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
            report_fun("Calculation done", 5)
        return resp

    def set_parameters(self, image, threshold, minimum_size, exclude_mask, close_holes, smooth_border, use_gauss,
                       close_holes_size, smooth_border_radius, gauss_radius, side_connection):
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
        self.edge_connection = not side_connection

    def get_info_text(self):
        return ""


class AutoThresholdAlgorithm(StackAlgorithm):
    """
    :type segmentation: np.ndarray
    """
    def __init__(self):
        super(AutoThresholdAlgorithm, self).__init__()
        self.threshold = None
        self.minimum_size = None
        self.sizes = None
        self.use_gauss = False
        self.gauss_radius = 0
        self.close_holes = False
        self.close_holes_size = 0
        self.smooth_border = False
        self.smooth_border_radius = 0
        self.suggested_size = 0
        self.edge_connection = True

    def calculation_run(self, report_fun):
        if self.use_gauss == "No":
            image = np.copy(self.image)
        else:
            image = self.get_gauss(self.use_gauss, self.gauss_radius)
            report_fun("Gauss done", 0)
        if self.exclude_mask is not None:
            report_fun("Components exclusion apply", 1)
            image[self.exclude_mask > 0] = 0
        report_fun("Threshold calculation", 1)
        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.ThresholdMaximumConnectedComponents(sitk_image, self.suggested_size)
        mask = sitk.GetArrayFromImage(sitk_mask)
        min_val = np.min(image[mask > 0])
        if self.threshold < min_val:
            self.threshold = min_val
            report_fun("Threshold: {}".format(self.threshold))
        mask = (image > self.threshold).astype(np.uint8)
        if self.close_holes:
            report_fun("Holes closing", 2)
            mask = close_small_holes(mask, self.close_holes_size)
        report_fun("Components calculating", 3)

        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(mask), self.edge_connection
                ), 20
            )
        )
        if self.smooth_border:
            report_fun("Smoothing borders", 4)
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
        report_fun("Calculation done", 5)
        return resp

    def set_parameters(self, image, suggested_size, threshold, minimum_size, exclude_mask, close_holes, smooth_border,
                       use_gauss, close_holes_size, smooth_border_radius, gauss_radius, side_connection):
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
        self.edge_connection = not side_connection

    def get_info_text(self):
        return ""
