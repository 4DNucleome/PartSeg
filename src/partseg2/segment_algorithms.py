from project_utils.algorithm_base import SegmentationAlgorithm
from project_utils.image_operations import gaussian
import numpy as np
import SimpleITK as sitk
from project_utils import bisect


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
        self.progress_signal.emit("Components calculating", 3)
        self.segmentation = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(
                    sitk.GetImageFromArray(mask)
                ), 20
            )
        )

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
