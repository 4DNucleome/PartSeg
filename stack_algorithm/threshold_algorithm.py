import numpy as np
import SimpleITK as sitk
from utils import bisect
from .segment import close_small_holes, opening

class ThresholdPreview(object):
    def __init__(self):
        pass

    @staticmethod
    def execute(image, threshold, exclude_mask):
        if exclude_mask is None:
            return (image > threshold).astype(np.uint8)
        else:
            res = (image > threshold).astype(exclude_mask.dtype())
            res[exclude_mask > 0] = 0
            res[res > 0] = image.max() + 1
            res[exclude_mask > 0] = exclude_mask[exclude_mask > 0]
            return res


class ThresholdAlgorithm(object):
    """
    :type segmentation: np.ndarray
    """
    def __init__(self):
        self.image = None
        self.threshold = None
        self.minimum_size = None
        self.segmentation = None
        self.sizes = None

    def execute(self, image, threshold, minimum_size, exclude_mask, close_holes, smooth_border):
        if (image is not self.image) or (threshold != self.threshold) or exclude_mask is not None:
            mask = (image > threshold).astype(np.uint8)
            if exclude_mask is not None:
                mask[exclude_mask > 0] = 0
            if close_holes:
                mask = close_small_holes(mask, 200)
            self.segmentation = sitk.GetArrayFromImage(
                sitk.RelabelComponent(
                    sitk.ConnectedComponent(
                        sitk.GetImageFromArray(mask)
                    ), 20
                )
            )
            if smooth_border:
                self.segmentation = opening(self.segmentation, 2, 20)
            self.sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.sizes[1:], minimum_size, lambda x, y: x > y)
        self.image = image
        self.threshold = threshold
        self.minimum_size = minimum_size
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0
        if exclude_mask is not None:
            resp[resp > 0] += exclude_mask.max()
            resp[exclude_mask > 0] = exclude_mask[exclude_mask > 0]
        return resp



