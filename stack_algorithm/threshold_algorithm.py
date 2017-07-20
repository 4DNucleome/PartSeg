import numpy as np
import SimpleITK as sitk
from utils import bisect


class ThresholdPreview(object):
    def __init__(self):
        pass

    @staticmethod
    def execute(image, threshold):
        return (image > threshold).astype(np.uint8)


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

    def execute(self, image, threshold, minimum_size):
        if (image is not self.image) or (threshold != self.threshold):
            mask = (image > threshold).astype(np.uint8)
            self.segmentation = sitk.GetArrayFromImage(
                sitk.RelabelComponent(
                    sitk.ConnectedComponent(
                        sitk.GetImageFromArray(mask)
                    )
                )
            )
            self.sizes = np.bincount(self.segmentation.flat)
        ind = bisect(self.sizes[1:], minimum_size, lambda x, y: x > y)
        self.image = image
        self.threshold = threshold
        self.minimum_size = minimum_size
        resp = np.copy(self.segmentation)
        resp[resp > ind] = 0
        return resp



