from abc import ABC

from ..image_operations import gaussian, RadiusType
from ..algorithm_describe_base import AlgorithmDescribeBase, SegmentationProfile
from PartSeg.tiff_image import Image
from typing import NamedTuple, Union, Callable, Optional
import numpy as np


def calculate_operation_radius(radius, spacing, gauss_type):
    if gauss_type == RadiusType.R2D:
        if len(spacing) == 3:
            spacing = spacing[1:]
    base = min(spacing)
    if base != max(spacing):
        ratio = [x / base for x in spacing]
        return [radius / r for r in ratio]
    return radius


class SegmentationResult(NamedTuple):
    segmentation: np.ndarray
    parameters: SegmentationProfile
    full_segmentation: Union[np.ndarray, None] = None
    cleaned_channel: Union[np.ndarray, None] = None


def report_empty_fun(_x, _y):
    pass


class SegmentationAlgorithm(AlgorithmDescribeBase, ABC):
    channel: Optional[np.ndarray]

    def __init__(self):
        super().__init__()
        self.image: Image = None
        self.channel = None
        self.segmentation = None
        self.mask = None

    def _clean(self):
        self.image = None
        self.segmentation = None
        self.channel = None
        self.mask = None

    def single_channel(self):
        return True

    def set_mask(self, mask):
        self.mask = mask

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        raise NotImplementedError()

    def get_info_text(self):
        raise NotImplementedError()

    def get_channel(self, channel_idx):
        return self.image.get_channel(channel_idx)

    def get_gauss(self, gauss_type, gauss_radius):
        if gauss_type == RadiusType.NO:
            return self.channel
        assert isinstance(gauss_type, RadiusType)
        gauss_radius = calculate_operation_radius(gauss_radius, self.image.spacing, gauss_type)
        layer = gauss_type == RadiusType.R2D
        return gaussian(self.channel, gauss_radius, layer=layer)

    def set_image(self, image):
        self.image = image
        self.channel = None

    def set_exclude_mask(self, exclude_mask):
        """For Stack Seg - designed for mask part of image - maybe use standardize it to mask"""
        pass

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    def get_segmentation_profile(self) -> SegmentationProfile:
        raise NotImplementedError()

    @staticmethod
    def get_steps_num():
        return 0
