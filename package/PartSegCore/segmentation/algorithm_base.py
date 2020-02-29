from abc import ABC, abstractmethod
from typing import NamedTuple, Union, Callable, Optional

import numpy as np

from PartSegCore.channel_class import Channel
from PartSegImage import Image

from ..image_operations import RadiusType
from ..algorithm_describe_base import AlgorithmDescribeBase, SegmentationProfile


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
    info_text: str = ""


def report_empty_fun(_x, _y):
    pass


class SegmentationAlgorithm(AlgorithmDescribeBase, ABC):
    """
    Base class for all segmentation algorithm.

    :ivar Image ~.image: Image to process
    :ivar numpy.ndarray ~.channel: selected channel
    :ivar numpy.ndarray ~.segmentation: final segmentation
    :ivar numpy.ndarray ~.mask: mask limiting segmentation area
    """

    def __init__(self):
        super().__init__()
        self.image: Optional[Image] = None
        self.channel = None
        self.segmentation = None
        self.mask = None

    def clean(self):
        self.image = None
        self.segmentation = None
        self.channel = None
        self.mask = None

    @staticmethod
    def single_channel():
        """Check if algorithm run on single channel"""
        return True

    @classmethod
    @abstractmethod
    def support_time(cls):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def support_z(cls):
        raise NotImplementedError()

    def set_mask(self, mask):
        """Set mask which will limit segmentation area"""
        self.mask = mask

    @abstractmethod
    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        raise NotImplementedError()

    @abstractmethod
    def get_info_text(self):
        raise NotImplementedError()

    def get_channel(self, channel_idx):
        channel = self.image.get_channel(channel_idx)
        if channel.shape[0] != 1:
            raise ValueError("This algorithm do not support time data")
        return channel[0]

    def set_image(self, image):
        self.image = image
        self.channel = None

    @abstractmethod
    def set_parameters(self, **kwargs):
        """Set parameters for next segmentation."""
        raise NotImplementedError()

    @abstractmethod
    def get_segmentation_profile(self) -> SegmentationProfile:
        """Get parameters seated by :py:meth:`set_parameters` method."""
        raise NotImplementedError()

    @staticmethod
    def get_steps_num():
        """Return number of algorithm steps if your algorithm report progress, else should return 0"""
        return 0

    @classmethod
    def get_channel_parameter_name(cls):
        for el in cls.get_fields():
            if el.value_type == Channel:
                return el.name
        raise ValueError("No channel defined")


class SegmentationLimitException(Exception):
    pass
