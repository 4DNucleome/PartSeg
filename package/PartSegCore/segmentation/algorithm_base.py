from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

from PartSegCore.channel_class import Channel
from PartSegImage import Image

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from ..image_operations import RadiusType
from ..project_info import AdditionalLayerDescription, ProjectInfoBase
from ..roi_info import ROIInfo
from ..utils import numpy_repr


def calculate_operation_radius(radius, spacing, gauss_type):
    if gauss_type == RadiusType.R2D and len(spacing) == 3:
        spacing = spacing[1:]
    base = min(spacing)
    if base != max(spacing):
        ratio = [x / base for x in spacing]
        return [radius / r for r in ratio]
    return radius


def dict_repr(dkt: dict) -> str:
    """
    calculate dict representation which use :py:func:`numpy_repr` for numpy representation.

    :param dict dkt: dict to be represented
    :return: string representation
    """
    res = []
    for k, v in dkt.items():
        if isinstance(v, dict):
            res.append(f"{k}: {dict_repr(v)}")
        elif isinstance(v, np.ndarray):
            res.append(f"{k}: {numpy_repr(v)}")
        else:
            res.append(f"{k}: {repr(v)}")
    return "{" + ", ".join(res) + "}"


@dataclass(frozen=True, repr=False)
class SegmentationResult:
    roi: np.ndarray
    parameters: ROIExtractionProfile
    additional_layers: Dict[str, AdditionalLayerDescription] = field(default_factory=dict)
    info_text: str = ""
    roi_annotation: Dict = field(default_factory=dict)
    alternative_representation: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def roi_info(self) -> ROIInfo:
        return ROIInfo(roi=self.roi, annotations=self.roi_annotation, alternative=self.alternative_representation)

    def __post_init__(self):
        if "ROI" in self.alternative_representation:
            raise ValueError("alternative_representation field cannot contain field with ROI key")
        for key, value in self.additional_layers.items():
            if value.name == "":
                value.name = key

    def __str__(self):  # pragma: no cover
        return (
            f"SegmentationResult(roi=[shape: {self.roi.shape}, dtype: {self.roi.dtype},"
            f" max: {np.max(self.roi)}], parameters={self.parameters},"
            f" additional_layers={list(self.additional_layers.keys())}, info_text={self.info_text},"
            f" alternative={dict_repr(self.alternative_representation)},"
            f" roi_annotation={dict_repr(self.roi_annotation)}"
        )

    def __repr__(self):  # pragma: no cover
        return (
            f"SegmentationResult(roi=[shape: {self.roi.shape}, dtype: {self.roi.dtype}, "
            f"max: {np.max(self.roi)}], parameters={self.parameters}, "
            f"additional_layers={list(self.additional_layers.keys())}, info_text={self.info_text},"
            f" alternative={dict_repr(self.alternative_representation)},"
            f" roi_annotation={dict_repr(self.roi_annotation)}"
        )


def report_empty_fun(_x, _y):  # pragma: no cover
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
        self._mask: Optional[np.ndarray] = None
        self.new_parameters: Dict[str, Any] = {}

    def __repr__(self):  # pragma: no cover
        if self.mask is None:
            mask_info = "mask=None"
        elif isinstance(self.mask, np.ndarray):
            mask_info = (
                f"mask_dtype={self.mask.dtype}, mask_shape={self.mask.shape}, mask_unique={np.unique(self.mask)}"
            )
        else:
            mask_info = f"mask={self.mask}"
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}(image={repr(self.image)}, "
            f"channel={self.channel} {mask_info}, value={self.get_segmentation_profile().values})"
        )

    def clean(self):
        self.image = None
        self.segmentation = None
        self.channel = None
        self.mask = None

    @staticmethod
    def single_channel():
        """Check if algorithm run on single channel"""
        return True

    @property
    def mask(self) -> Optional[np.ndarray]:
        if self._mask is not None and not self.support_time():
            return self.image.clip_array(self._mask, t=0)
        return self._mask

    @mask.setter
    def mask(self, val: Optional[np.ndarray]):
        if val is None:
            self._mask = None
            return
        self._mask = self.image.fit_mask_to_image(val)

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

    def calculation_run_wrap(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        try:
            return self.calculation_run(report_fun)
        except SegmentationLimitException:  # pragma: no cover
            raise
        except Exception:  # pragma: no cover
            parameters = self.get_segmentation_profile()
            image = self.image
            raise SegmentationException(self.get_name(), parameters, image)

    @abstractmethod
    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        raise NotImplementedError()

    @classmethod
    def segment_project(cls, project: ProjectInfoBase, parameters: dict) -> SegmentationResult:
        """

        :param ProjectInfoBase project:
        :param dict parameters:
        :return:
        :rtype:
        """
        instance = cls()
        instance.set_image(project.image)
        instance.set_mask(project.mask)
        instance.set_parameters(**parameters)
        return instance.calculation_run(report_empty_fun)

    @abstractmethod
    def get_info_text(self):
        raise NotImplementedError()

    def get_channel(self, channel_idx):
        if self.support_time():
            return self.image.get_data_by_axis(c=channel_idx)
        if self.image.shape[self.image.time_pos] != 1:
            raise ValueError("This algorithm do not support time data")
        if self.image.channels <= channel_idx:
            raise SegmentationException(
                f"Image {self.image} has only {self.image.channels} when requested channel {channel_idx}"
            )
        return self.image.get_data_by_axis(c=channel_idx, t=0)

    def set_image(self, image):
        self.image = image
        self.channel = None
        self._mask = None

    def set_parameters(self, **kwargs):
        base_names = [x.name for x in self.get_fields() if isinstance(x, AlgorithmProperty)]
        if set(base_names) != set(kwargs.keys()):
            missed_arguments = ", ".join(set(base_names).difference(set(kwargs.keys())))
            additional_arguments = ", ".join(set(kwargs.keys()).difference(set(base_names)))
            raise ValueError(f"Missed arguments {missed_arguments}; Additional arguments: {additional_arguments}")
        self.new_parameters = deepcopy(kwargs)

    @abstractmethod
    def get_segmentation_profile(self) -> ROIExtractionProfile:
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


class SegmentationException(Exception):
    pass
