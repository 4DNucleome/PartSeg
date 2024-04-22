from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from textwrap import indent
from typing import Any, Callable, Dict, MutableMapping, Optional

import numpy as np
from local_migrator import REGISTER, class_to_str

from PartSegCore.algorithm_describe_base import (
    AlgorithmDescribeBase,
    AlgorithmProperty,
    ROIExtractionProfile,
    base_model_to_algorithm_property,
)
from PartSegCore.image_operations import RadiusType
from PartSegCore.project_info import AdditionalLayerDescription
from PartSegCore.roi_info import ROIInfo
from PartSegCore.utils import BaseModel, numpy_repr
from PartSegImage import Channel, Image


def calculate_operation_radius(radius, spacing, gauss_type):
    if gauss_type == RadiusType.R2D and len(spacing) == 3:
        spacing = spacing[1:]
    base = min(spacing)
    if base != max(spacing):
        ratio = [x / base for x in spacing]
        return [radius / r for r in ratio]
    return radius


def dict_repr(dkt: MutableMapping) -> str:
    """
    calculate dict representation which use :py:func:`numpy_repr` for numpy representation.

    :param dict dkt: dict to be represented
    :return: string representation
    """
    res = []
    for k, v in dkt.items():
        if isinstance(v, MutableMapping):
            res.append(f"{k}: {dict_repr(v)}")
        elif isinstance(v, np.ndarray):
            res.append(f"{k}: {numpy_repr(v)}")
        else:
            res.append(f"{k}: {v!r}")
    return "{" + ", ".join(res) + "}"


@dataclass(frozen=True, repr=False)
class ROIExtractionResult:
    """
    Class to store results of roi extraction process.

    :ivar np.ndarray roi: Region of Interest represented as numpy array.
    :ivar ROIExtractionProfile parameters: parameters of roi extraction process.
    :ivar Dict[str,AdditionalLayerDescription] ~.additional_layers: additional layers returned by algorithm.
        Could be previewer using napari viewer.
    :ivar dict roi_annotation: Annotation for ROI. Currently displayed as tooltip
    :ivar Dict[str,np.ndarray] alternative_representation: Arrays with alternative representations of ROI.
    :ivar Optional[str] ~.file_path: information on which file roi extraction was performed.
    :ivar ROIInfo ~.roi_info: ROIInfo for current roi.
    :ivar Optional[np.ndarray] ~.points: array of points.
    """

    # TODO add alternative representation using dict mapping.
    roi: np.ndarray
    parameters: ROIExtractionProfile
    additional_layers: Dict[str, AdditionalLayerDescription] = field(default_factory=dict)
    info_text: str = ""
    roi_annotation: Dict = field(default_factory=dict)
    alternative_representation: Dict[str, np.ndarray] = field(default_factory=dict)
    file_path: Optional[str] = None
    roi_info: Optional[ROIInfo] = None
    points: Optional[np.ndarray] = None

    def __post_init__(self):
        if "ROI" in self.alternative_representation:
            raise ValueError("alternative_representation field cannot contain field with ROI key")
        for key, value in self.additional_layers.items():
            if not value.name:
                value.name = key
        if self.roi_info is None:
            object.__setattr__(
                self,
                "roi_info",
                ROIInfo(roi=self.roi, annotations=self.roi_annotation, alternative=self.alternative_representation),
            )

    def __str__(self):  # pragma: no cover
        return (
            f"ROIExtractionResult(roi=[shape: {self.roi.shape}, dtype: {self.roi.dtype},"
            f" max: {np.max(self.roi)}], parameters={self.parameters},"
            f" additional_layers={list(self.additional_layers.keys())}, info_text={self.info_text},"
            f" alternative={dict_repr(self.alternative_representation)},"
            f" roi_annotation={dict_repr(self.roi_annotation)}"
        )

    def __repr__(self):  # pragma: no cover
        return (
            f"ROIExtractionResult(roi=[shape: {self.roi.shape}, dtype: {self.roi.dtype}, "
            f"max: {np.max(self.roi)}], parameters={self.parameters}, "
            f"additional_layers={list(self.additional_layers.keys())}, info_text={self.info_text},"
            f" alternative={dict_repr(self.alternative_representation)},"
            f" roi_annotation={dict_repr(self.roi_annotation)}"
        )


SegmentationResult = ROIExtractionResult


def report_empty_fun(_x, _y):  # pragma: no cover # skipcq: PTC-W0049
    pass


class AlgorithmInfo(BaseModel, arbitrary_types_allowed=True):
    algorithm_name: str
    parameters: Any
    image: Image


class ROIExtractionAlgorithm(AlgorithmDescribeBase, ABC):
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
            f"{self.__class__.__module__}.{self.__class__.__name__}(\n"
            + indent(f"image={self.image!r},\n", " " * 4)
            + indent(f"channel={numpy_repr(self.channel)},\n{mask_info},", " " * 4)
            + indent(f"\nvalue={self.get_segmentation_profile().values!r})", " " * 4)
        )

    def clean(self):
        self.image = None
        self.segmentation = None
        self.channel = None
        self.mask = None

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
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def support_z(cls):
        raise NotImplementedError

    def set_mask(self, mask):
        """Set mask which will limit segmentation area"""
        self.mask = mask

    def calculation_run_wrap(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        try:
            return self.calculation_run(report_fun)
        except SegmentationLimitException:  # pragma: no cover
            raise
        except Exception as e:  # pragma: no cover
            parameters = self.get_segmentation_profile()
            image = self.image
            raise SegmentationException(
                AlgorithmInfo(algorithm_name=self.get_name(), parameters=parameters, image=image)
            ) from e

    @abstractmethod
    def calculation_run(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        raise NotImplementedError

    @abstractmethod
    def get_info_text(self):
        raise NotImplementedError

    def get_channel(self, channel_idx):
        if self.support_time():
            return self.image.get_data_by_axis(c=channel_idx)
        if self.image.is_time:
            raise ValueError("This algorithm do not support time data")
        if isinstance(channel_idx, int) and self.image.channels <= channel_idx:
            raise SegmentationException(
                f"Image {self.image} has only {self.image.channels} when requested channel {channel_idx}"
            )
        if isinstance(channel_idx, str) and channel_idx not in self.image.channel_names:
            raise SegmentationException(
                f"Image {self.image} has only {self.image.channel_names} when requested channel '{channel_idx}'"
            )
        return self.image.get_data_by_axis(c=channel_idx, t=0)

    def set_image(self, image):
        self.image = image
        self.channel = None
        self.mask = None

    def set_parameters(self, _params=None, **kwargs):
        # FIXME when drop python 3.7 use positional only argument
        if _params is not None:
            if isinstance(_params, dict):
                kwargs = _params
            else:
                self.new_parameters = _params
                return
        if self.__new_style__:
            kwargs = REGISTER.migrate_data(class_to_str(self.__argument_class__), {}, kwargs)
            self.new_parameters = self.__argument_class__(**kwargs)  # pylint: disable=not-callable
            return

        base_names = [x.name for x in self.get_fields() if isinstance(x, AlgorithmProperty)]
        if set(base_names) != set(kwargs.keys()):
            missed_arguments = ", ".join(set(base_names).difference(set(kwargs.keys())))
            additional_arguments = ", ".join(set(kwargs.keys()).difference(set(base_names)))
            raise ValueError(f"Missed arguments {missed_arguments}; Additional arguments: {additional_arguments}")
        self.new_parameters = deepcopy(kwargs)

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile(name="", algorithm=self.get_name(), values=deepcopy(self.new_parameters))

    @staticmethod
    def get_steps_num():
        """Return number of algorithm steps if your algorithm report progress, else should return 0"""
        return 0

    @classmethod
    def get_channel_parameter_name(cls):
        if cls.__new_style__:
            fields = base_model_to_algorithm_property(cls.__argument_class__)
        else:
            fields = cls.get_fields()
        for el in fields:
            if el.value_type == Channel:
                return el.name
        raise ValueError("No channel defined")


class SegmentationLimitException(Exception):
    pass


class SegmentationException(Exception):
    pass


SegmentationAlgorithm = ROIExtractionAlgorithm  # rename backward compatibility
