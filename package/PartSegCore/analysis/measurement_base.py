import sys
import typing
from abc import ABC
from enum import Enum
from typing import Any, Dict, Optional, Set, Union

import numpy as np
from pydantic import BaseModel, Field
from sympy import Symbol, symbols

from PartSegImage.image import Spacing

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmDescribeNotFound
from ..channel_class import Channel
from ..class_generator import enum_register
from ..class_register import register_class
from ..universal_const import Units


class PerComponent(Enum):
    No = 1
    Yes = 2
    Mean = 3

    def __str__(self):
        return self.name.replace("_", " ")


class AreaType(Enum):
    ROI = 1
    Mask = 2
    Mask_without_ROI = 3

    def __str__(self):
        return self.name.replace("_", " ")


enum_register.register_class(AreaType)
enum_register.register_class(PerComponent)


def _migrate_leaf_dict(dkt):
    new_dkt = dkt.copy()
    new_dkt["parameter_dict"] = new_dkt.pop("dict")
    replace_name_dict = {
        "Moment of inertia": "Moment",
        "Components Number": "Components number",
        "Pixel Brightness Sum": "Pixel brightness sum",
        "Longest main axis length": "First principal axis length",
        "Middle main axis length": "Second principal axis length",
        "Shortest main axis length": "Third principal axis length",
        "split on part volume": "distance splitting volume",
        "split on part pixel brightness sum": "distance splitting pixel brightness sum",
        "Rim Volume": "rim volume",
        "Rim Pixel Brightness Sum": "rim pixel brightness sum",
        "segmentation distance": "ROI distance",
    }
    if new_dkt["name"] in replace_name_dict:
        new_dkt["name"] = replace_name_dict[new_dkt["name"]]

    return new_dkt


@register_class(
    version="0.0.0",
    old_paths=[
        "PartSeg.utils.analysis.statistics_calculation.Leaf",
        "PartSeg.utils.analysis.measurement_base.Leaf",
        "segmentation_analysis.statistics_calculation.Leaf",
    ],
    migrations=[("0.0.1", _migrate_leaf_dict)],
)
class Leaf(BaseModel):
    """
    Class for describe calculation of basic measurement

    :ivar str name: node name of method used to calculate
    :ivar dict parameter_dict: additional parameters of method
    :ivar float power: power to be applied to result of calculation methods
    :ivar AreaType area: which type of ROI should be used for calculation
    :ivar PerComponent per_component: if value should be calculated per component or for whole roi set
    :ivar Channel channel: probably not used TODO Check
    """

    name: str
    parameter_dict: typing.Any = Field(default_factory=dict)
    power: float = 1.0
    area: Optional[AreaType] = None
    per_component: Optional[PerComponent] = None
    channel: Optional[Channel] = None

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        """
        Get set with number of channels needed for calculate this measurement

        :param measurement_dict: dict with all measurementh method.
        :return: set of channels num
        """
        resp = set()
        if self.channel is not None and self.channel >= 0:
            resp.add(self.channel)
        try:
            measurement_method = measurement_dict[self.name]
            for el in measurement_method.get_fields():
                if isinstance(el, str):
                    continue
                if issubclass(el.value_type, Channel):
                    if isinstance(self.parameter_dict, dict):
                        if el.name in self.parameter_dict:
                            resp.add(self.parameter_dict[el.name])
                    else:
                        if hasattr(self.parameter_dict, el.name):
                            resp.add(getattr(self.parameter_dict, el.name))
        except KeyError as e:
            raise AlgorithmDescribeNotFound(self.name) from e
        return resp

    def _parameters_string(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> str:
        if len(self.parameter_dict) == 0 and self.channel is None:
            return ""
        arr = []
        if self.channel is not None and self.channel >= 0:
            arr.append(f"channel={self.channel+1}")
        if self.name in measurement_dict:
            measurement_method = measurement_dict[self.name]
            fields_dict = measurement_method.get_fields_dict()
            arr.extend(f"{fields_dict[k].user_name}={v}" for k, v in self.parameter_dict.items())
        else:
            arr.extend(f"{k.replace('_', ' ')}={v}" for k, v in self.parameter_dict.items())
        return "[" + ", ".join(arr) + "]"

    def _plugin_info(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> str:
        if self.name not in measurement_dict:
            return ""
        measurement_method = measurement_dict[self.name]
        if (
            hasattr(measurement_method, "__module__")
            and measurement_method.__module__.split(".", 1)[0] != "PartSegCore"
        ):
            if getattr(sys, "frozen", False):
                return f"[{measurement_method.__module__.split('.', 2)[1]}] "
            return f"[{measurement_method.__module__.split('.', 1)[0]}] "
        return ""

    def pretty_print(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> str:
        resp = self.name
        if self.area is not None:
            resp = str(self.area) + " " + resp
        resp = self._plugin_info(measurement_dict) + resp
        if self.per_component is not None:
            if self.per_component == PerComponent.Yes:
                resp += " per component "
            elif self.per_component == PerComponent.Mean:
                resp += " mean component "
        resp += self._parameters_string(measurement_dict)
        if self.power != 1.0:
            resp += f" to the power {self.power}"
        return resp

    def __str__(self):  # pragma: no cover
        return self.pretty_print({})

    def get_unit(self, ndim) -> Symbol:
        from PartSegCore.analysis import MEASUREMENT_DICT

        method = MEASUREMENT_DICT[self.name]
        if self.power != 1:
            return method.get_units(ndim) ** self.power
        return method.get_units(ndim)

    def is_per_component(self) -> bool:
        return self.per_component == PerComponent.Yes

    def need_mask(self):
        return self.area in [AreaType.Mask, AreaType.Mask_without_ROI]


def replace(self, **kwargs) -> Leaf:
    for key in list(kwargs.keys()):
        if key == "power":
            continue
        if not hasattr(self, key):
            raise ValueError(f"Unknown parameter {key}")
        if getattr(self, key) is not None and (key != "parameter_dict" or self.parameter_dict):
            del kwargs[key]

    return self.copy(update=kwargs)


Leaf.replace_ = replace


@register_class(
    old_paths=[
        "PartSeg.utils.analysis.statistics_calculation.Node",
        "PartSeg.utils.analysis.measurement_base.Node",
        "segmentation_analysis.statistics_calculation.Node",
    ]
)
class Node(BaseModel):
    left: Union["Node", Leaf]
    op: str
    right: Union["Node", Leaf]

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        return self.left.get_channel_num(measurement_dict) | self.right.get_channel_num(measurement_dict)

    def __str__(self):  # pragma: no cover
        left_text = "(" + str(self.left) + ")" if isinstance(self.left, Node) else str(self.left)
        right_text = "(" + str(self.right) + ")" if isinstance(self.right, Node) else str(self.right)
        return left_text + self.op + right_text

    def pretty_print(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> str:  # pragma: no cover
        left_text = (
            "(" + self.left.pretty_print(measurement_dict) + ")"
            if isinstance(self.left, Node)
            else self.left.pretty_print(measurement_dict)
        )
        right_text = (
            "(" + self.right.pretty_print(measurement_dict) + ")"
            if isinstance(self.right, Node)
            else self.right.pretty_print(measurement_dict)
        )
        return left_text + self.op + right_text

    def get_unit(self, ndim) -> Symbol:
        if self.op == "/":
            return self.left.get_unit(ndim) / self.right.get_unit(ndim)
        raise ValueError(f"Unknown operator '{self.op}'")

    def is_per_component(self) -> bool:
        return self.left.is_per_component() or self.right.is_per_component()

    def need_mask(self):
        return self.left.need_mask() or self.right.need_mask()


Node.update_forward_refs()


@register_class(
    old_paths=[
        "PartSeg.utils.analysis.statistics_calculation.StatisticEntry",
        "PartSeg.utils.analysis.measurement_base.StatisticEntry",
        "segmentation_analysis.statistics_calculation.StatisticEntry",
    ]
)
class MeasurementEntry(BaseModel):
    name: str
    calculation_tree: Union[Node, Leaf]

    def get_unit(self, unit: Units, ndim) -> str:
        return str(self.calculation_tree.get_unit(ndim)).format(str(unit))

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        return self.calculation_tree.get_channel_num(measurement_dict)


class MeasurementMethodBase(AlgorithmDescribeBase, ABC):
    """
    This is base class For all measurement calculation classes
    based on text_info[0] the measurement name wil be generated, based_on text_info[1] the description is generated
    """

    __argument_class__ = BaseModel

    text_info = "", ""

    need_class_method = [
        "get_description",
        "is_component",
        "calculate_property",
        "get_starting_leaf",
        "get_units",
        "need_channel",
    ]

    @classmethod
    def get_name(cls) -> str:
        return str(cls.get_starting_leaf().name)

    @classmethod
    def get_description(cls) -> str:
        """Measurement long description"""
        return "" if isinstance(cls.text_info, str) else cls.text_info[1]

    @classmethod
    def is_component(cls) -> bool:
        """Return information if Need information about components"""
        return False

    @staticmethod
    def calculate_property(
        # image: Image,
        channel: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        voxel_size: Spacing,
        result_scalar: float,
        roi_alternative: Dict[str, np.ndarray],
        roi_annotation: Dict[int, Any],
        **kwargs,
    ):
        """
        Main function for calculating measurement

        :param channel: main channel selected for measurement
        :param channel_{i}: for channel requested using :py:meth:`get_fields`
            ``AlgorithmProperty("channel", "Channel", 0, value_type=Channel)``
        :param area_array: array representing current area returned by :py:meth:`area_type`
        :param roi: array representing roi
        :param mask: array representing mask (upper level roi)
        :param voxel_size: size of single voxel in meters
        :param result_scalar: scalar to get proper units in result
        :param roi_alternative: dict with alternative roi representation (for plugin specific mapping)
        :param roi_annotation: dict with roi annotations (for plugin specific mapping)

        List incomplete.
        """
        raise NotImplementedError()

    @classmethod
    def get_starting_leaf(cls) -> Leaf:
        """This leaf is put on default list"""
        return Leaf(name=cls._display_name())

    @classmethod
    def _display_name(cls):
        return cls.text_info if isinstance(cls.text_info, str) else cls.text_info[0]

    @classmethod
    def get_units(cls, ndim) -> symbols:
        """Return units for measurement. They are shown to user"""
        raise NotImplementedError()

    @classmethod
    def need_channel(cls):
        """if need image data"""
        return False

    @staticmethod
    def area_type(area: AreaType):
        """Map chosen area type to proper area type. Allow to correct Area type."""
        return area

    @staticmethod
    def need_full_data():
        return False
