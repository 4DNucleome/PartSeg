import sys
from abc import ABC
from enum import Enum
from typing import Any, ClassVar, Dict, ForwardRef, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from local_migrator import REGISTER, class_to_str, register_class, rename_key
from pydantic import Field, validator
from sympy import Symbol, symbols

from PartSegCore.algorithm_describe_base import (
    AlgorithmDescribeBase,
    AlgorithmDescribeNotFound,
    base_model_to_algorithm_property,
)
from PartSegCore.universal_const import Units
from PartSegCore.utils import BaseModel
from PartSegImage import Channel
from PartSegImage.image import Spacing


@register_class(
    old_paths=[
        "PartSeg.utils.analysis.statistics_calculation.PerComponent",
        "PartSeg.utils.analysis.measurement_base.PerComponent",
        "segmentation_analysis.statistics_calculation.PerComponent",
    ]
)
class PerComponent(Enum):
    """How measurement should be calculated"""

    No = 1
    Yes = 2
    Mean = 3
    Per_Mask_component = 4

    def __str__(self):
        return self.name.replace("_", " ")


@register_class(
    old_paths=[
        "PartSeg.utils.analysis.statistics_calculation.AreaType",
        "PartSeg.utils.analysis.measurement_base.AreaType",
        "segmentation_analysis.statistics_calculation.AreaType",
    ]
)
class AreaType(Enum):
    """On which area type measurement should be calculated"""

    ROI = 1
    Mask = 2
    Mask_without_ROI = 3

    def __str__(self):
        return self.name.replace("_", " ")


def has_mask_components(component_and_mask_info: Iterable[Tuple[PerComponent, AreaType]]) -> bool:
    """Check if any measurement will return value per mask component"""
    return any(
        (cmp == PerComponent.Yes and area != AreaType.ROI) or cmp == PerComponent.Per_Mask_component
        for cmp, area in component_and_mask_info
    )


def has_roi_components(component_and_mask_info: Iterable[Tuple[PerComponent, AreaType]]) -> bool:
    """Check if any measurement will return value per ROI component"""
    return any((cmp == PerComponent.Yes and area == AreaType.ROI) for cmp, area in component_and_mask_info)


def _migrate_leaf_dict(dkt):
    from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT

    new_dkt = dkt.copy()
    new_dkt["parameter_dict"] = new_dkt.pop("dict")
    new_dkt["name"] = MEASUREMENT_DICT[new_dkt["name"]].get_name()

    return new_dkt


@register_class(
    version="0.0.2",
    old_paths=[
        "PartSeg.utils.analysis.statistics_calculation.Leaf",
        "PartSeg.utils.analysis.measurement_base.Leaf",
        "segmentation_analysis.statistics_calculation.Leaf",
    ],
    migrations=[("0.0.1", _migrate_leaf_dict), ("0.0.2", rename_key("parameter_dict", "parameters"))],
)
class Leaf(BaseModel):
    """
    Class for describe calculation of basic measurement
    """

    name: str
    parameters: Any = Field(default_factory=dict)
    power: float = 1.0
    area: Optional[AreaType] = None
    per_component: Optional[PerComponent] = None
    channel: Optional[Channel] = None

    @validator("parameters")
    def _validate_parameters(cls, v, values):  # pylint: disable=no-self-use
        if not isinstance(v, dict) or "name" not in values:
            return v
        from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT

        if values["name"] not in MEASUREMENT_DICT:
            return v

        method = MEASUREMENT_DICT[values["name"]]
        if not method.__new_style__ or not method.__argument_class__.__fields__:
            return v

        v = REGISTER.migrate_data(class_to_str(method.__argument_class__), {}, v)
        return method.__argument_class__(**v)

    @validator("per_component")
    def _validate_per_component(cls, v, values):  # pylint: disable=no-self-use
        if not isinstance(v, PerComponent) or "area" not in values or values["area"] is None:
            return v
        if v == PerComponent.Per_Mask_component and values["area"] != AreaType.ROI:
            raise ValueError("Per_Mask_component can be used only with ROI area")
        return v

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        """
        Get set with number of channels needed for calculate this measurement

        :param measurement_dict: dict with all measurementh method.
        :return: set of channels num
        """
        resp = set()
        if self.channel is not None and self.channel.value != -1:
            resp.add(self.channel)
        try:
            measurement_method = measurement_dict[self.name]
            if measurement_method.__new_style__:
                fields = base_model_to_algorithm_property(measurement_method.__argument_class__)
            else:
                fields = measurement_method.get_fields()
            for el in fields:
                if isinstance(el, str):
                    continue
                if el.value_type is Channel:
                    if isinstance(self.parameters, dict):
                        if el.name in self.parameters:
                            resp.add(Channel(self.parameters[el.name]))
                    elif hasattr(self.parameters, el.name):
                        resp.add(getattr(self.parameters, el.name))
        except KeyError as e:
            raise AlgorithmDescribeNotFound(self.name) from e
        return resp

    def _parameters_string(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> str:
        parameters = dict(self.parameters)
        if not parameters and self.channel is None:
            return ""
        arr = []
        if self.channel is not None and self.channel.value != -1:
            arr.append(f"channel={self.channel}")
        if self.name in measurement_dict:
            measurement_method = measurement_dict[self.name]
            fields_dict = measurement_method.get_fields_dict()
            arr.extend(f"{fields_dict[k].user_name}={v}" for k, v in parameters.items())
        else:
            arr.extend(f"{k.replace('_', ' ')}={v}" for k, v in parameters.items())
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
        """
        Pretty print for presentation in user interface.

        :param measurement_dict: dict with additional information used for more detailed description
        :return: string with indentation
        """

        resp = self.name
        if self.area is not None:
            resp = f"{self.area} {resp}"
        resp = self._plugin_info(measurement_dict) + resp
        if self.per_component is not None:
            if self.per_component == PerComponent.Yes:
                resp += " per component "
            elif self.per_component == PerComponent.Per_Mask_component:
                resp += " per mask component "
            elif self.per_component == PerComponent.Mean:
                resp += " mean component "
        resp += self._parameters_string(measurement_dict)
        if self.power != 1.0:
            resp += f" to the power {self.power}"
        return resp

    def __str__(self):  # pragma: no cover
        return self.pretty_print({})

    def get_unit(self, ndim: int) -> Symbol:
        """
        Return unit of selected measurement reflecting dimensionality.

        :param ndim: data dimensionality
        """
        from PartSegCore.analysis import MEASUREMENT_DICT

        method = MEASUREMENT_DICT[self.name]
        if self.power != 1:
            return method.get_units(ndim) ** self.power
        return method.get_units(ndim)

    def is_per_component(self) -> bool:
        """If measurement return list of result or single value."""
        return self.per_component in {PerComponent.Yes, PerComponent.Per_Mask_component}

    def need_mask(self) -> bool:
        """If this measurement need mast for proper calculation."""
        return (
            self.area in {AreaType.Mask, AreaType.Mask_without_ROI}
            or self.per_component is PerComponent.Per_Mask_component
        )


def replace(self, **kwargs) -> Leaf:
    for key in list(kwargs.keys()):
        if key == "power":
            continue
        if not hasattr(self, key):
            raise ValueError(f"Unknown parameter {key}")
        if getattr(self, key) is not None and (key != "parameters" or dict(self.parameters)):
            del kwargs[key]

    return self.copy(update=kwargs)


Leaf.replace_ = replace

Node = ForwardRef("Node")


@register_class(
    old_paths=[
        "PartSeg.utils.analysis.statistics_calculation.Node",
        "PartSeg.utils.analysis.measurement_base.Node",
        "segmentation_analysis.statistics_calculation.Node",
    ]
)
class Node(BaseModel):
    """
    Class for describe operation between two measurements
    """

    left: Union[Node, Leaf]
    op: str = Field(
        description="Operation to perform between left and right child. Currently only division (`/`) supported"
    )
    right: Union[Node, Leaf]

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        return self.left.get_channel_num(measurement_dict) | self.right.get_channel_num(measurement_dict)

    def __str__(self):  # pragma: no cover
        left_text = f"({self.left!s})" if isinstance(self.left, Node) else str(self.left)

        right_text = f"({self.right!s})" if isinstance(self.right, Node) else str(self.right)

        return left_text + self.op + right_text

    def pretty_print(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> str:  # pragma: no cover
        left_text = (
            f"({self.left.pretty_print(measurement_dict)})"
            if isinstance(self.left, Node)
            else self.left.pretty_print(measurement_dict)
        )

        right_text = (
            f"({self.right.pretty_print(measurement_dict)})"
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
    """Describe single measurement in measurement set"""

    name: str
    calculation_tree: Union[Node, Leaf]

    def get_unit(self, unit: Units, ndim) -> str:
        return str(self.calculation_tree.get_unit(ndim)).format(str(unit))

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        return self.calculation_tree.get_channel_num(measurement_dict)


class MeasurementMethodBase(AlgorithmDescribeBase, ABC):
    """
    This is base class For all measurement calculation classes
    based on text_info[0] the measurement name will be generated,
    based_on text_info[1] the description is generated
    """

    __argument_class__ = BaseModel

    text_info = "", ""

    need_class_method: ClassVar[List[str]] = [
        "get_description",
        "is_component",
        "calculate_property",
        "get_starting_leaf",
        "get_units",
        "need_channel",
    ]

    @classmethod
    def get_name(cls) -> str:
        """Name of measurement"""
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
        raise NotImplementedError

    @classmethod
    def get_starting_leaf(cls) -> Leaf:
        """This leaf is put on a default list"""
        if (
            hasattr(cls, "__argument_class__")
            and cls.__argument_class__ is not None
            and cls.__argument_class__ is not BaseModel
        ):
            return Leaf(name=cls._display_name(), parameters=cls.__argument_class__())
        return Leaf(name=cls._display_name())

    @classmethod
    def _display_name(cls):
        return cls.text_info if isinstance(cls.text_info, str) else cls.text_info[0]

    @classmethod
    def get_units(cls, ndim) -> symbols:
        """Return units for measurement. They are shown to user"""
        raise NotImplementedError

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
