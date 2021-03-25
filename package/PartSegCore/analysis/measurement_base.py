from abc import ABC
from enum import Enum
from typing import Dict, Optional, Set, Union

from sympy import symbols

from ..algorithm_describe_base import AlgorithmDescribeBase, AlgorithmDescribeNotFound
from ..channel_class import Channel
from ..class_generator import BaseSerializableClass, enum_register
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


class Leaf(BaseSerializableClass):
    """
    Class for describe calculation of basic measurement

    :ivar str name: node name of method used to calculate
    :ivar dict dict: additional parameters of method
    :ivar float power: power to be applied to result of calculation methods
    :ivar AreaType area: which type of ROI should be used for calculation
    :ivar PerComponent per_component: if value should be calculated per component or for whole roi set
    :ivar Channel channel: probably not used TODO Check
    """

    # noinspection PyMissingConstructor,PyShadowingBuiltins, PyUnusedLocal
    # pylint: disable=W0104,W0221,W0622
    def __init__(
        self,
        name: str,
        dict: Dict = None,
        power: float = 1.0,
        area: Optional[AreaType] = None,
        per_component: Optional[PerComponent] = None,
        channel: Optional[Channel] = None,
    ):
        ...

    name: str
    dict: Dict = dict()
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
                if issubclass(el.value_type, Channel) and el.name in self.dict:
                    resp.add(self.dict[el.name])
        except KeyError:
            raise AlgorithmDescribeNotFound(self.name)
        return resp

    def pretty_print(self, measurement_dict: Dict[str, "MeasurementMethodBase"]):  # pragma: no cover
        resp = self.name
        if self.area is not None:
            resp = str(self.area) + " " + resp
        if self.per_component is not None and self.per_component == PerComponent.Yes:
            resp += " per component "
        if self.per_component is not None and self.per_component == PerComponent.Mean:
            resp += " mean component "
        if len(self.dict) != 0 or self.channel is not None:
            resp += "["
            arr = []
            if self.channel is not None and self.channel >= 0:
                arr.append(f"channel={self.channel+1}")
            if len(self.dict) > 0:
                try:
                    measurement_method = measurement_dict[self.name]
                    fields_dict = measurement_method.get_fields_dict()
                    for k, v in self.dict.items():
                        arr.append(f"{fields_dict[k].user_name}={v}")
                except KeyError:
                    arr.append("class not found")
            resp += ", ".join(arr)
            resp += "]"
        if self.power != 1.0:
            resp += f" to the power {self.power}"
        return resp

    def __str__(self):  # pragma: no cover
        resp = self.name
        if self.area is not None:
            resp = str(self.area) + " " + resp
        if self.per_component is not None and self.per_component == PerComponent.Yes:
            resp += " per component "
        if len(self.dict) != 0 or self.channel is not None:
            resp += "["
            arr = []
            if self.channel is not None and self.channel >= 0:
                arr.append(f"channel={self.channel}")
            for k, v in self.dict.items():
                arr.append(f"{k.replace('_', ' ')}={v}")
            resp += ", ".join(arr)
            resp += "]"
        if self.power != 1.0:
            resp += f" to the power {self.power}"
        return resp

    def get_unit(self, ndim):
        from PartSegCore.analysis import MEASUREMENT_DICT

        method = MEASUREMENT_DICT[self.name]
        if self.power != 1:
            return method.get_units(ndim) ** self.power
        return method.get_units(ndim)

    def is_per_component(self):
        return self.per_component == PerComponent.Yes


def replace(self, **kwargs):
    for key in list(kwargs.keys()):
        if key == "power":
            continue
        if not hasattr(self, key):
            raise ValueError(f"Unknown parameter {key}")
        if getattr(self, key) is not None:
            del kwargs[key]

    dkt = self.asdict()
    dkt.update(kwargs)
    return Leaf(**dkt)


Leaf.replace_ = replace


class Node(BaseSerializableClass):
    left: Union["Node", Leaf]
    op: str
    right: Union["Node", Leaf]

    # noinspection PyMissingConstructor, PyUnusedLocal
    # pylint: disable=W0104
    def __init__(self, left: Union["Node", Leaf], op: str, right: Union["Node", Leaf]):
        ...

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        return self.left.get_channel_num(measurement_dict) | self.right.get_channel_num(measurement_dict)

    def __str__(self):  # pragma: no cover
        left_text = "(" + str(self.left) + ")" if isinstance(self.left, Node) else str(self.left)
        right_text = "(" + str(self.right) + ")" if isinstance(self.right, Node) else str(self.right)
        return left_text + self.op + right_text

    def pretty_print(self, measurement_dict: Dict[str, "MeasurementMethodBase"]):  # pragma: no cover
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

    def get_unit(self, ndim):
        if self.op == "/":
            return self.left.get_unit(ndim) / self.right.get_unit(ndim)
        raise ValueError(f"Unknown operator '{self.op}'")

    def is_per_component(self):
        return self.left.is_per_component() or self.right.is_per_component()


class MeasurementEntry(BaseSerializableClass):
    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    __old_names__ = "StatisticEntry"

    # noinspection PyMissingConstructor,PyUnusedLocal
    # pylint: disable=W0104
    def __init__(self, name: str, calculation_tree: Union[Node, Leaf]):
        ...

    name: str
    calculation_tree: Union[Node, Leaf]

    def get_unit(self, unit: Units, ndim):
        return str(self.calculation_tree.get_unit(ndim)).format(str(unit))

    def get_channel_num(self, measurement_dict: Dict[str, "MeasurementMethodBase"]) -> Set[Channel]:
        return self.calculation_tree.get_channel_num(measurement_dict)


class MeasurementMethodBase(AlgorithmDescribeBase, ABC):
    """
    This is base class For all measurement calculation classes
    based on text_info[0] the measurement name wil be generated, based_on text_info[1] the description is generated
    """

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
    def get_name(cls):
        return str(cls.get_starting_leaf().name)

    @classmethod
    def get_description(cls):
        """Measurement long description"""
        return cls.text_info[1]

    @classmethod
    def is_component(cls):
        """Return information if Need information about components"""
        return False

    @classmethod
    def get_fields(cls):
        """Additional fields needed by algorithm. like radius of dilation"""
        return []

    @staticmethod
    def calculate_property(**kwargs):
        """Main function for calculating measurement"""
        raise NotImplementedError()

    @classmethod
    def get_starting_leaf(cls) -> Leaf:
        """This leaf is putted on default list"""
        return Leaf(cls.text_info[0])

    @classmethod
    def get_units(cls, ndim) -> symbols:
        """Return units for measurement. They are shown to user"""
        raise NotImplementedError()

    @classmethod
    def need_channel(cls):
        """TBA"""
        return False

    @staticmethod
    def area_type(area: AreaType):
        """Map chosen area type to proper area type. Allow to correct Area type."""
        return area
