import inspect
import typing
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum

from PartSegCore.channel_class import Channel


class AlgorithmDescribeNotFound(Exception):
    """
    When algorithm description not found
    """


class AlgorithmProperty:
    """
    This class is used to verbose describe algorithm parameters

    :param str name: name of parameter used in code
    :param str user_name: name presented to user in interface
    :param default_value: initial value which be used during interface generation
    :param str help_text: toll tip presented to user when keep mouse over widget
    :type value_type: type
    """

    def __init__(
        self,
        name: str,
        user_name: str,
        default_value: typing.Union[str, int, float, object],
        options_range=None,
        single_steep=None,
        possible_values=None,
        value_type=None,
        help_text="",
        per_dimension=False,
        **kwargs,
    ):
        if "property_type" in kwargs:
            warnings.warn("property_type is deprecated, use value_type instead", DeprecationWarning, stacklevel=2)
            value_type = kwargs["property_type"]
            del kwargs["property_type"]
        if len(kwargs) != 0:
            raise ValueError(", ".join(kwargs.keys()) + " are not expected")

        self.name = name
        self.user_name = user_name
        if isinstance(possible_values, list):
            self.value_type = list
        elif value_type is not None:
            self.value_type = value_type
        else:
            self.value_type = type(default_value)
        self.default_value = default_value
        self.range = options_range
        self.possible_values = possible_values
        self.single_step = single_steep
        self.help_text = help_text
        self.per_dimension = per_dimension
        if self.value_type is list and default_value not in possible_values:
            raise ValueError(f"default_value ({default_value}) should be one of possible values ({possible_values}).")

    def __repr__(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}(name='{self.name}',"
            f" user_name='{self.user_name}', "
            + f"default_value={self.default_value}, type={self.value_type}, range={self.range},"
            f"possible_values={self.possible_values})"
        )


class AlgorithmDescribeBase(ABC):
    """
    This is abstract class for all algorithm exported to user interface.
    Based on get_name and get_fields methods the interface will be generated
    For each group of algorithm base abstract class will add additional methods
    """

    @classmethod
    def get_doc_from_fields(cls):
        resp = "{\n"
        for el in cls.get_fields():
            if isinstance(el, AlgorithmProperty):
                resp += f"  {el.name}: {el.value_type} - "
                if el.help_text:
                    resp += el.help_text
                resp += f"(default values: {el.default_value})\n"
        resp += "}\n"
        return resp

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Algorithm name. It will be used during interface generating and in registering
        to proper :py:class:`PartSeg.PartSegCore.algorithm_describe_base.Register`.

        :return: name of algorithm
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        """
        This function return list of parameters needed by algorithm. It is used for generate form in User Interface

        :return: list of algorithm parameters and comments
        """
        raise NotImplementedError()

    @classmethod
    def get_fields_dict(cls) -> typing.Dict[str, AlgorithmProperty]:
        return {v.name: v for v in cls.get_fields() if isinstance(v, AlgorithmProperty)}

    @classmethod
    def get_default_values(cls):
        result = {}
        for el in cls.get_fields():
            if isinstance(el, AlgorithmProperty):
                if issubclass(el.value_type, AlgorithmDescribeBase):
                    result[el.name] = {
                        "name": el.default_value,
                        "values": el.possible_values[el.default_value].get_default_values(),
                    }
                else:
                    result[el.name] = el.default_value
        return result


def is_static(fun):
    args = inspect.getfullargspec(fun).args
    if len(args) == 0:
        return True
    return args[0] != "self"


AlgorithmType = typing.TypeVar("AlgorithmType", bound=type(AlgorithmDescribeBase))


class Register(OrderedDict, typing.Generic[AlgorithmType]):
    """
    Dict used for register :class:`.AlgorithmDescribeBase` classes.
    All registers from `PartSeg.PartSegCore.register` are this
    :param class_methods: list of method which should be implemented as class method it will be checked during add
    as args or with :meth:`.Register.register`  method
    :param methods: list of method which should be instance method
    """

    def __init__(self, *args: AlgorithmType, class_methods=None, methods=None, suggested_base_class=None, **kwargs):
        """
        :param class_methods: list of method which should be class method
        :param methods: list of method which should be instance method
        :param kwargs: elements passed to OrderedDict constructor (may be initial elements). I suggest to not use this.
        """
        super().__init__(**kwargs)
        self.suggested_base_class = suggested_base_class
        self.class_methods = (
            list(class_methods) if class_methods else getattr(suggested_base_class, "need_class_method", [])
        )
        self.methods = list(methods) if methods else getattr(suggested_base_class, "need_method", [])
        for el in args:
            self.register(el)

    def values(self) -> typing.Iterable[AlgorithmType]:  # pylint: disable=W0235
        # noinspection PyTypeChecker
        return super().values()

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and isinstance(other, Register)
            and self.class_methods == other.class_methods
            and self.methods == other.methods
            and self.suggested_base_class == other.suggested_base_class
        )

    def __getitem__(self, item) -> AlgorithmType:  # pylint: disable=W0235
        return super().__getitem__(item)

    def register(self, value: AlgorithmType, replace=False):
        """
        Function for registering :class:`.AlgorithmDescribeBase` based algorithms
        :param value: algorithm to register
        :param replace: replace existing algorithm, be patient with this
        """
        self.check_function(value, "get_name", True)
        try:
            name = value.get_name()
        except NotImplementedError:
            raise ValueError(f"Class {value} need to implement get_name class method")
        if name in self and not replace:
            raise ValueError(f"Object with this name: {name} already exist and register is not in replace mode")
        if not isinstance(name, str):
            raise ValueError(f"Function get_name of class {value} need return string not {type(name)}")
        self[name] = value

    @staticmethod
    def check_function(ob, function_name, is_class):
        fun = getattr(ob, function_name, None)
        if not is_class and not inspect.isfunction(fun):
            raise ValueError(f"Class {ob} need to define method {function_name}")
        if is_class and not inspect.ismethod(fun) and not is_static(fun):
            raise ValueError(f"Class {ob} need to define classmethod {function_name}")

    def __setitem__(self, key: str, value: AlgorithmType):
        if not issubclass(value, AlgorithmDescribeBase):
            raise ValueError(
                f"Class {value} need to inherit from " f"{AlgorithmDescribeBase.__module__}.AlgorithmDescribeBase"
            )
        self.check_function(value, "get_name", True)
        self.check_function(value, "get_fields", True)
        try:
            val = value.get_name()
        except NotImplementedError:
            raise ValueError(f"Method get_name of class {value} need to be implemented")
        if not isinstance(val, str):
            raise ValueError(f"Function get_name of class {value} need return string not {type(val)}")
        if key != val:
            raise ValueError("Object need to be registered under name returned by gey_name function")
        try:
            val = value.get_fields()
        except NotImplementedError:
            raise ValueError(f"Method get_fields of class {value} need to be implemented")
        if not isinstance(val, list):
            raise ValueError(f"Function get_fields of class {value} need return list not {type(val)}")
        for el in self.class_methods:
            self.check_function(value, el, True)
        for el in self.methods:
            self.check_function(value, el, False)

        super().__setitem__(key, value)

    def get_default(self) -> str:
        """
        Calculate default algorithm name for given dict.

        :return: name of algorithm
        """
        try:
            return next(iter(self.keys()))
        except StopIteration:
            raise ValueError("Register does not contain any algorithm.")


class ROIExtractionProfile:
    """

    :ivar str ~.name: name for segmentation profile
    :ivar str ~.algorithm: Name of algorithm
    :ivar dict ~.values: algorithm parameters
    """

    def __init__(self, name: str, algorithm: str, values: dict):
        self.name = name
        self.algorithm = algorithm
        self.values = values

    def pretty_print(self, algorithm_dict):
        try:
            algorithm = algorithm_dict[self.algorithm]
        except KeyError:
            return str(self)
        if self.name in {"", "Unknown"}:
            return (
                "ROI extraction profile\nAlgorithm: "
                + self.algorithm
                + "\n"
                + self._pretty_print(self.values, algorithm.get_fields_dict())
            )
        return (
            "ROI extraction profile name: "
            + self.name
            + "\nAlgorithm: "
            + self.algorithm
            + "\n"
            + self._pretty_print(self.values, algorithm.get_fields_dict())
        )

    @classmethod
    def _pretty_print(cls, values: dict, translate_dict: typing.Dict[str, AlgorithmProperty], indent=0):
        res = ""
        for k, v in values.items():
            if k not in translate_dict:
                if isinstance(v, dict):
                    res += " " * indent + f"{k}: {cls._pretty_print(v, {}, indent + 2)}\n"
                else:
                    res += " " * indent + f"{k}: {v}\n"
                continue
            desc = translate_dict[k]
            res += " " * indent + desc.user_name + ": "
            if issubclass(desc.value_type, Channel):
                res += str(Channel(v))
            elif issubclass(desc.value_type, AlgorithmDescribeBase):
                res += desc.possible_values[v["name"]].get_name()
                if v["values"]:
                    res += "\n"
                    res += cls._pretty_print(v["values"], desc.possible_values[v["name"]].get_fields_dict(), indent + 2)
            elif isinstance(v, dict):
                res += cls._pretty_print(v, {}, indent + 2)
            else:
                res += str(v)
            res += "\n"
        return res[:-1]

    @classmethod
    def print_dict(cls, dkt, indent=0, name: str = ""):
        if isinstance(dkt, Enum):
            return dkt.name
        if not isinstance(dkt, dict):
            # FIXME update in future method of proper printing channel number
            if name.startswith("channel") and isinstance(dkt, int):
                return dkt + 1
            return dkt
        return "\n" + "\n".join(
            " " * indent + f"{k.replace('_', ' ')}: {cls.print_dict(v, indent + 2, k)}" for k, v in dkt.items()
        )

    def __str__(self):
        return (
            "Segmentation profile name: " + self.name + "\nAlgorithm: " + self.algorithm + self.print_dict(self.values)
        )

    def __repr__(self):
        return f"SegmentationProfile(name={self.name}, algorithm={repr(self.algorithm)}, values={self.values})"
