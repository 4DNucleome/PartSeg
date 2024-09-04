import inspect
import math
import textwrap
import typing
import warnings
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from importlib.metadata import version

from local_migrator import REGISTER, class_to_str
from packaging.version import parse as parse_version
from pydantic import BaseModel as PydanticBaseModel
from pydantic import create_model, validator
from pydantic.fields import Field, FieldInfo
from typing_extensions import Annotated

from PartSegCore.utils import BaseModel
from PartSegImage import Channel

if typing.TYPE_CHECKING:
    from pydantic.fields import ModelField

try:
    # pydantic 1
    from pydantic.fields import UndefinedType
    from pydantic.main import ModelMetaclass

    def field_serializer(_):
        def decorator(func):
            return func

        return decorator

except ImportError:
    # pydantic 2
    from pydantic import field_serializer
    from pydantic._internal._model_construction import ModelMetaclass
    from pydantic_core import PydanticUndefinedType as UndefinedType


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
        default_value: typing.Union[str, float, object],
        options_range=None,
        possible_values=None,
        value_type=None,
        help_text="",
        per_dimension=False,
        mgi_options=None,
        **kwargs,
    ):
        if "property_type" in kwargs:
            warnings.warn("property_type is deprecated, use value_type instead", DeprecationWarning, stacklevel=2)
            value_type = kwargs["property_type"]
            del kwargs["property_type"]
        if kwargs:
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
        self.range = tuple(self.value_type(x) for x in options_range) if options_range is not None else None
        self.possible_values = possible_values
        self.help_text = help_text
        self.per_dimension = per_dimension
        self.mgi_options = mgi_options if mgi_options is not None else {}
        if self.value_type is list and default_value not in possible_values:
            raise ValueError(f"default_value ({default_value}) should be one of possible values ({possible_values}).")

    def __repr__(self):
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}(name='{self.name}',"
            f" user_name='{self.user_name}', "
            f"default_value={self.default_value}, type={self.value_type}, range={self.range},"
            f"possible_values={self.possible_values})"
        )


class _GetDescriptionClass:
    __slots__ = ("_name",)

    def __init__(self):
        self._name = None

    def __set_name__(self, owner, name: str):
        if self._name is None:
            self._name = name

    def __get__(self, obj, klass):
        if klass is None:
            klass = type(obj)

        name = typing.cast(str, self._name)
        fields_dkt = {
            field.name: (
                Annotated[field.value_type, field.user_name, field.range, field.help_text],
                field.default_value,
            )
            for field in klass.get_fields()
            if not isinstance(field, str)
        }

        model = create_model(name, **fields_dkt)
        model.__qualname__ = f"{klass.__qualname__}.{name}"
        setattr(klass, name, model)
        return model


def _partial_abstractmethod(funcobj):
    funcobj.__is_partial_abstractmethod__ = True
    return funcobj


class AlgorithmDescribeBaseMeta(ABCMeta):
    def __new__(cls, name, bases, attrs, **kwargs):
        cls2 = super().__new__(cls, name, bases, attrs, **kwargs)
        if (
            not inspect.isabstract(cls2)
            and hasattr(cls2.get_fields, "__is_partial_abstractmethod__")
            and cls2.__argument_class__ is None
        ):
            raise RuntimeError("class need to have __argument_class__ set or get_fields functions defined")
        cls2.__new_style__ = getattr(cls2.get_fields, "__is_partial_abstractmethod__", False)
        return cls2


class AlgorithmDescribeBase(ABC, metaclass=AlgorithmDescribeBaseMeta):
    """
    This is abstract class for all algorithm exported to user interface.
    Based on get_name and get_fields methods the interface will be generated
    For each group of algorithm base abstract class will add additional methods
    """

    __argument_class__: typing.Optional[typing.Type[PydanticBaseModel]] = None
    __new_style__: bool

    @classmethod
    def get_doc_from_fields(cls):
        resp = "{\n"
        for el in get_fields_from_algorithm(cls):
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
        raise NotImplementedError

    @classmethod
    @_partial_abstractmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        """
        This function return list of parameters needed by algorithm. It is used for generate form in User Interface

        :return: list of algorithm parameters and comments
        """
        if hasattr(cls, "__argument_class__") and cls.__argument_class__ is not None:
            warnings.warn(
                "Class has __argument_class__ defined, one should not use get_fields",
                category=FutureWarning,
                stacklevel=2,
            )
            return base_model_to_algorithm_property(cls.__argument_class__)
        raise NotImplementedError

    @classmethod
    def _get_fields(cls):
        return base_model_to_algorithm_property(cls.__argument_class__) if cls.__new_style__ else cls.get_fields()

    @classmethod
    def get_fields_dict(cls) -> typing.Dict[str, AlgorithmProperty]:
        return {v.name: v for v in get_fields_from_algorithm(cls) if isinstance(v, AlgorithmProperty)}

    @classmethod
    def get_default_values(cls):
        if cls.__new_style__:
            return cls.__argument_class__()  # pylint: disable=not-callable
        return {
            el.name: (
                {
                    "name": el.default_value,
                    "values": el.possible_values[el.default_value].get_default_values(),
                }
                if issubclass(el.value_type, AlgorithmDescribeBase)
                else el.default_value
            )
            for el in cls.get_fields()
            if isinstance(el, AlgorithmProperty)
        }


def get_fields_from_algorithm(ald_desc: AlgorithmDescribeBase) -> typing.List[typing.Union[AlgorithmProperty, str]]:
    if ald_desc.__new_style__:
        return base_model_to_algorithm_property(ald_desc.__argument_class__)
    return ald_desc.get_fields()


def is_static(fun):
    if fun is None:
        return False
    args = inspect.getfullargspec(fun).args
    return True if len(args) == 0 else args[0] != "self"


AlgorithmType = typing.TypeVar("AlgorithmType", bound=typing.Type[AlgorithmDescribeBase])


class Register(typing.Dict, typing.Generic[AlgorithmType]):
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
        self._old_mapping = {}
        for el in args:
            self.register(el)

    def values(self) -> typing.Iterable[AlgorithmType]:
        # noinspection PyTypeChecker
        return typing.cast(typing.Iterable[AlgorithmType], super().values())

    def __eq__(self, other):
        return (
            super().__eq__(other)
            and isinstance(other, Register)
            and self.class_methods == other.class_methods
            and self.methods == other.methods
            and self.suggested_base_class == other.suggested_base_class
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, item) -> AlgorithmType:
        # FIXME add better strategy to get proper class when there is conflict of names
        try:
            return typing.cast(AlgorithmType, super().__getitem__(item))
        except KeyError:
            return typing.cast(AlgorithmType, super().__getitem__(self._old_mapping[item]))

    def __contains__(self, item):
        return super().__contains__(item) or item in self._old_mapping

    def register(
        self, value: AlgorithmType, replace: bool = False, old_names: typing.Optional[typing.List[str]] = None
    ):
        """
        Function for registering :class:`.AlgorithmDescribeBase` based algorithms
        :param value: algorithm to register
        :param replace: replace existing algorithm, be patient with
        :param old_names: list of old names for registered class
        """
        self.check_function(value, "get_name", True)
        try:
            name = value.get_name()
        except (NotImplementedError, AttributeError):
            raise ValueError(f"Class {value} need to implement classmethod 'get_name'") from None
        if name in self and not replace:
            raise ValueError(
                f"Object {self[name]} with this name: '{name}' already exist and register is not in replace mode"
            )
        if not isinstance(name, str):
            raise ValueError(f"Function get_name of class {value} need return string not {type(name)}")
        self[name] = value
        if old_names is not None:
            # FIXME add better strategy to get proper class when there is conflict of names
            for old_name in old_names:
                if old_name in self._old_mapping and not replace:
                    raise ValueError(
                        f"Old value mapping for name '{old_name}' already registered."
                        f" Currently pointing to {self._old_mapping[old_name]}"
                    )
                self._old_mapping[old_name] = name
        return value

    @staticmethod
    def check_function(ob, function_name, is_class):
        fun = getattr(ob, function_name, None)
        if not is_class and not inspect.isfunction(fun):
            raise ValueError(f"Class {ob} need to define method {function_name}")
        if is_class and not inspect.ismethod(fun) and not is_static(fun):
            raise ValueError(f"Class {ob} need to define classmethod '{function_name}'")

    def __setitem__(self, key: str, value: AlgorithmType):
        if not issubclass(value, AlgorithmDescribeBase):
            raise ValueError(
                f"Class {value} need to be subclass of {AlgorithmDescribeBase.__module__}.AlgorithmDescribeBase"
            )
        self.check_function(value, "get_name", True)
        self.check_function(value, "get_fields", True)
        try:
            val = value.get_name()
        except (NotImplementedError, AttributeError):
            raise ValueError(f"Class {value} need to implement classmethod 'get_name'") from None
        if not isinstance(val, str):
            raise ValueError(f"Function get_name of class {value} need return string not {type(val)}")
        if key != val:
            raise ValueError("Object need to be registered under name returned by get_name function")
        if not value.__new_style__:
            try:
                val = value.get_fields()
                if not isinstance(val, list):
                    raise ValueError(f"Function get_fields of class {value} need return list not {type(val)}")
            except NotImplementedError:
                raise ValueError(f"Method get_fields of class {value} need to be implemented") from None
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
            raise ValueError("Register does not contain any algorithm.") from None


class AddRegisterMeta(ModelMetaclass):
    def __new__(cls, name, bases, attrs, **kwargs):
        methods = kwargs.pop("methods", [])
        suggested_base_class = kwargs.pop("suggested_base_class", None)
        class_methods = kwargs.pop("class_methods", [])
        cls2 = super().__new__(cls, name, bases, attrs, **kwargs)  # pylint: disable=too-many-function-args
        cls2.__register__ = Register(
            class_methods=class_methods, methods=methods, suggested_base_class=suggested_base_class
        )
        return cls2

    def __getitem__(self, item) -> AlgorithmType:
        return self.__register__[item]

    def __contains__(self, item) -> bool:
        return self.__register__.__contains__(item)

    def get(self, item, default=None):
        return self.__register__.get(item, default)


class AlgorithmSelection(BaseModel, metaclass=AddRegisterMeta):  # pylint: disable=E1139
    """
    Base class for algorithm selection.
    For given algorithm there should be Register instance set __register__ class variable.
    """

    name: str
    values: typing.Union[typing.Dict[str, typing.Any], PydanticBaseModel] = Field(..., union_mode="left_to_right")
    class_path: str = ""
    if typing.TYPE_CHECKING:
        __register__: Register

    if parse_version(version("pydantic")) < parse_version("2"):

        class Config:
            smart_union = True

    @validator("name")
    def check_name(cls, v):
        if v not in cls.__register__:
            raise ValueError(f"Missed algorithm {v}")
        return v

    @validator("class_path", always=True)
    def update_class_path(cls, v, values):
        if v or "name" not in values:
            return v
        klass = cls.__register__[values["name"]]
        return class_to_str(klass)

    @field_serializer("values")
    def val_serializer(self, value, _info):
        if isinstance(value, PydanticBaseModel):
            return value.dict()
        return value

    @validator("values", pre=True)
    def update_values(cls, v, values):
        # FIXME add better strategy to get proper class when there is conflict of names
        if "name" not in values or not isinstance(v, dict):
            return v
        klass = cls.__register__[values["name"]]
        if not klass.__new_style__ or not klass.__argument_class__.__fields__:
            return v

        dkt_migrated = REGISTER.migrate_data(class_to_str(klass.__argument_class__), {}, v)
        return klass.__argument_class__(**dkt_migrated)

    @classmethod
    def register(
        cls, value: AlgorithmType, replace=False, old_names: typing.Optional[typing.List[str]] = None
    ) -> AlgorithmType:
        """
        Function for registering :class:`.AlgorithmDescribeBase` based algorithms
        :param value: algorithm to register
        :param replace: replace existing algorithm, be patient with
        :param old_names: list of old names for registered class
        """
        return cls.__register__.register(value, replace=replace, old_names=old_names)

    @classmethod
    def get_default(cls):
        name = cls.__register__.get_default()
        return cls(name=name, values=cls[name].get_default_values())

    def algorithm(self):
        return self.__register__[self.name]


class ROIExtractionProfileMeta(ModelMetaclass):
    def __new__(cls, name, bases, attrs, **kwargs):
        cls2 = super().__new__(cls, name, bases, attrs, **kwargs)  # pylint: disable=too-many-function-args

        def allow_positional_args(func):
            @wraps(func)
            def _wraps(self, *args, **kwargs):
                if args:
                    warnings.warn(
                        "Positional arguments are deprecated, use keyword arguments instead",
                        FutureWarning,
                        stacklevel=2,
                    )
                    kwargs.update(dict(zip(self.__fields__, args)))
                return func(self, **kwargs)

            return _wraps

        cls2.__init__ = allow_positional_args(cls2.__init__)
        return cls2


class ROIExtractionProfile(BaseModel, metaclass=ROIExtractionProfileMeta):  # pylint: disable=E1139
    """
    :ivar str ~.name: name for segmentation profile
    :ivar str ~.algorithm: Name of algorithm
    :ivar dict ~.values: algorithm parameters
    """

    name: str
    algorithm: str
    values: typing.Any

    @validator("values")
    def validate_values(cls, v, values):  # pylint: disable=no-self-use
        if not isinstance(v, dict):
            return v
        if "algorithm" not in values:
            return v
        from PartSegCore.analysis import AnalysisAlgorithmSelection
        from PartSegCore.mask.algorithm_description import MaskAlgorithmSelection

        name = values["algorithm"]
        is_analysis = name in AnalysisAlgorithmSelection
        is_mask = name in MaskAlgorithmSelection
        if is_analysis == is_mask:
            return v
        algorithm = AnalysisAlgorithmSelection[name] if is_analysis else MaskAlgorithmSelection[name]
        if not algorithm.__new_style__:
            return v
        dkt_migrated = REGISTER.migrate_data(class_to_str(algorithm.__argument_class__), {}, v)
        return algorithm.__argument_class__(**dkt_migrated)

    def pretty_print(self, algorithm_dict):
        if isinstance(algorithm_dict, AlgorithmSelection):
            algorithm_dict = algorithm_dict.__register__
        try:
            algorithm = algorithm_dict[self.algorithm]
        except KeyError:
            return f"{self}\n "
        values = self.values if isinstance(self.values, dict) else self.values.dict()
        if self.name in {"", "Unknown"}:
            return (
                "ROI extraction profile\nAlgorithm: "
                + self.algorithm
                + "\n"
                + self._pretty_print(values, algorithm.get_fields_dict())
            )
        return (
            f"ROI extraction profile name: {self.name}\nAlgorithm: {self.algorithm}\n"
            f"{self._pretty_print(values, algorithm.get_fields_dict())}"
        )

    @classmethod
    def _pretty_print(
        cls, values: typing.MutableMapping, translate_dict: typing.Dict[str, AlgorithmProperty], indent=0
    ):
        if not isinstance(values, typing.MutableMapping):
            return textwrap.indent(str(values), " " * indent)
        res = ""
        for k, v in values.items():
            if k not in translate_dict:
                if isinstance(v, typing.MutableMapping):
                    res += " " * indent + f"{k}: {cls._pretty_print(v, {}, indent + 2)}\n"
                else:
                    res += " " * indent + f"{k}: {v}\n"
                continue
            desc = translate_dict[k]
            res += " " * indent + desc.user_name + ": "
            if issubclass(desc.value_type, Channel) and not isinstance(v, Channel):
                res += str(Channel(v))
            elif issubclass(desc.value_type, AlgorithmDescribeBase):
                if isinstance(v, AlgorithmSelection):
                    name = v.name
                    values_ = v.values
                else:
                    name = v["name"]
                    values_ = v["values"]
                res += desc.possible_values[name].get_name()
                if values_:
                    res += "\n"
                    res += cls._pretty_print(values_, desc.possible_values[name].get_fields_dict(), indent + 2)
            elif isinstance(v, typing.MutableMapping):
                res += cls._pretty_print(v, {}, indent + 2)
            else:
                res += str(v)
            res += "\n"
        return res[:-1]

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.name == other.name
            and self.algorithm == other.algorithm
            and self.values == other.values
        )


def _next_after(type_, value, inf):
    if issubclass(type_, int):
        if inf == math.inf:
            return value + 1
        return value - 1
    if hasattr(math, "nextafter"):
        # TODO fix after drop python 3.8
        return math.nextafter(value, inf)
    return value


def _next_after_with_none(type_, value, inf):
    if value is None:
        return None
    return _next_after(type_, value, inf)


def _calc_value_range(field_info: FieldInfo):
    if field_info.metadata is None:
        return (0, 1000)

    import annotated_types as at

    value_range = (0, 1000)
    for el in field_info.metadata:
        if isinstance(el, at.Ge):
            value_range = el.ge, value_range[1]
        elif isinstance(el, at.Gt):
            value_range = _next_after(field_info.annotation, el.gt, math.inf), value_range[1]
        elif isinstance(el, at.Le):
            value_range = value_range[0], el.le
        elif isinstance(el, at.Lt):
            value_range = value_range[0], _next_after(field_info.annotation, el.lt, -math.inf)
    return value_range


def _field_to_algorithm_property_pydantic_2(name: str, field_info: FieldInfo):
    user_name = field_info.title

    value_range = None
    possible_values = None

    value_type = field_info.annotation
    default_value = field_info.default
    help_text = field_info.description
    if user_name is None:
        user_name = name.replace("_", " ").capitalize()
    if not hasattr(value_type, "__origin__"):
        if issubclass(value_type, (int, float)):
            value_range = _calc_value_range(field_info)

        if issubclass(field_info.annotation, AlgorithmSelection):
            value_type = AlgorithmDescribeBase
            if isinstance(field_info.default, UndefinedType):
                default_value = field_info.default_factory().name
            else:
                default_value = field_info.default.name
            possible_values = field_info.annotation.__register__

    extra = field_info.json_schema_extra or {}

    return AlgorithmProperty(
        name=name,
        user_name=user_name,
        default_value=default_value,
        options_range=value_range,
        value_type=value_type,
        possible_values=possible_values,
        help_text=help_text,
        mgi_options=extra.get("options", {}),
    )


def _field_to_algorithm_property_pydantic_1(name: str, field: "ModelField"):
    user_name = field.field_info.title
    value_range = None
    possible_values = None

    value_type = getattr(field, "annotation", field.type_)
    default_value = field.field_info.default
    help_text = field.field_info.description
    if user_name is None:
        user_name = name.replace("_", " ").capitalize()
    if not hasattr(field.type_, "__origin__"):
        if issubclass(field.type_, (int, float)):
            value_range = (
                field.field_info.ge or _next_after_with_none(field.type_, field.field_info.gt, math.inf) or 0,
                field.field_info.le or _next_after_with_none(field.type_, field.field_info.lt, -math.inf) or 1000,
            )
        if issubclass(field.type_, AlgorithmSelection):
            value_type = AlgorithmDescribeBase
            if isinstance(field.field_info.default, UndefinedType):
                default_value = field.field_info.default_factory().name
            else:
                default_value = field.field_info.default.name
            possible_values = field.type_.__register__

    return AlgorithmProperty(
        name=name,
        user_name=user_name,
        default_value=default_value,
        options_range=value_range,
        value_type=value_type,
        possible_values=possible_values,
        help_text=help_text,
        mgi_options=field.field_info.extra.get("options", {}),
    )


def base_model_to_algorithm_property_pydantic_1(
    obj: typing.Type[BaseModel],
) -> typing.List[typing.Union[str, AlgorithmProperty]]:
    """
    Convert pydantic model to list of AlgorithmPropert nad strings.

    :param obj:
    :return:
    """
    res = []
    value: ModelField
    if hasattr(obj, "header") and obj.header():
        res.append(obj.header())
    for name, value in obj.__fields__.items():
        ap = _field_to_algorithm_property(name, value)
        if value.field_info.extra.get("hidden", False):
            continue
        pos = len(res)
        if "position" in value.field_info.extra:
            pos = value.field_info.extra["position"]
        if "prefix" in value.field_info.extra:
            res.insert(pos, value.field_info.extra["prefix"])
            pos += 1

        res.insert(pos, ap)

        if "suffix" in value.field_info.extra:
            res.insert(pos + 1, value.field_info.extra["suffix"])
    return res


def base_model_to_algorithm_property_pydantic_2(
    obj: typing.Type[BaseModel],
) -> typing.List[typing.Union[str, AlgorithmProperty]]:
    """
    Convert pydantic model to list of AlgorithmPropert nad strings.

    :param obj:
    :return:
    """
    res = []
    field_info: FieldInfo
    if hasattr(obj, "header") and obj.header():
        res.append(obj.header())
    for name, field_info in obj.__fields__.items():
        ap = _field_to_algorithm_property(name, field_info)
        extra = field_info.json_schema_extra or {}
        if extra.get("hidden", False):
            continue
        pos = len(res)
        if "position" in extra:
            pos = extra["position"]
        if "prefix" in extra:
            res.insert(pos, extra["prefix"])
            pos += 1

        res.insert(pos, ap)

        if "suffix" in extra:
            res.insert(pos + 1, extra["suffix"])
    return res


if version("pydantic") < "2":
    _field_to_algorithm_property = _field_to_algorithm_property_pydantic_1
    base_model_to_algorithm_property = base_model_to_algorithm_property_pydantic_1
else:
    _field_to_algorithm_property = _field_to_algorithm_property_pydantic_2
    base_model_to_algorithm_property = base_model_to_algorithm_property_pydantic_2
