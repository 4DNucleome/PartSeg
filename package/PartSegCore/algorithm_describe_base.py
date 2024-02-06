import inspect
import textwrap
import typing
import warnings
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from functools import wraps

from local_migrator import REGISTER, class_to_str
from pydantic import BaseModel as PydanticBaseModel
from pydantic import create_model, validator
from pydantic.fields import ModelField, UndefinedType
from pydantic.main import ModelMetaclass
from typing_extensions import Annotated

from PartSegCore.utils import BaseModel
from PartSegImage import Channel

T = typing.TypeVar("T", bound="AlgorithmDescribeBase")

TypeT = typing.Type[T]


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
        self.range = options_range
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
    def __new__(cls, name, bases, attrs, method_from_fun=None, additional_parameters=None, **kwargs):
        cls2 = super().__new__(cls, name, bases, attrs, **kwargs)
        if (
            not inspect.isabstract(cls2)
            and hasattr(cls2.get_fields, "__is_partial_abstractmethod__")
            and cls2.__argument_class__ is None
        ):
            raise RuntimeError("class need to have __argument_class__ set or get_fields functions defined")
        cls2.__new_style__ = getattr(cls2.get_fields, "__is_partial_abstractmethod__", False)
        cls2.__from_function__ = getattr(cls2, "__from_function__", False)
        cls2.__abstract_getters__ = {}
        cls2.__method_name__ = method_from_fun or getattr(cls2, "__method_name__", None)
        cls2.__additional_parameters_name__ = additional_parameters or getattr(
            cls2, "__additional_parameters_name__", None
        )
        if cls2.__additional_parameters_name__ is None:
            cls2.__additional_parameters_name__ = cls._get_calculation_method_params_name(cls2)

        cls2.__support_from_function__ = (
            cls2.__method_name__ is not None and cls2.__additional_parameters_name__ is not None
        )

        cls2.__abstract_getters__, cls2.__support_from_function__ = cls._get_abstract_getters(
            cls2, cls2.__support_from_function__, method_from_fun
        )
        return cls2

    @staticmethod
    def _get_abstract_getters(
        cls2, support_from_function, calculation_method
    ) -> typing.Tuple[typing.Dict[str, typing.Any], bool]:
        abstract_getters = {}
        if hasattr(cls2, "__abstractmethods__") and cls2.__abstractmethods__:
            # get all abstract methods that starts with `get_`
            for method_name in cls2.__abstractmethods__:
                if method_name.startswith("get_"):
                    method = getattr(cls2, method_name)
                    if "return" not in method.__annotations__:
                        msg = f"Method {method_name} of {cls2.__qualname__} need to have return type defined"
                        try:
                            file_name = inspect.getsourcefile(method)
                            line = inspect.getsourcelines(method)[1]
                            msg += f" in {file_name}:{line}"
                        except TypeError:
                            pass
                        raise RuntimeError(msg)

                    abstract_getters[method_name[4:]] = getattr(cls2, method_name).__annotations__["return"]
                elif method_name != calculation_method:
                    support_from_function = False
        return abstract_getters, support_from_function

    @staticmethod
    def _get_calculation_method_params_name(cls2) -> typing.Optional[str]:
        if cls2.__method_name__ is None:
            return None
        signature = inspect.signature(getattr(cls2, cls2.__method_name__))
        if "arguments" in signature.parameters:
            return "arguments"
        if "params" in signature.parameters:
            return "params"
        if "parameters" in signature.parameters:
            return "parameters"
        raise RuntimeError(f"Cannot determine arguments parameter name in {cls2.__method_name__}")

    @staticmethod
    def _validate_if_all_abstract_getters_are_defined(abstract_getters, kwargs):
        abstract_getters_set = set(abstract_getters)
        kwargs_set = set(kwargs.keys())

        if abstract_getters_set != kwargs_set:
            # Provide a nice error message with information about what is missing and is obsolete
            missing_text = ", ".join(sorted(abstract_getters_set - kwargs_set))
            if missing_text:
                missing_text = f"Not all abstract methods are provided, missing: {missing_text}."
            else:
                missing_text = ""
            extra_text = ", ".join(sorted(kwargs_set - abstract_getters_set))
            if extra_text:
                extra_text = f"There are extra attributes in call: {extra_text}."
            else:
                extra_text = ""

            raise ValueError(f"{missing_text} {extra_text}")

    @staticmethod
    def _validate_function_parameters(func, method, method_name) -> set:
        """
        Validate if all parameters without default values are defined in self.__calculation_method__

        :param func: function to validate
        :return: set of parameters that should be dropped
        """
        signature = inspect.signature(func)
        base_method_signature = inspect.signature(method)
        take_all = False

        for parameter in signature.parameters.values():
            if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY}:
                raise ValueError(f"Function {func} should not have positional only parameters")
            if (
                parameter.default is inspect.Parameter.empty
                and parameter.name not in base_method_signature.parameters
                and parameter.kind != inspect.Parameter.VAR_KEYWORD
            ):
                raise ValueError(f"Parameter {parameter.name} is not defined in {method_name} method")

            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                take_all = True

        if take_all:
            return set()

        return {
            parameters.name
            for parameters in base_method_signature.parameters.values()
            if parameters.name not in signature.parameters
        }

    @staticmethod
    def _get_argument_class_from_signature(func, argument_name: str):
        signature = inspect.signature(func)
        if argument_name not in signature.parameters:
            return BaseModel
        return signature.parameters[argument_name].annotation

    @staticmethod
    def _get_parameters_from_signature(func):
        signature = inspect.signature(func)
        return [parameters.name for parameters in signature.parameters.values()]

    def from_function(self, func=None, **kwargs):
        """generate new class from function"""

        # Test if all abstract methods values are provided in kwargs

        if not self.__support_from_function__:
            raise RuntimeError("This class does not support from_function method")

        self._validate_if_all_abstract_getters_are_defined(self.__abstract_getters__, kwargs)

        # check if all values have correct type
        for key, value in kwargs.items():
            if not isinstance(value, self.__abstract_getters__[key]):
                raise TypeError(f"Value for {key} should be {self.__abstract_getters__[key]}")

        def _getter_by_name(name):
            def _func():
                return kwargs[name]

            return _func

        parameters_order = self._get_parameters_from_signature(getattr(self, self.__method_name__))

        def _class_generator(func_):
            drop_attr = self._validate_function_parameters(
                func_, getattr(self, self.__method_name__), self.__method_name__
            )

            @wraps(func_)
            def _calculate_method(*args, **kwargs_):
                for attr, name in zip(args, parameters_order):
                    if name in kwargs_:
                        raise ValueError(f"Parameter {name} is defined twice")
                    kwargs_[name] = attr

                for name in drop_attr:
                    kwargs_.pop(name, None)
                return func_(**kwargs_)

            class_dkt = {f"get_{name}": _getter_by_name(name) for name in self.__abstract_getters__}

            class_dkt[self.__method_name__] = _calculate_method
            class_dkt["__argument_class__"] = self._get_argument_class_from_signature(
                func_, self.__additional_parameters_name__
            )
            class_dkt["__from_function__"] = True

            return type(func_.__name__.replace("_", " ").title().replace(" ", ""), (self,), class_dkt)

        if func is None:
            return _class_generator
        return _class_generator(func)


class AlgorithmDescribeBase(ABC, metaclass=AlgorithmDescribeBaseMeta):
    """
    This is abstract class for all algorithm exported to user interface.
    Based on get_name and get_fields methods the interface will be generated
    For each group of algorithm base abstract class will add additional methods
    """

    __argument_class__: typing.Optional[typing.Type[PydanticBaseModel]] = None
    __new_style__: bool

    def __new__(cls, *args, **kwargs):
        if cls.__from_function__:
            return getattr(cls, cls.__method_name__)(*args, **kwargs)
        return super().__new__(cls)

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
    @typing.overload
    def from_function(cls: TypeT, func: typing.Callable[..., typing.Any], **kwargs) -> TypeT:
        ...

    @classmethod
    @typing.overload
    def from_function(cls: TypeT, **kwargs) -> typing.Callable[[typing.Callable[..., typing.Any]], TypeT]:
        ...

    @classmethod
    def from_function(
        cls: TypeT, func=None, **kwargs
    ) -> typing.Union[TypeT, typing.Callable[[typing.Callable], TypeT]]:
        def _from_function(func_) -> typing.Type["AlgorithmDescribeBase"]:
            if "name" not in kwargs:
                kwargs["name"] = func_.__name__.replace("_", " ").title()
            return AlgorithmDescribeBaseMeta.from_function(cls, func_, **kwargs)

        if func is None:
            return _from_function
        return _from_function(func)

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
        except NotImplementedError:
            raise ValueError(f"Class {value} need to implement get_name class method") from None
        if name in self and not replace:
            raise ValueError(
                f"Object {self[name]} with this name: {name} already exist and register is not in replace mode"
            )
        if not isinstance(name, str):
            raise ValueError(f"Function get_name of class {value} need return string not {type(name)}")
        self[name] = value
        if old_names is not None:
            # FIXME add better strategy to get proper class when there is conflict of names
            for old_name in old_names:
                if old_name in self._old_mapping and not replace:
                    raise ValueError(
                        f"Old value mapping for name {old_name} already registered."
                        f" Currently pointing to {self._old_mapping[name]}"
                    )
                self._old_mapping[old_name] = name
        return value

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
                f"Class {value} need to inherit from {AlgorithmDescribeBase.__module__}.AlgorithmDescribeBase"
            )
        self.check_function(value, "get_name", True)
        self.check_function(value, "get_fields", True)
        try:
            val = value.get_name()
        except NotImplementedError:
            raise ValueError(f"Method get_name of class {value} need to be implemented") from None
        if not isinstance(val, str):
            raise ValueError(f"Function get_name of class {value} need return string not {type(val)}")
        if key != val:
            raise ValueError("Object need to be registered under name returned by gey_name function")
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
    values: typing.Union[PydanticBaseModel, typing.Dict[str, typing.Any]]
    class_path: str = ""
    if typing.TYPE_CHECKING:
        __register__: Register

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
        return cls.__register__.register(value, replace, old_names)

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

    @classmethod
    def print_dict(cls, dkt, indent=0, name: str = "") -> str:
        if isinstance(dkt, Enum):
            return dkt.name
        if not isinstance(dkt, typing.MutableMapping):
            # FIXME update in future method of proper printing channel number
            if name.startswith("channel") and isinstance(dkt, int):
                return str(dkt + 1)
            return str(dkt)
        return "\n" + "\n".join(
            " " * indent + f"{k.replace('_', ' ')}: {cls.print_dict(v, indent + 2, k)}" for k, v in dkt.items()
        )

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and self.name == other.name
            and self.algorithm == other.algorithm
            and self.values == other.values
        )


def _field_to_algorithm_property(name: str, field: "ModelField"):
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
                field.field_info.ge or field.field_info.gt or 0,
                field.field_info.le or field.field_info.lt or 1000,
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


def base_model_to_algorithm_property(obj: typing.Type[BaseModel]) -> typing.List[typing.Union[str, AlgorithmProperty]]:
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
