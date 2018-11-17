import typing
import inspect
from collections import OrderedDict


class AlgorithmProperty(object):
    """
    :type name: str
    :type value_type: type
    :type default_value: typing.Union[object, str, int, float]
    """

    def __init__(self, name: str, user_name: str, default_value, options_range=None, single_steep=None,
                 possible_values=None, property_type=None):
        self.name = name
        self.user_name = user_name
        if type(options_range) is list:
            self.value_type = list
        elif property_type is not None:
            self.value_type = property_type
        else:
            self.value_type = type(default_value)
        self.default_value = default_value
        self.range = options_range
        self.possible_values = possible_values
        self.single_step = single_steep
        if self.value_type is list:
            assert default_value in options_range

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}(name='{self.name}'," \
               f" user_name='{self.user_name}', " + \
               f"default_value={self.default_value}, type={self.value_type}, range={self.range}," \
               f"possible_values={self.possible_values})"


class AlgorithmDescribeBase:
    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def get_fields(cls) -> typing.List[AlgorithmProperty]:
        raise NotImplementedError()


class Register(OrderedDict):

    def __getitem__(self, item):
        return super().__getitem__(item)

    def register(self, value: typing.Type[AlgorithmDescribeBase], replace=False):
        self.check_function(value, "get_name", True)
        name = value.get_name()
        if name in self and not replace:
            raise ValueError("Object with this name already exist and register is not in replace mode")
        if not isinstance(name, str):
            raise ValueError(f"Function get_name of class {value} need return string not {type(name)}")
        self[name] = value

    def check_function(self, ob, function_name, is_class):
        fun = getattr(ob, function_name, None)
        if not is_class and not inspect.isfunction(fun):
            raise ValueError(f"Class {ob} need to define method {function_name}")
        if is_class and not inspect.ismethod(fun):
            raise ValueError(f"Class {ob} need to define classmethod {function_name}")

    def __setitem__(self, key, value):
        if not issubclass(value, AlgorithmDescribeBase):
            raise ValueError(f"Class {value} need to inherit from "
                             f"{AlgorithmDescribeBase.__module__}.AlgorithmDescribeBase")
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
            raise ValueError(f"Function get_name of class {value} need return list not {type(val)}")
        super().__setitem__(key, value)
