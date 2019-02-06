import json
import sys
import collections
import importlib
import pprint
import inspect
import itertools
import typing
from enum import Enum

_PY36 = sys.version_info[:2] >= (3, 6)

# TODO read about dataclsss and maybe apply

_class_template = """\
from builtins import property as _property, tuple as _tuple
from operator import attrgetter as _attrgetter
from collections import OrderedDict
import typing
\"\"\"
{imports}
\"\"\"

class {typename}({base_classes}):
    "{typename}({signature})"
    __slots__ = {slots!r}

    _fields = {field_names!r}
    _root = True
    
    #def __new__(cls, {signature}):
    #    ob = super().__new__(cls)
    #    cls.__init__(ob, {arg_list})
    #    return ob

    def __init__(self, {signature}):
        'Create new instance of {typename}({arg_list})'
        {init_fields}

    @classmethod
    def _make(cls, iterable):
        'Make a new {typename} object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != {num_fields:d}:
            raise TypeError('Expected {num_fields:d} arguments, got %d' % len(result))
        return result

    def replace_(self, **kwargs):
        'Return a new {typename} object replacing specified fields with new values'
        dkt = self.asdict()
        dkt.update(kwargs)
        return self.__class__(**dkt)
        
    def _replace(self, **kwargs):
        return self.replace_(**kwargs)

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + '({repr_fmt})' % _attrgetter(*self.__slots__)(self)
    
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.as_tuple() == other.as_tuple()

    def asdict(self):
        'Return a new OrderedDict which maps field names to their values.'
        return OrderedDict(zip(self._fields, _attrgetter(*self.__slots__)(self)))
    
    def as_tuple(self):
        return {tuple_fields}

    def __getnewargs__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return tuple(_attrgetter(*self.__slots__)(self))

{field_definitions}
"""

_repr_template = '{name}=%r'

_field_template = '''\
    {name} = _property(_attrgetter("_{name}"), doc='getter for field _{name}')
'''


class RegisterClass:
    def __init__(self):
        self.exact_class_register = dict()
        self.predict_class_register = collections.defaultdict(list)

    def register_class(self, cls: type, old_name=None):
        path = extract_type_info(cls)[0]
        name = cls.__name__
        if path in self.exact_class_register:
            raise ValueError("class already registered")
        self.exact_class_register[path] = cls
        self.predict_class_register[name].append(cls)
        if old_name is not None:
            self.predict_class_register[old_name].append(cls)

    def get_class(self, path: str):
        if path in self.exact_class_register:
            return self.exact_class_register[path]
        else:
            name = path[path.rfind(".")+1:]
            if name not in self.predict_class_register:
                try:
                    importlib.import_module(path[:path.rfind(".")])
                except ImportError:
                    pass
            if name in self.predict_class_register:
                if len(self.predict_class_register[name]) == 1:
                    return self.predict_class_register[name][0]
                return iter(self.predict_class_register[name])
            raise ValueError(f"unregistered class {path}")

    def clear(self):  # for testing purpose
        self.exact_class_register.clear()
        self.predict_class_register.clear()


base_serialize_register = RegisterClass()
enum_register = RegisterClass()


def extract_type_name(type_):
    if not hasattr(type_, "__name__"):
        if hasattr(type_, "__module__") and type_.__module__ == "typing" and hasattr(type_, "__origin__"):
            return type_.__origin__
        return str(type_)
    return type_.__name__


def extract_type_info(type_):
    # noinspection PyUnresolvedReferences
    # if issubclass(type_, (typing.Any, typing.Union)):
    #    return str(type_), type_.__module__
    if hasattr(type_, "__module__"):
        if type_.__module__ == "typing":
            return str(type_), type_.__module__
        return "{}.{}".format(type_.__module__, type_.__name__), type_.__module__
    else:
        return type_.__name__, None


_prohibited = ('__new__', '__init__', '__slots__', '__getnewargs__',
               '_fields', '_field_defaults', '_field_types',
               '_make', 'replace_', 'asdict', '_source', 'asdict')

_special = ('__module__', '__name__', '__qualname__', '__annotations__')


def add_classes(types_list, translate_dict, global_state):
    for type_ in types_list:
        if type_ in translate_dict:
            continue
        if hasattr(type_, "__module__") and type_.__module__ == "typing":
            if hasattr(type_, "__args__") and isinstance(type_.__args__, collections.Iterable) \
                    and len(type_.__args__) > 0:
                add_classes(type_.__args__, translate_dict, global_state)
                if hasattr(type_, "__origin__"):
                    type_str = \
                        str(type_.__origin__) + "[" + ", ".join([translate_dict[x] for x in type_.__args__]) + "]"
                    translate_dict[type_] = type_str
                    continue
            if isinstance(type_, typing._ForwardRef):
                translate_dict[type_] = f"'{type_.__forward_arg__}'"
                continue
            translate_dict[type_] = str(type_)
            continue

        name = extract_type_name(type_)
        while name in global_state:
            name += "a"
        translate_dict[type_] = name

        global_state[name] = type_


def _make_class(typename, types, defaults_dict, base_classes):
    if base_classes:
        # TODO add function inheritance
        types_ = types
        types = {}
        for el in base_classes:
            if hasattr(el, "__annotations__"):
                types.update(el.__annotations__)
        types.update(types_)
    field_names = list(types.keys())
    import_set = set()
    type_dict = {}
    for name_, type_ in types.items():
        type_str, module = extract_type_info(type_)
        type_dict[name_] = type_str
        if module:
            import_set.add("import " + module)
    translate_dict = {type(None): "None"}
    global_state = {typename: "a", "typing": typing}
    add_classes(itertools.chain(types.values(), base_classes), translate_dict, global_state)

    del global_state[typename]

    signature = ", ".join(["{}: {} = {}".format(name_, translate_dict[type_], pprint.pformat(
        defaults_dict[name_])) if name_ in defaults_dict else "{}: {}".format(name_, translate_dict[type_])
                           for name_, type_ in types.items()])
    init_sig = ["self._{name} = {name}".format(name=name_) for name_ in type_dict.keys()]
    tuple_list = ["self._{name}".format(name=name_) for name_ in type_dict.keys()]
    class_definition = _class_template.format(
        imports="\n".join(import_set),
        typename=typename,
        init_fields="\n        ".join(init_sig),
        tuple_fields=", ".join(tuple_list),
        signature=signature,
        field_names=tuple(field_names),
        slots=tuple(["_" + x for x in field_names]),
        num_fields=len(field_names),
        arg_list=repr(tuple(field_names)).replace("'", "")[1:-1],
        repr_fmt=', '.join(_repr_template.format(name=name)
                           for name in field_names),
        field_definitions='\n'.join(_field_template.format(name=name)
                                    for index, name in enumerate(field_names)),
        base_classes=", ".join([translate_dict[x] for x in base_classes])
    )
    global_state["__name__"] = 'serialize_%s' % typename
    try:
        exec(class_definition, global_state)
    except AttributeError as e:
        print(class_definition, file=sys.stderr)
        raise e
    except NameError as e:
        for i, el in enumerate(class_definition.split("\n"), 1):
            print(f"{i}: {el}", file=sys.stderr)
        raise e

    result = global_state[typename]
    result._source = class_definition
    result._field_defaults = defaults_dict
    result.__annotations__ = types
    result.__signature__ = inspect.signature(result)
    result._field_types = collections.OrderedDict(types)
    return result


class BaseMeta(type):
    def __new__(mcs, name, bases, attrs):
        # print("BaseMeta.__new__", mcs, name, bases, attrs)
        if attrs.get('_root', False):
            return super().__new__(mcs, name, bases, attrs)
        """if name in class_register:
            raise ValueError(f"Class {name} already exists")"""
        types = attrs.get("__annotations__", {})
        defaults = []
        defaults_dict = {}
        for field_name in types:
            if field_name in attrs:
                default_value = attrs[field_name]
                defaults.append(default_value)
                defaults_dict[field_name] = default_value
            elif defaults:
                raise TypeError("Non-default namedtuple field {field_name} cannot "
                                "follow default field(s) {default_names}"
                                .format(field_name=field_name,
                                        default_names=', '.join(defaults_dict.keys())))
        result = _make_class(name, types, defaults_dict, [x for x in bases])
        # nm_tpl.__new__.__annotations__ = collections.OrderedDict(types)
        # nm_tpl.__new__.__defaults__ = tuple(defaults)
        # nm_tpl._field_defaults = defaults_dict
        module = attrs.get("__module__", None)
        if module is None:
            try:
                module = sys._getframe(1).f_globals.get('__name__', '__main__')
            except (AttributeError, ValueError):
                pass
        if module is not None:
            result.__module__ = module

        for key in attrs:
            if key in _prohibited:
                raise AttributeError("Cannot overwrite NamedTuple attribute " + key)
            elif key not in _special and key not in result._fields:
                setattr(result, key, attrs[key])
        base_serialize_register.register_class(result)
        return result


class BaseSerializableClass(metaclass=BaseMeta):
    _root = True
    # __signature__ = ()
    __readonly__ = True

    def __init__(self, *args, **kwargs):
        pass

    def asdict(self) -> collections.OrderedDict:
        pass

    def replace_(self, **_kwargs):
        return self

    """def __new__(cls, *fields, **kwargs):
        ob = super().__new__(cls)
        cls.__init__(ob, *fields, **kwargs)
        return ob

    def __init__(self, *args, **kwargs):
        pass"""


class SerializeClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return {"__Enum__": True, "__subtype__": extract_type_info(o.__class__)[0], "value": o.value}
        if isinstance(o, BaseSerializableClass):
            return {"__Serializable__": True, "__subtype__": extract_type_info(o.__class__)[0],  **o.asdict()}
        return super().default(o)


def serialize_hook(dkt: dict):
    if "__ReadOnly__" in dkt or "__Serializable__" in dkt:
        # Backward compatibility"
        try:
            cls = base_serialize_register.get_class(dkt["__subtype__"])
        except ValueError:
            dkt["__error__"] = True
            return dkt
        del dkt["__subtype__"]
        if "__Serializable__" in dkt:
            del dkt["__Serializable__"]
        else:
            del dkt["__ReadOnly__"]
        if isinstance(cls, collections.Iterator):
            keys = set(dkt.keys())
            for el in cls:
                el_keys = set(inspect.signature(el).parameters.keys())
                if keys == el_keys:
                    cls = el
                    break
            else:
                raise ValueError(f"cannot decode {dkt}")
        return cls(**dkt)
    if "__Enum__" in dkt:
        try:
            cls = enum_register.get_class(dkt["__subtype__"])
        except ValueError:
            dkt["__error__"] = True
            return dkt
        del dkt["__Enum__"]
        del dkt["__subtype__"]
        if isinstance(cls, collections.Iterator):
            raise ValueError("Two enum with same name")
        return cls(dkt["value"])
    return dkt
