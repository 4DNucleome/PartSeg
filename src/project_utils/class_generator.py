import json
import sys
import collections
import typing
import pprint
import inspect
from .class_generate_base import BaseReadonlyClass as BaseReadonlyClass_

_PY36 = sys.version_info[:2] >= (3, 6)

_class_template = """\
from builtins import property as _property, tuple as _tuple
from operator import attrgetter as _attrgetter
from collections import OrderedDict
from {module} import BaseReadonlyClass
\"\"\"
{imports}
\"\"\"

class {typename}(BaseReadonlyClass):
    "{typename}({signature})"

    __slots__ = {slots!r}

    _fields = {field_names!r}
    
    #def __new__(cls, {signature}):
    #    ob = super().__new__(cls)
    #    cls.__init__(ob, {arg_list})
    #    return ob

    def __init__(self, {signature_without_types}):
        'Create new instance of {typename}({arg_list})'
        {init_fields}

    @classmethod
    def _make(cls, iterable):
        'Make a new {typename} object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != {num_fields:d}:
            raise TypeError('Expected {num_fields:d} arguments, got %d' % len(result))
        return result

    def replace_(self, **kwds):
        'Return a new {typename} object replacing specified fields with new values'
        dkt = self.asdict()
        dkt.update(kwds)
        return self.__class__(**dkt)

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

{field_defs}
"""

_repr_template = '{name}=%r'

_field_template = '''\
    {name} = _property(_attrgetter("_{name}"), doc='getter for field _{name}')
'''

class_register = dict()


def extract_type_info(type_):
    if isinstance(type_, (typing.GenericMeta, typing._Any, typing._Union)):
        return str(type_), type_.__module__
    elif hasattr(type_, "__module__"):
        return "{}.{}".format(type_.__module__, type_.__name__), type_.__module__
    else:
        return type_.__name__, None


_prohibited = ('__new__', '__init__', '__slots__', '__getnewargs__',
               '_fields', '_field_defaults', '_field_types',
               '_make', 'replace_', 'asdict', '_source', 'asdict')

_special = ('__module__', '__name__', '__qualname__', '__annotations__')


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
    signature = ", ".join(["{}: {} = {}".format(name_, type_dict[name_], pprint.pformat(
        defaults_dict[name_])) if name_ in defaults_dict else "{}: {}".format(name_, type_dict[name_]) for name_ in
                           types.keys()])
    signature_without_types = ", ".join(["{} = {}".format(name_, pprint.pformat(defaults_dict[name_]))
                                         if name_ in defaults_dict else "{}".format(name_) for name_ in
                                         types.keys()])
    init_sig = ["self._{name} = {name}".format(name=name_) for name_ in type_dict.keys()]
    tuple_list = ["self._{name}".format(name=name_) for name_ in type_dict.keys()]
    class_definition = _class_template.format(
        imports="\n".join(import_set),
        typename=typename,
        init_fields="\n        ".join(init_sig),
        tuple_fields=", ".join(tuple_list),
        signature=signature,
        module=BaseReadonlyClass_.__module__,
        field_names=tuple(field_names),
        signature_without_types=signature_without_types,
        slots=tuple(["_" + x for x in field_names]),
        num_fields=len(field_names),
        arg_list=repr(tuple(field_names)).replace("'", "")[1:-1],
        repr_fmt=', '.join(_repr_template.format(name=name)
                           for name in field_names),
        field_defs='\n'.join(_field_template.format(name=name)
                             for index, name in enumerate(field_names))
    )

    namespace = dict(__name__='namedtuple_%s' % typename)
    try:
        exec(class_definition, namespace)
    except AttributeError as e:
        print(class_definition, file=sys.stderr)
        raise e
    except NameError as e:
        for i, el in enumerate(class_definition.split("\n"), 1):
            print(f"{i}: {el}")
        raise e

    result = namespace[typename]
    result._source = class_definition
    result._field_defaults = defaults_dict
    result.__annotations__ = types
    result.__signature__ = inspect.signature(result)
    result._field_types = collections.OrderedDict(types)
    return result


class BaseMeta(type):
    def __new__(mcs, name, bases, attrs):
        print("BaseMeta.__new__", mcs, name, bases, attrs)
        if attrs.get('_root', False):
            return super().__new__(mcs, name, bases, attrs)
        if name in class_register:
            raise ValueError(f"Class {name} already exists")
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
        result = _make_class(name, types, defaults_dict, [x for x in bases if x != BaseReadonlyClass])
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

        class_register[name] = result
        return result


class BaseReadonlyClass(metaclass=BaseMeta):
    _root = True
    __signature__ = ()

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


class ReadonlyClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, BaseReadonlyClass_):
            return {"__ReadOnly__": True, "__subtype__": o.__class__.__name__,  **o.asdict()}
        return super().default(o)


def readonly_hook(_, dkt):
    if "__ReadOnly__" in dkt:
        del dkt["__ReadOnly__"]
        cls = class_register[dkt["__subtype__"]]
        del dkt["__subtype__"]
        res = cls(**dkt)
        return res
    return dkt
