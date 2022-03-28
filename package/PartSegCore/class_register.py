"""
This module contains utility for registration migration information for class.
"""
import importlib
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from packaging.version import Version
from packaging.version import parse as parse_version


def class_to_str(cls) -> str:
    """Get full qualified name for e given class."""
    if cls.__module__.startswith("pydantic.dataclass"):
        cls = cls.__mro__[1]
        return class_to_str(cls)
    return f"{cls.__module__}.{cls.__qualname__}"


def get_super_class(cls: Type) -> Optional[Type]:
    """Get parent class for a given class"""
    if len(cls.__mro__) < 2:
        return None
    if cls.__module__.startswith("pydantic.dataclass"):
        return get_super_class(cls.__mro__[1])
    return cls.__mro__[1]


def str_to_version(version: Union[str, Version]) -> Version:
    """If version passed as sting then convert it to Version object, otherwise return untouched."""
    return parse_version(version) if isinstance(version, str) else version


MigrationCallable = Callable[[Dict[str, Any]], Dict[str, Any]]
MigrationInfo = Tuple[Version, MigrationCallable]
"""Type describing single migration entry. For given class Version number should be unique."""
MigrationStartInfo = Tuple[Union[str, Version], MigrationCallable]

T = TypeVar("T")

RegisterReturnType = Union[Type[T], Callable[[Type[T]], Type[T]]]


@dataclass(frozen=True)
class TypeInfo:
    """
    Class for storing information in :py:class:`~.MigrationRegistration`

    :ivar str base_path: full qualified path to current class module and name
    :ivar typing.Type type_: class itself
    :ivar packaging.version.Version version: current clas version
    :ivar typing.List[~.MigrationInfo] migrations: list of migrations for deserialize old version
    :ivar bool use_parent_migrations: if migrations from parent class should be applied when deserialized object.
    """

    base_path: str
    type_: Type
    version: Version
    migrations: List[MigrationInfo]
    use_parent_migrations: bool


class MigrationRegistration:
    """
    Implementation of class register to storage information needed for migration from previous version.
    """

    def __init__(self):
        self._data_dkt: Dict[str, TypeInfo] = {}
        self._parent_migrations: Dict[str, bool] = {}

    def register(
        self,
        cls: Type = None,
        version: Union[str, Version] = "0.0.0",
        migrations: List[MigrationStartInfo] = None,
        old_paths: List[str] = None,
        use_parent_migrations: bool = True,
    ) -> RegisterReturnType:
        """
        Register class instance for storage information needed for deserialization of object from older version.


        :param cls: class to be registered
        :param version: current version of class
        :param  typing.List[MigrationInfo] migrations: list of migrations for deserialize old version
        :param old_paths: old name of class with modules
        :param use_parent_migrations: if migrations from parent class should be applied when deserialized object
        :return: class itself if cls parameter is provided. Otherwise,
            one argument function which will consume Type to be registered.
        """
        if migrations is None:
            migrations = []
        else:
            migrations = list(sorted((str_to_version(x), y) for x, y in migrations))
        if old_paths is None:
            old_paths = []
        version = str_to_version(version)

        if migrations and max(x for x, _ in migrations) > version:
            raise ValueError("class version lower than in migrations")

        def _register(cls_):
            base_path = class_to_str(cls_)
            type_info = TypeInfo(
                base_path=base_path,
                type_=cls_,
                version=version,
                migrations=migrations,
                use_parent_migrations=use_parent_migrations,
            )
            if base_path in self._data_dkt:
                raise RuntimeError(f"Class name {base_path} already taken by {self._data_dkt[base_path].base_path}")
            self._data_dkt[base_path] = type_info
            for name in old_paths:
                if name in self._data_dkt and self._data_dkt[name].base_path != base_path:
                    raise RuntimeError(f"Class name {name} already taken by {self._data_dkt[name].base_path}")
                self._data_dkt[name] = type_info
            return cls_

        if cls is None:
            return _register
        return _register(cls)

    def use_parent_migrations(self, name: str) -> bool:
        """
        Check if parent migrations should be used.

        :param name: full qualified path to class
        :return: information if parent class migrations should be applied.
        """
        self._register_missed(class_str=name)
        return self._data_dkt[name].use_parent_migrations

    def get_version(self, cls: Type) -> Version:
        """For a given class return version with which given class was registered using :py:meth:`register`"""
        class_str = class_to_str(cls)
        self._register_missed(class_str=class_str)
        return self._data_dkt[class_str].version

    def get_class(self, class_str: str) -> Type:
        """
        Get class base of qualified name. Could be done using current or old path.

        Qualified name is determined using :py:func:`class_to_str` and old path comes from the ``old_paths`` argument
        of :py:meth:`register` method.
        """
        self._register_missed(class_str=class_str)
        return self._data_dkt[class_str].type_

    def migrate_data(
        self, class_str: str, class_str_to_version_dkt: Dict[str, Union[str, Version]], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply migrations base on register state. Current implementation does not support multiple inheritance.

        :param class_str: fully qualified class path
        :param class_str_to_version_dkt: for each parent class information about version during serialization.
            If class is absent from this dict then assumed version is "0.0.0"
        :param data: dict of kwargs to constructor of class
        """
        if self.use_parent_migrations(class_str):
            super_klass = get_super_class(self.get_class(class_str))
            if super_klass is not None:
                data = self.migrate_data(class_to_str(super_klass), class_str_to_version_dkt, data)
        version = str_to_version(class_str_to_version_dkt.get(class_str, "0.0.0"))
        for version_, migration in self._data_dkt[class_str].migrations:
            if version < version_:
                data = migration(data)
        return data

    def _register_missed(self, class_str):
        """Register class if missed from register"""
        if class_str in self._data_dkt:
            return
        module_name, class_name = class_str.rsplit(".", maxsplit=1)
        class_path = [class_name]
        while True:
            try:
                module = importlib.import_module(module_name)
                break
            except ModuleNotFoundError:
                module_name, class_name_ = module_name.rsplit(".", maxsplit=1)
                class_path.append(class_name_)
        if class_str in self._data_dkt:
            return
        class_ = module
        for name in class_path[::-1]:
            class_ = getattr(class_, name)
        self.register(class_)


# THe global instance of register is use because registration is performed on import time.
# There should no information storage for objects.
REGISTER = MigrationRegistration()
"""Default register to storage class information"""


def rename_key(from_key: str, to_key: str, optional=False) -> MigrationCallable:
    """
    simple migration function for rename fields

    :param from_key: original name
    :param to_key: destination name
    :param optional: if migration is required (for backward compatibility)
    :return: migration function
    """

    def _migrate(dkt: Dict[str, Any]) -> Dict[str, Any]:
        if optional and from_key not in dkt:
            return dkt
        res_dkt = dkt.copy()
        res_dkt[to_key] = res_dkt.pop(from_key)
        return res_dkt

    return _migrate


def update_argument(argument_name):
    """
    This is decorator for move conversion of dict to class outside function code.
    It first inspects function signature to determine type th which argument should be converted.
    Then, if argument is passed as dict then all migrations from :py:attr:`REGISTER` all applied,
    then object is constructed and replace base one.

    :param argument_name: name of argument which should be converted

    Example::

        @register_class(version="0.0.1", migrations=[("0.0.1", rename_key("value", "value1"))])
        class DataClass:
            def __init__(value1, value2)
                self.value1 = value1
                self.value2 = value2

        @update_argument("arg")
        def some_func(arg: DataClass):
            print(arg.value1)

        some_func({"value": 1, "value2": 5})

    """

    def _wrapper(func):
        signature = inspect.signature(func)
        if argument_name not in signature.parameters:  # pragma: no cover
            raise RuntimeError("Argument should be accessible using inspect module.")
        arg_index = list(signature.parameters).index(argument_name)
        klass = signature.parameters[argument_name].annotation
        if not inspect.isclass(klass):  # pragma: no cover
            raise ValueError(f"Annotation {klass} of {argument_name} parameter is not a class")

        @wraps(func)
        def _update_from_dict(*args, **kwargs):
            if argument_name in kwargs and isinstance(kwargs[argument_name], dict):
                kwargs = kwargs.copy()
                kw = REGISTER.migrate_data(class_to_str(klass), {}, kwargs[argument_name])
                kwargs[argument_name] = klass(**kw)
            elif len(args) > arg_index and isinstance(args[arg_index], dict):
                args = list(args)
                kw = REGISTER.migrate_data(class_to_str(klass), {}, args[arg_index])
                args[arg_index] = klass(**kw)
            return func(*args, **kwargs)

        return _update_from_dict

    return _wrapper


def register_class(
    cls: Optional[Type[T]] = None,
    version: Union[str, Version] = "0.0.0",
    migrations: List[MigrationStartInfo] = None,
    old_paths: List[str] = None,
    use_parent_migrations: bool = True,
) -> RegisterReturnType:
    """
    This is wrapper for call :py:meth:`MigrationRegistration.register` of default register instance.
    Please see its documentation for details.

    :param cls: class to be registered
    :param version: current version of class
    :param  typing.List[MigrationInfo] migrations: list of migrations for deserialize old version
    :param old_paths: old name of class with modules
    :param use_parent_migrations: if migrations from parent class should be applied when deserialized object
    :return: class itself if cls parameter is provided. Otherwise,
        one argument function which will consume Type to be registered.

    Examples::

        @register_class(version="0.0.1", migrations=[("0.0.1", rename_key("value", "value1"))])
        class DataClass:
            def __init__(value1, value2)
                self.value1 = value1
                self.value2 = value2

    or::


        class DataClass2:
            def __init__(value1, value2)
                self.value1 = value1
                self.value2 = value2

        register_class(DataClass2, version="0.0.1", migrations=[("0.0.1", rename_key("value", "value1"))])

    """
    return REGISTER.register(cls, version, migrations, old_paths, use_parent_migrations)
