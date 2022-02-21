"""
This module contains utility for class registration, to provide migration information.
"""
import importlib
import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Type, Union

from packaging.version import Version
from packaging.version import parse as parse_version


def class_to_str(cls) -> str:
    if cls.__module__.startswith("pydantic.dataclass"):
        cls = cls.__mro__[1]
        return class_to_str(cls)
    return f"{cls.__module__}.{cls.__qualname__}"


def str_to_version(version):
    return parse_version(version) if isinstance(version, str) else version


MigrationCallable = Callable[[Dict[str, Any]], Dict[str, Any]]
MigrationInfo = Tuple[Version, MigrationCallable]
MigrationStartInfo = Tuple[Union[str, Version], MigrationCallable]


@dataclass(frozen=True)
class TypeInfo:
    base_path: str
    type_: Type
    version: Version
    migrations: List[MigrationInfo]


class MigrationRegistration:
    def __init__(self):
        self._data_dkt: Dict[str, TypeInfo] = {}

    def register(
        self,
        cls: Type = None,
        version: Union[str, Version] = "0.0.0",
        migrations: List[MigrationStartInfo] = None,
        old_paths: List[str] = None,
    ):
        if migrations is None:
            migrations = []
        else:

            migrations = list(
                sorted(map(lambda x: x if isinstance(x, Version) else (parse_version(x[0]), x[1]), migrations))
            )
        if old_paths is None:
            old_paths = []
        version = str_to_version(version)

        if migrations and max(str_to_version(x[0]) for x in migrations) > version:
            raise ValueError("class version lower than in migrations")

        def _register(cls_):
            base_path = class_to_str(cls_)
            type_info = TypeInfo(base_path=base_path, type_=cls_, version=version, migrations=migrations)
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

    def get_version(self, cls: Type) -> Version:
        class_str = class_to_str(cls)
        self._register_missed(class_str=class_str)
        return self._data_dkt[class_str].version

    def get_class(self, class_str: str) -> Type:
        self._register_missed(class_str=class_str)
        return self._data_dkt[class_str].type_

    def migrate_data(
        self, class_str_to_version_dkt: Dict[str, Union[str, Version]], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        for class_str, version in class_str_to_version_dkt.items():
            try:
                self._register_missed(class_str=class_str)
            except ValueError:
                continue
            if isinstance(version, str):
                version = parse_version(version)
            for version_, migration in self._data_dkt[class_str].migrations:
                if version < version_:
                    data = migration(data)

        return data

    def _register_missed(self, class_str):
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


REGISTER = MigrationRegistration()


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
    def _wrapper(func):
        signature = inspect.signature(func)
        if argument_name not in signature.parameters:
            raise RuntimeError("Argument should be accessible using inspect module.")
        arg_index = list(signature.parameters).index(argument_name)

        @wraps(func)
        def _update_from_dict(*args, **kwargs):
            if args and hasattr(args[0], "__argument_class__") and args[0].__argument_class__ is not None:
                if argument_name in kwargs and isinstance(kwargs[argument_name], dict):
                    kwargs = kwargs.copy()
                    kw = REGISTER.migrate_data(
                        {class_to_str(args[0].__argument_class__): "0.0.0"}, kwargs[argument_name]
                    )
                    kwargs[argument_name] = args[0].__argument_class__(**kw)
                elif len(args) > arg_index and isinstance(args[arg_index], dict):
                    args = list(args)
                    kw = REGISTER.migrate_data({class_to_str(args[0].__argument_class__): "0.0.0"}, args[arg_index])
                    args[arg_index] = args[0].__argument_class__(**kw)
            return func(*args, **kwargs)

        return _update_from_dict

    return _wrapper


def register_class(
    cls: Type = None,
    version: Union[str, Version] = "0.0.0",
    migrations: List[MigrationStartInfo] = None,
    old_paths: List[str] = None,
):
    return REGISTER.register(cls, version, migrations, old_paths)
