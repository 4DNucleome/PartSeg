"""
This module contains utility for class registration, to provide migration information.
"""
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type, Union

from packaging.version import Version
from packaging.version import parse as parse_version


def class_to_str(cls) -> str:
    if cls.__module__.startswith("pydantic.dataclass"):
        cls = cls.__mro__[1]
        return class_to_str(cls)
    return cls.__module__ + "." + cls.__name__


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
        if isinstance(version, str):
            version = parse_version(version)

        if migrations:
            version = max(version, max(map(lambda x: x[0], migrations)))

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

    def migrate_data(self, class_str, version: Union[str, Version], data: Dict[str, Any]) -> Dict[str, Any]:
        self._register_missed(class_str=class_str)
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
        module = importlib.import_module(module_name)
        if class_str in self._data_dkt:
            return
        class_ = getattr(module, class_name)
        self.register(class_)


REGISTER = MigrationRegistration()


def register_class(
    cls: Type = None,
    version: Union[str, Version] = "0.0.0",
    migrations: List[MigrationStartInfo] = None,
    old_paths: List[str] = None,
):
    return REGISTER.register(cls, version, migrations, old_paths)
