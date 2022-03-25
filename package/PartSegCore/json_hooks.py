import dataclasses
import enum
import json
from pathlib import Path

import numpy as np
import pydantic

from PartSegImage import Channel

from ._old_json_hooks import part_hook
from .class_register import REGISTER, class_to_str


def add_class_info(obj, dkt):
    dkt["__class__"] = class_to_str(obj.__class__)
    dkt["__class_version_dkt__"] = {
        class_to_str(sup_obj): str(REGISTER.get_version(sup_obj))
        for sup_obj in obj.__class__.__mro__
        if class_to_str(sup_obj)
        not in {
            "object",
            "pydantic.main.BaseModel",
            "pydantic.utils.Representation",
            "enum.Enum",
            "builtins.object",
            "PartSegCore.utils.BaseModel",
        }
    }
    return dkt


class PartSegEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, enum.Enum):
            dkt = {"value": o.value}
            return add_class_info(o, dkt)
        if dataclasses.is_dataclass(o):
            fields = dataclasses.fields(o)
            dkt = {x.name: getattr(o, x.name) for x in fields}
            return add_class_info(o, dkt)

        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, pydantic.BaseModel):
            try:
                dkt = dict(o)
            except (ValueError, TypeError):
                dkt = o.dict()
            return add_class_info(o, dkt)

        if hasattr(o, "as_dict"):
            dkt = o.as_dict()
            return add_class_info(o, dkt)

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Channel):
            return o.value
        if isinstance(o, dict) and "__error__" in o:
            del o["__error__"]  # different environments without same plugins installed
        return super().default(o)


def partseg_object_hook(dkt: dict):
    if "__class__" in dkt:
        # the migration code should be called here
        cls_str = dkt.pop("__class__")
        version_dkt = dkt.pop("__class_version_dkt__") if "__class_version_dkt__" in dkt else {cls_str: "0.0.0"}
        try:
            dkt_migrated = REGISTER.migrate_data(cls_str, version_dkt, dkt)
            cls = REGISTER.get_class(cls_str)
            return cls(**dkt_migrated)
        except Exception as e:  # pylint: disable=W0703
            dkt["__class__"] = cls_str
            dkt["__class_version_dkt__"] = version_dkt
            dkt["__error__"] = e

    if "__ReadOnly__" in dkt or "__Serializable__" in dkt or "__Enum__" in dkt:
        is_enum = "__Enum__" in dkt
        for el in ("__Enum__", "__Serializable__", "__ReadOnly__"):
            dkt.pop(el, None)
        cls_str = dkt["__subtype__"]
        del dkt["__subtype__"]
        try:
            dkt_migrated = REGISTER.migrate_data(cls_str, {}, dkt)
            cls = REGISTER.get_class(cls_str)
            return cls(**dkt_migrated)
        except Exception:  # pylint: disable=W0703
            dkt["__subtype__"] = cls_str
            dkt["__Enum__" if is_enum else "__Serializable__"] = True

    return part_hook(dkt)
