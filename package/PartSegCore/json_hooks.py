import local_migrator

from PartSegCore._old_json_hooks import part_hook

PLUGINS_STR = "plugins."
PLUGINS_STR_LEN = len(PLUGINS_STR)


class PartSegEncoder(local_migrator.Encoder):
    pass


def _validate_plugin_prefix(dkt: dict):
    if dkt["__class__"].startswith(PLUGINS_STR):
        # workaround for plans exported from an old PartSeg bundle
        dkt["__class__"] = dkt["__class__"][PLUGINS_STR_LEN:]
        if "__class_version_dkt__" in dkt:
            for name, value in list(dkt["__class_version_dkt__"].items()):
                if name.startswith(PLUGINS_STR):
                    dkt["__class_version_dkt__"][name[PLUGINS_STR_LEN:]] = value
                    del dkt["__class_version_dkt__"][name]


def partseg_object_hook(dkt: dict):
    if "__class__" in dkt:
        _validate_plugin_prefix(dkt)
        return local_migrator.object_hook(dkt)

    if "__ReadOnly__" in dkt or "__Serializable__" in dkt or "__Enum__" in dkt:
        if problematic_fields := local_migrator.check_for_errors_in_dkt_values(dkt):
            dkt["__error__"] = f"Error in fields: {', '.join(problematic_fields)}"
            return dkt
        is_enum = "__Enum__" in dkt
        for el in ("__Enum__", "__Serializable__", "__ReadOnly__"):
            dkt.pop(el, None)
        cls_str = dkt["__subtype__"]
        del dkt["__subtype__"]
        try:
            dkt_migrated = local_migrator.REGISTER.migrate_data(cls_str, {}, dkt)
            cls = local_migrator.REGISTER.get_class(cls_str)
            return cls(**dkt_migrated)
        except Exception:  # pylint: disable=broad-except
            dkt["__subtype__"] = cls_str
            dkt["__Enum__" if is_enum else "__Serializable__"] = True

    return part_hook(dkt)
