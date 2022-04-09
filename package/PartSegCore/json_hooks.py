import nme

from ._old_json_hooks import part_hook


class PartSegEncoder(nme.NMEEncoder):
    pass


def partseg_object_hook(dkt: dict):
    if "__class__" in dkt:
        return nme.nme_object_hook(dkt)

    if "__ReadOnly__" in dkt or "__Serializable__" in dkt or "__Enum__" in dkt:
        problematic_fields = nme.check_for_errors_in_dkt_values(dkt)
        if problematic_fields:
            dkt["__error__"] = f"Error in fields: {', '.join(problematic_fields)}"
            return dkt
        is_enum = "__Enum__" in dkt
        for el in ("__Enum__", "__Serializable__", "__ReadOnly__"):
            dkt.pop(el, None)
        cls_str = dkt["__subtype__"]
        del dkt["__subtype__"]
        try:
            dkt_migrated = nme.REGISTER.migrate_data(cls_str, {}, dkt)
            cls = nme.REGISTER.get_class(cls_str)
            return cls(**dkt_migrated)
        except Exception:  # pylint: disable=W0703
            dkt["__subtype__"] = cls_str
            dkt["__Enum__" if is_enum else "__Serializable__"] = True

    return part_hook(dkt)
