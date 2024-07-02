from enum import Enum

from local_migrator import register_class


# noinspection NonAsciiCharacters
@register_class(old_paths=["PartSeg.utils.universal_const.Units"])
class Units(Enum):
    mm = 0
    µm = 1  # noqa: PLC2401
    nm = 2
    pm = 3

    def __str__(self):
        return self.name.replace("_", " ")


_UNITS_LIST = ["mm", "µm", "nm", "pm"]
UNIT_SCALE = [10**3, 10**6, 10**9, 10**12]
