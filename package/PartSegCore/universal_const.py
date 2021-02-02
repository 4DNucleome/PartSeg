from enum import Enum

from PartSegCore.class_generator import enum_register


# noinspection NonAsciiCharacters
class Units(Enum):
    mm = 0
    µm = 1
    nm = 2
    pm = 3

    def __str__(self):
        return self.name.replace("_", " ")


enum_register.register_class(Units)

_UNITS_LIST = ["mm", "µm", "nm", "pm"]
UNIT_SCALE = [10 ** 3, 10 ** 6, 10 ** 9, 10 ** 12]
