from enum import Enum


# noinspection NonAsciiCharacters
class Units(Enum):
    mm = 0
    µm = 1
    nm = 2
    pm = 3

    def __str__(self):
        return self.name.replace("_", " ")


_UNITS_LIST = ["mm", "µm", "nm", "pm"]
UNIT_SCALE = [10**3, 10**6, 10**9, 10**12]
