from enum import Enum
from ..partseg_utils.segmentation.algorithm_describe_base import Register
from ..partseg_utils.segmentation.restartable_segmentation_algorithms import final_algorithm_list

part_algorithm_dict = Register()

assert hasattr(part_algorithm_dict, "register")

for el in final_algorithm_list:
    part_algorithm_dict.register(el)


class SegmentationProfile(object):
    def __init__(self, name, algorithm, values):
        self.name = name
        self.algorithm = algorithm
        self.values = values

    @classmethod
    def print_dict(cls, dkt, indent=0):
        if isinstance(dkt, Enum):
            return dkt.name
        if not isinstance(dkt, dict):
            return dkt
        return "\n" + "\n".join(
            [" " * indent + f"{k.replace('_', ' ')}: {cls.print_dict(v, indent + 2)}"
             for k, v in dkt.items()])

    def __str__(self):
        return "Segmentation profile name: " + self.name + "\nAlgorithm: " + \
               self.algorithm + self.print_dict(self.values)

    def __repr__(self):
        return f"SegmentationProfile(name={self.name}, algorithm={self.algorithm}, values={self.values})"
