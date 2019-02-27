import sys
from enum import Enum

from PartSeg.utils.channel_class import Channel
from ..segmentation.algorithm_describe_base import Register, AlgorithmProperty, AlgorithmDescribeBase
from ..segmentation.restartable_segmentation_algorithms import final_algorithm_list
from typing import Dict

analysis_algorithm_dict = Register()

assert hasattr(analysis_algorithm_dict, "register")

for el in final_algorithm_list:
    analysis_algorithm_dict.register(el)


class SegmentationProfile(object):
    def __init__(self, name, algorithm, values):
        self.name = name
        self.algorithm = algorithm
        self.values = values

    def pretty_print(self, algorithm_dict):
        try:
            algorithm = algorithm_dict[self.algorithm]
        except KeyError:
            return str(self)
        return "Segmentation profile name: " + self.name + "\nAlgorithm: " + \
               self.algorithm + "\n" + self._pretty_print(self.values, algorithm.get_fields_dict())

    @classmethod
    def _pretty_print(cls, values: dict, translate_dict: Dict[str, AlgorithmProperty], indent=0):
        res = ""
        for k, v in values.items():
            if k in translate_dict:
                desc = translate_dict[k]
                res += " " * indent + desc.user_name + ": "
                if issubclass(desc.value_type, Channel):
                    res += str(Channel(v))
                elif issubclass(desc.value_type, AlgorithmDescribeBase):
                    res += desc.possible_values[v["name"]].get_name()
                    if v['values']:
                        res += "\n"
                        res += cls._pretty_print(v["values"], desc.possible_values[v["name"]].get_fields_dict(), indent+2)
                else:
                    res += str(v)
            else:
                raise ValueError("wrong argument")
            res += "\n"
        return res[:-1]


    @classmethod
    def print_dict(cls, dkt, indent=0, name: str = ""):
        if isinstance(dkt, Enum):
            return dkt.name
        if not isinstance(dkt, dict):
            # FIXME update in future method of proper printing channel number
            if name.startswith("channel") and isinstance(dkt, int):
                return dkt + 1
            return dkt
        return "\n" + "\n".join(
            [" " * indent + f"{k.replace('_', ' ')}: {cls.print_dict(v, indent + 2, k)}"
             for k, v in dkt.items()])

    def __str__(self):
        return "Segmentation profile name: " + self.name + "\nAlgorithm: " + \
               self.algorithm + self.print_dict(self.values)

    def __repr__(self):
        return f"SegmentationProfile(name={self.name}, algorithm={self.algorithm}, values={self.values})"
