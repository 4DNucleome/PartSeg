from collections import namedtuple
import typing
from io import BytesIO
from textwrap import indent
from project_utils.class_generator import BaseReadonlyClass
from project_utils.mask_create import MaskProperty
from .algorithm_description import SegmentationProfile
from .statistics_calculation import StatisticProfile
from project_utils.settings import ProfileEncoder, profile_hook

# HistoryElement = namedtuple("HistoryElement", ["algorithm_name", "algorithm_values", "mask_property", "arrays"])

class HistoryElement(BaseReadonlyClass):
    algorithm_name: str
    algorithm_values: typing.Dict[str, typing.Any]
    mask_property: MaskProperty
    arrays: BytesIO

class PartEncoder(ProfileEncoder):
    def default(self, o):
        if isinstance(o, StatisticProfile):
            return {"__StatisticProfile__": True, **o.to_dict()}
        if isinstance(o, SegmentationProfile):
            return {"__SegmentationProperty__": True, "name": o.name, "algorithm": o.algorithm, "values": o.values}
        return super().default(o)


def part_hook(_, dkt):
    if "__StatisticProfile__" in dkt:
        del dkt["__StatisticProfile__"]
        res = StatisticProfile(**dkt)
        return res
    if "__SegmentationProperty__" in dkt:
        del dkt["__SegmentationProperty__"]
        res = SegmentationProfile(**dkt)
        return res
    return profile_hook(_, dkt)


class SegmentationPipelineElement(BaseReadonlyClass):
    segmentation: SegmentationProfile
    mask_property: MaskProperty

    def __str__(self):
        return indent(str(self.segmentation), '    ') + "\n\n" + indent(str(self.mask_property), '    ')

class SegmentationPipeline(BaseReadonlyClass):
    name: str
    segmentation: SegmentationProfile
    mask_history: typing.List[SegmentationPipelineElement]

    def __str__(self):
        return f"Segmentation pipeline name: {self.name}\n" + \
               "––––––––––––––\n".join([str(x) for x in self.mask_history]) + "––––––––––––––\nLast segmentation\n" +\
               str(self.segmentation)
