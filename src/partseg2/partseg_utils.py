from collections import namedtuple

from .algorithm_description import SegmentationProfile
from .statistics_calculation import StatisticProfile
from project_utils.settings import ProfileEncoder, profile_hook

HistoryElement = namedtuple("HistoryElement", ["algorithm_name", "algorithm_values", "arrays"])


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
