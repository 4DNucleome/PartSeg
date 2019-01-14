from .algorithm_description import SegmentationProfile
from .statistics_calculation import StatisticProfile
from ..project_utils_qt.settings import ProfileEncoder, profile_hook
from .batch_processing.calculation_plan import CalculationPlan, CalculationTree


class PartEncoder(ProfileEncoder):
    def default(self, o):
        if isinstance(o, StatisticProfile):
            return {"__StatisticProfile__": True, **o.to_dict()}
        if isinstance(o, SegmentationProfile):
            return {"__SegmentationProperty__": True, "name": o.name, "algorithm": o.algorithm, "values": o.values}
        if isinstance(o, CalculationPlan):
            return {"__CalculationPlan__": True, "tree": o.execution_tree, "name": o.name}
        if isinstance(o, CalculationTree):
            return {"__CalculationTree__": True, "operation": o.operation, "children": o.children}
        return super().default(o)


def part_hook(dkt):
    if "__StatisticProfile__" in dkt:
        del dkt["__StatisticProfile__"]
        res = StatisticProfile(**dkt)
        return res
    if "__SegmentationProperty__" in dkt:
        del dkt["__SegmentationProperty__"]
        res = SegmentationProfile(**dkt)
        return res
    if "__CalculationPlan__" in dkt:
        del dkt["__CalculationPlan__"]
        return CalculationPlan(**dkt)
    if "__CalculationTree__" in dkt:
        return CalculationTree(operation=dkt["operation"], children=dkt["children"])
    return profile_hook(dkt)
