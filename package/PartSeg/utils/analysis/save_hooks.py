from PartSeg.utils.algorithm_describe_base import SegmentationProfile
from ..analysis.measurement_calculation import MeasurementProfile
from ..json_hooks import ProfileEncoder, profile_hook
from ..analysis.calculation_plan import CalculationPlan, CalculationTree


class PartEncoder(ProfileEncoder):
    def default(self, o):
        if isinstance(o, MeasurementProfile):
            return {"__MeasurementProfile__": True, **o.to_dict()}
        if isinstance(o, CalculationPlan):
            return {"__CalculationPlan__": True, "tree": o.execution_tree, "name": o.name}
        if isinstance(o, CalculationTree):
            return {"__CalculationTree__": True, "operation": o.operation, "children": o.children}
        return super().default(o)


def part_hook(dkt):
    if "__StatisticProfile__" in dkt:
        del dkt["__StatisticProfile__"]
        res = MeasurementProfile(**dkt)
        return res
    if "__MeasurementProfile__" in dkt:
        del dkt["__MeasurementProfile__"]
        res = MeasurementProfile(**dkt)
        return res
    if "__CalculationPlan__" in dkt:
        del dkt["__CalculationPlan__"]
        return CalculationPlan(**dkt)
    if "__CalculationTree__" in dkt:
        return CalculationTree(operation=dkt["operation"], children=dkt["children"])
    return profile_hook(dkt)
