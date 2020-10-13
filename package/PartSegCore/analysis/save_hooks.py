from ..json_hooks import ProfileEncoder, profile_hook
from .calculation_plan import CalculationPlan, CalculationTree
from .measurement_calculation import MeasurementProfile


class PartEncoder(ProfileEncoder):
    # pylint: disable=E0202
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
    if (
        "__subtype__" in dkt
        and "statistic_profile" in dkt
        and dkt["__subtype__"]
        in (
            "PartSegCore.analysis.calculation_plan.MeasurementCalculate",
            "PartSeg.utils.analysis.calculation_plan.StatisticCalculate",
        )
    ):
        dkt["measurement_profile"] = dkt["statistic_profile"]
        del dkt["statistic_profile"]
    return profile_hook(dkt)
