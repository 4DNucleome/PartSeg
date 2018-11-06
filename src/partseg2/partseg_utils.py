import typing
from io import BytesIO
from textwrap import indent

from partseg2.batch_processing.calculation_plan import CalculationPlan, CalculationTree
from project_utils.class_generator import BaseReadonlyClass
from project_utils.mask_create import MaskProperty
from .algorithm_description import SegmentationProfile
from .statistics_calculation import StatisticProfile
from project_utils.settings import ProfileEncoder, profile_hook
import numpy as np


# HistoryElement = namedtuple("HistoryElement", ["algorithm_name", "algorithm_values", "mask_property", "arrays"])

class HistoryElement(BaseReadonlyClass):
    algorithm_name: str
    algorithm_values: typing.Dict[str, typing.Any]
    mask_property: MaskProperty
    arrays: BytesIO

    @classmethod
    def create(cls, segmentation: np.ndarray, full_segmentation: np.ndarray, mask: typing.Union[np.ndarray, None],
               algorithm_name: str, algorithm_values: dict, mask_property: MaskProperty):
        arrays = BytesIO()
        arrays_dict = {"segmentation": segmentation, "full_segmentation": full_segmentation}
        if mask is not None:
            arrays_dict["mask"] = mask
        np.savez_compressed(arrays, **arrays_dict)
        arrays.seek(0)
        return HistoryElement(algorithm_name=algorithm_name, algorithm_values=algorithm_values,
                              mask_property=mask_property, arrays=arrays)


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


def part_hook(_, dkt):
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
               "\n––––––––––––––\n".join([str(x) for x in self.mask_history]) + \
               "––––––––––––––\nLast segmentation\n" + str(self.segmentation)
