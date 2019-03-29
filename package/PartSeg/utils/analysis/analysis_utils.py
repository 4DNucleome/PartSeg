import typing
from io import BytesIO
from textwrap import indent
import numpy as np
from ..class_generator import BaseSerializableClass
from ..mask_create import MaskProperty
from PartSeg.utils.algorithm_describe_base import SegmentationProfile


class HistoryElement(BaseSerializableClass):
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


class SegmentationPipelineElement(BaseSerializableClass):
    segmentation: SegmentationProfile
    mask_property: MaskProperty

    def pretty_print(self, algorithm_dict):
        return indent("Segmentation:\n" + self.segmentation.pretty_print(algorithm_dict).split("\n", 1)[1], '    ') +\
               "\n\n" + indent(str(self.mask_property), '    ')

    def __str__(self):
        return indent(str(self.segmentation), '    ') + "\n\n" + indent(str(self.mask_property), '    ')


class SegmentationPipeline(BaseSerializableClass):
    name: str
    segmentation: SegmentationProfile
    mask_history: typing.List[SegmentationPipelineElement]

    def pretty_print(self, algorithm_dict):
        return f"Segmentation pipeline name: {self.name}\n" + \
               "\n––––––––––––––\n".join([x.pretty_print(algorithm_dict) for x in self.mask_history]) + \
               "\n––––––––––––––\nLast segmentation:\n" + \
               self.segmentation.pretty_print(algorithm_dict).split("\n", 1)[1]

    def __str__(self):
        return f"Segmentation pipeline name: {self.name}\n" + \
               "\n––––––––––––––\n".join([str(x) for x in self.mask_history]) + \
               "\n––––––––––––––\nLast segmentation\n" + str(self.segmentation)
