import typing
from io import BytesIO
from textwrap import indent
import numpy as np
from ..partseg_utils.class_generator import BaseReadonlyClass
from ..partseg_utils.mask_create import MaskProperty
from .algorithm_description import SegmentationProfile


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
               "\n––––––––––––––\nLast segmentation\n" + str(self.segmentation)
