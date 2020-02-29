import typing
from textwrap import indent

from PartSegCore.algorithm_describe_base import SegmentationProfile
from ..class_generator import BaseSerializableClass
from ..mask_create import MaskProperty


class SegmentationPipelineElement(BaseSerializableClass):
    segmentation: SegmentationProfile
    mask_property: MaskProperty

    def pretty_print(self, algorithm_dict):
        return (
            indent("Segmentation:\n" + self.segmentation.pretty_print(algorithm_dict).split("\n", 1)[1], "    ")
            + "\n\n"
            + indent(str(self.mask_property), "    ")
        )

    def __str__(self):
        return indent(str(self.segmentation), "    ") + "\n\n" + indent(str(self.mask_property), "    ")


class SegmentationPipeline(BaseSerializableClass):
    name: str
    segmentation: SegmentationProfile
    mask_history: typing.List[SegmentationPipelineElement]

    def pretty_print(self, algorithm_dict):
        return (
            f"Segmentation pipeline name: {self.name}\n"
            + "\n––––––––––––––\n".join([x.pretty_print(algorithm_dict) for x in self.mask_history])
            + "\n––––––––––––––\nLast segmentation:\n"
            + self.segmentation.pretty_print(algorithm_dict).split("\n", 1)[1]
        )

    def __str__(self):
        return (
            f"Segmentation pipeline name: {self.name}\n"
            + "\n––––––––––––––\n".join([str(x) for x in self.mask_history])
            + "\n––––––––––––––\nLast segmentation\n"
            + str(self.segmentation)
        )
