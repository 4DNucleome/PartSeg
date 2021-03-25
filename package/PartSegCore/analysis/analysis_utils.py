import typing
from textwrap import indent

from PartSegCore.algorithm_describe_base import ROIExtractionProfile

from ..class_generator import BaseSerializableClass
from ..mask_create import MaskProperty


class SegmentationPipelineElement(BaseSerializableClass):
    segmentation: ROIExtractionProfile
    mask_property: MaskProperty

    def pretty_print(self, algorithm_dict):
        return (
            indent("Segmentation:\n" + self.segmentation.pretty_print(algorithm_dict).split("\n", 1)[1], "    ")
            + "\n\n"
            + indent(str(self.mask_property), "    ")
        )

    def __str__(self):
        return indent(str(self.segmentation), "    ") + "\n\n" + indent(str(self.mask_property), "    ")

    def __repr__(self):
        return (
            "SegmentationPipelineElement(segmentation="
            f"{repr(self.segmentation)},mask_property={repr(self.mask_property)})"
        )


class SegmentationPipeline(BaseSerializableClass):
    name: str
    segmentation: ROIExtractionProfile
    mask_history: typing.List[SegmentationPipelineElement]

    def pretty_print(self, algorithm_dict):
        return (
            (
                f"Segmentation pipeline name: {self.name}\n"
                + "\n––––––––––––––\n".join(x.pretty_print(algorithm_dict) for x in self.mask_history)
            )
            + "\n––––––––––––––\nLast segmentation:\n"
        ) + self.segmentation.pretty_print(algorithm_dict).split("\n", 1)[1]

    def __str__(self):
        return (
            (
                f"Segmentation pipeline name: {self.name}\n"
                + "\n––––––––––––––\n".join(str(x) for x in self.mask_history)
            )
            + "\n––––––––––––––\nLast segmentation\n"
        ) + str(self.segmentation)

    def __repr__(self):
        return (
            f"SegmentationPipeline(name={self.name},\nmask_history={self.mask_history},\n"
            f"segmentation={repr(self.segmentation)})"
        )
