from textwrap import indent

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.mask_create import MaskProperty
from PartSegCore.utils import BaseModel


class SegmentationPipelineElement(BaseModel):
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
        return f"SegmentationPipelineElement(segmentation={self.segmentation!r},mask_property={self.mask_property!r})"


class SegmentationPipeline(BaseModel):
    name: str
    segmentation: ROIExtractionProfile
    mask_history: list[SegmentationPipelineElement]

    def pretty_print(self, algorithm_dict):
        return (
            (
                f"Segmentation pipeline name: {self.name}\n"
                + "\n--------------\n".join(x.pretty_print(algorithm_dict) for x in self.mask_history)
            )
            + "\n--------------\nLast segmentation:\n"
        ) + self.segmentation.pretty_print(algorithm_dict).split("\n", 1)[1]

    def __str__(self):
        return (
            (
                f"Segmentation pipeline name: {self.name}\n"
                + "\n--------------\n".join(str(x) for x in self.mask_history)
            )
            + "\n--------------\nLast segmentation\n"
        ) + str(self.segmentation)

    def __repr__(self):
        return (
            f"SegmentationPipeline(name={self.name},\nmask_history={self.mask_history},\n"
            f"segmentation={self.segmentation!r})"
        )
