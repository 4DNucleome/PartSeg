import warnings

from PartSegCore.algorithm_describe_base import AlgorithmSelection
from PartSegCore.segmentation.segmentation_algorithm import (
    AutoThresholdAlgorithm,
    CellFromNucleusFlow,
    MorphologicalWatershed,
    SplitImageOnParts,
    ThresholdAlgorithm,
    ThresholdFlowAlgorithm,
    ThresholdPreview,
)


class MaskAlgorithmSelection(
    AlgorithmSelection,
    class_methods=["support_time", "support_z"],
    methods=["set_image", "set_mask", "get_info_text", "calculation_run"],
):
    """Register for segmentation method visible in PartSeg ROI Mask."""


MaskAlgorithmSelection.register(ThresholdAlgorithm)
MaskAlgorithmSelection.register(ThresholdFlowAlgorithm)
MaskAlgorithmSelection.register(ThresholdPreview)
MaskAlgorithmSelection.register(AutoThresholdAlgorithm)
MaskAlgorithmSelection.register(CellFromNucleusFlow)
MaskAlgorithmSelection.register(MorphologicalWatershed)
MaskAlgorithmSelection.register(SplitImageOnParts)


def __getattr__(name):  # pragma: no cover
    if name == "mask_algorithm_dict":
        warnings.warn(
            "mask_algorithm_dict is deprecated. Please use MaskAlgorithmSelection instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return MaskAlgorithmSelection.__register__
    raise AttributeError(f"module {__name__} has no attribute {name}")
