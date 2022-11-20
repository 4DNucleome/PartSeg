import warnings

from PartSegCore.algorithm_describe_base import AlgorithmSelection
from PartSegCore.segmentation.restartable_segmentation_algorithms import (
    BorderRim,
    LowerThresholdAlgorithm,
    LowerThresholdFlowAlgorithm,
    MaskDistanceSplit,
    OtsuSegment,
    RangeThresholdAlgorithm,
    UpperThresholdAlgorithm,
    UpperThresholdFlowAlgorithm,
)


class AnalysisAlgorithmSelection(
    AlgorithmSelection,
    class_methods=["support_time", "support_z"],
    methods=["set_image", "set_mask", "get_info_text", "calculation_run"],
):
    """Register for segmentation method visible in PartSeg ROI Analysis."""


AnalysisAlgorithmSelection.register(LowerThresholdAlgorithm)
AnalysisAlgorithmSelection.register(UpperThresholdAlgorithm)
AnalysisAlgorithmSelection.register(RangeThresholdAlgorithm)
AnalysisAlgorithmSelection.register(LowerThresholdFlowAlgorithm, old_names=["Lower threshold flow"])
AnalysisAlgorithmSelection.register(UpperThresholdFlowAlgorithm, old_names=["Upper threshold flow"])
AnalysisAlgorithmSelection.register(OtsuSegment)
AnalysisAlgorithmSelection.register(BorderRim)
AnalysisAlgorithmSelection.register(MaskDistanceSplit, old_names=["Split Mask on Part"])


def __getattr__(name):  # pragma: no cover
    if name == "analysis_algorithm_dict":
        warnings.warn(
            "analysis_algorithm_dict is deprecated. Please use AnalysisAlgorithmSelection instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return AnalysisAlgorithmSelection.__register__
    raise AttributeError(f"module {__name__} has no attribute {name}")
