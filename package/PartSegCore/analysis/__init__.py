import warnings

from PartSegCore.analysis.algorithm_description import AnalysisAlgorithmSelection
from PartSegCore.analysis.analysis_utils import SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.analysis.io_utils import ProjectTuple
from PartSegCore.analysis.load_functions import load_metadata
from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT

__all__ = (
    "AnalysisAlgorithmSelection",
    "ProjectTuple",
    "SegmentationPipeline",
    "SegmentationPipelineElement",
    "MEASUREMENT_DICT",
    "load_metadata",
)


def __getattr__(name):  # pragma: no cover
    if name == "analysis_algorithm_dict":
        warnings.warn(
            "analysis_algorithm_dict is deprecated. Please use AnalysisAlgorithmSelection instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return AnalysisAlgorithmSelection.__register__
    raise AttributeError(f"module {__name__} has no attribute {name}")
