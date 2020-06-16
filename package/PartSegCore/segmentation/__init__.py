import sys
import os

from .algorithm_base import SegmentationAlgorithm
from .watershed import BaseWatershed
from .threshold import BaseThreshold
from .noise_filtering import NoiseFilteringBase
from .restartable_segmentation_algorithms import RestartableAlgorithm
from .segmentation_algorithm import StackAlgorithm


__all__ = [
    "SegmentationAlgorithm",
    "BaseWatershed",
    "NoiseFilteringBase",
    "BaseThreshold",
    "RestartableAlgorithm",
    "StackAlgorithm",
]


if os.path.basename(sys.argv[0]) in ["sphinx-build", "sphinx-build.exe"]:
    for el in __all__:
        globals()[el].__module__ = __name__
