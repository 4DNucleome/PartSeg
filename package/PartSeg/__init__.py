from packaging.version import parse

from PartSeg.changelog import changelog
from PartSeg.version import version

__version__ = version
parsed_version = parse(version)

__author__ = "Grzegorz Bokota"

APP_NAME = "PartSeg"
APP_LAB = "LFSG"
MASK_NAME = "ROI Mask"
ANALYSIS_NAME = "ROI Analysis"

__all__ = (
    "ANALYSIS_NAME",
    "APP_LAB",
    "APP_NAME",
    "MASK_NAME",
    "__author__",
    "__version__",
    "changelog",
    "parsed_version",
)
