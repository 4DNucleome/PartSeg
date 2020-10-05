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
    "__author__",
    "parsed_version",
    "__version__",
    "changelog",
    "APP_NAME",
    "MASK_NAME",
    "APP_LAB",
    "ANALYSIS_NAME",
)
