from packaging.version import parse

from PartSeg.version import version
from PartSeg.changelog import changelog

__version__ = version
parsed_version = parse(version)

__author__ = "Grzegorz Bokota"

APP_NAME = "PartSeg"
APP_LAB = "LFSG"
MASK_NAME = "Mask Segmentation"
SEGMENTATION_NAME = "ROI Analysis"
