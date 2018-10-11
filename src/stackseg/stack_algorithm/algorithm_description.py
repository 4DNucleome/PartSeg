from abc import ABCMeta, abstractmethod
from os import path
from typing import Type

import numpy as np
import tifffile
from PyQt5.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QWidget, QVBoxLayout, QLabel, QFormLayout, \
    QAbstractSpinBox, QScrollArea
from six import with_metaclass

from partseg.io_functions import save_stack_segmentation, load_stack_segmentation
#from qt_import import QDoubleSpinBox, QSpinBox, QComboBox, QWidget, QFormLayout, QAbstractSpinBox, QCheckBox, QThread, \
#    pyqtSignal, QLabel, QVBoxLayout
from project_utils.algorithms_description import AlgorithmProperty
from project_utils.image_operations import RadiusType

from .threshold_algorithm import ThresholdAlgorithm, ThresholdPreview, \
    AutoThresholdAlgorithm




only_threshold_algorithm = [AlgorithmProperty("threshold", "Threshold", 1000, (0, 10 ** 6), 100),
                            AlgorithmProperty("use_gauss", "Use gauss", RadiusType.NO, None),
                            AlgorithmProperty("gauss_radius", "Gauss radius", 1.0, (0, 10), 0.1)]

threshold_algorithm = [AlgorithmProperty("threshold", "Threshold", 10000, (0, 10 ** 6), 100),
                       AlgorithmProperty("minimum_size", "Minimum size", 8000, (20, 10 ** 6), 1000),
                       AlgorithmProperty("close_holes", "Close small holes", True, (True, False)),
                       AlgorithmProperty("close_holes_size", "Small holes size", 200, (0, 10**3), 10),
                       AlgorithmProperty("smooth_border", "Smooth borders", True, (True, False)),
                       AlgorithmProperty("smooth_border_radius", "Smooth borders radius", 2, (0, 20), 1),
                       AlgorithmProperty("use_gauss", "Use gauss", RadiusType.NO, None),
                       AlgorithmProperty("gauss_radius", "Gauss radius", 1.0, (0, 10), 0.1),
                       AlgorithmProperty("side_connection", "Connect only sides", False, (True, False))]

auto_threshold_algorithm = [AlgorithmProperty("suggested_size", "Suggested size", 200000, (0, 10 ** 6), 1000),
                            AlgorithmProperty("threshold", "Minimum threshold", 10000, (0, 10 ** 6), 100),
                            AlgorithmProperty("minimum_size", "Minimum size", 8000, (20, 10 ** 6), 1000),
                            AlgorithmProperty("close_holes", "Close small holes", True, (True, False)),
                            AlgorithmProperty("close_holes_size", "Small holes size", 200, (0, 10 ** 3), 10),
                            AlgorithmProperty("smooth_border", "Smooth borders", True, (True, False)),
                            AlgorithmProperty("smooth_border_radius", "Smooth borders radius", 2, (0, 20), 1),
                            AlgorithmProperty("use_gauss", "Use gauss", RadiusType.NO, None),
                            AlgorithmProperty("gauss_radius", "Gauss radius", 1.0, (0, 10), 0.1),
                            AlgorithmProperty("side_connection", "Connect only sides", False, (True, False))]

stack_algorithm_dict = {
    "Threshold": (threshold_algorithm, ThresholdAlgorithm),
    "Only Threshold": (only_threshold_algorithm, ThresholdPreview),
    "Auto Threshold": (auto_threshold_algorithm, AutoThresholdAlgorithm)
}
