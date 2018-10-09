from abc import ABC

from qt_import import QThread, pyqtSignal
import numpy as np




class SegmentationAlgorithm(object):
    def __init__(self):
        super().__init__()
        self.image = None
        self.segmentation = None
        self.spacing = None
        self.use_psychical_unit = False
        self.unit_scalar = 1

    def _clean(self):
        self.image = None
        self.segmentation = None

    def calculation_run(self, report_fun):
        raise NotImplementedError()

    def get_info_text(self):
        raise NotImplementedError()

    def set_size_information(self, spacing, use_physical_unit, unit_scalar):
        self.unit_scalar = unit_scalar
        self.spacing = spacing
        self.use_psychical_unit = use_physical_unit


    """def set_parameters_wait(self, *args, **kwargs):
        self.mutex.lock()
        if self.isRunning():
            self.cache = args, kwargs
            self.clean_later = False
        else:
            self.set_parameters(*args, **kwargs)
        self.mutex.unlock()"""

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()
