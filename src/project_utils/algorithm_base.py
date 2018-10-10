from abc import ABC

from project_utils.image_operations import gaussian
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

    def get_gauss(self, gauss_type, gauss_radius):
        spacing = self.spacing
        if gauss_type == "2d":
            layer = True
            if len(self.spacing) == 3:
                spacing = self.spacing[1:]
        elif gauss_type == "3d":
            layer = False
        elif gauss_type == "No":
            return self.image
        else:
            raise ValueError(f"Wrong value of gauss_type: {gauss_type}")
        base = min(spacing)
        if base != max(spacing):
            ratio = [x / base for x in self.spacing]
            gauss_radius = [gauss_radius / r for r in ratio]
        return gaussian(self.image, gauss_radius, layer=layer)


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
