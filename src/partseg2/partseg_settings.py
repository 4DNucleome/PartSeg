from PyQt5.QtCore import pyqtSignal

from project_utils.settings import BaseSettings
import numpy as np

class PartSettings(BaseSettings):
    mask_changed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self._mask = None

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if self._image.shape[:-1] != value.shape:
            raise ValueError("mask do not fit to image")
        self._mask = value
        self.mask_changed.emit()

    def _image_changed(self):
        super()._image_changed()
        self._mask = None

    def load_profiles(self):
        pass

    def components_mask(self):
        return np.array([0] + [1] * self.segmentation.max(), dtype=np.uint8)

def load_project(file_path, settings):
    pass

def save_project(*args, **kwwargs):
    pass

def save_labeled_image(file_path, settings):
    pass