from collections import namedtuple

from PyQt5.QtCore import pyqtSignal

from partseg.statistics_calculation import StatisticProfile
from project_utils.settings import BaseSettings, ProfileEncoder, profile_hook
import numpy as np

MASK_COLORS = {"black":np.array((0,0,0)), "white": np.array((255, 255, 255)), "red": np.array((255, 0, 0)),
               "green": np.array((0, 255, 0)), "blue": np.array((0, 0, 255))}

class PartEncoder(ProfileEncoder):
    def default(self, o):
        if isinstance(o, StatisticProfile):
            return {"__StatisticProfile__": True, **o.to_dict()}
        return super().default(o)


def part_hook(_, dkt):
    if "__StatisticProfile__" in dkt:
        del dkt["__StatisticProfile__"]
        res = StatisticProfile(**dkt)
        return res
    return profile_hook(_, dkt)


class PartSettings(BaseSettings):
    mask_changed = pyqtSignal()
    json_encoder_class = PartEncoder
    decode_hook = part_hook

    def __init__(self, json_path):
        super().__init__(json_path)
        self._mask = None
        self.full_segmentation = None
        self.segmentation_history = []
        self.undo_segmentation_history = []

    @property
    def use_physical_unit(self):
        return self.get("use_physical_unit", False)

    def set_use_physical_unit(self, value):
        self.set("use_physical_unit", value)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        if value is not None and self._image.shape[:-1] != value.shape:
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


HistoryElement = namedtuple("HistoryElement", ["algorithm_name", "algorithm_values", "segmentation", "mask"])