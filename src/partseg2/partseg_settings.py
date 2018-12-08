import typing
from PyQt5.QtCore import pyqtSignal

from partseg2.batch_processing.calculation_plan import CalculationPlan
from partseg2.statistics_calculation import StatisticProfile
from partseg_utils.cmap_utils import CmapProfile
from .algorithm_description import SegmentationProfile
from .partseg_utils import HistoryElement, SegmentationPipeline
from .save_hooks import PartEncoder, part_hook
from .io_functions import save_project, save_cmap, load_project, ProjectTuple
from project_utils_qt.settings import BaseSettings
import numpy as np

MASK_COLORS = {"black": np.array((0, 0, 0)), "white": np.array((255, 255, 255)), "red": np.array((255, 0, 0)),
               "green": np.array((0, 255, 0)), "blue": np.array((0, 0, 255))}


class PartSettings(BaseSettings):
    """
    last_executed_algorithm - parameter for caring last used algorithm
    """
    mask_changed = pyqtSignal()
    json_encoder_class = PartEncoder
    decode_hook = part_hook
    last_executed_algorithm: str

    def __init__(self, json_path):
        super().__init__(json_path)
        self._mask = None
        self.full_segmentation = None
        self.segmentation_history: typing.List[HistoryElement] = []
        self.undo_segmentation_history: typing.List[HistoryElement] = []
        self.last_executed_algorithm = ""

    @property
    def use_physical_unit(self):
        return self.get("use_physical_unit", False)

    def set_use_physical_unit(self, value):
        self.set("use_physical_unit", value)

    @property
    def mask(self):
        if self._image.mask is not None:
            return self._image.mask[0]
        return None

    @mask.setter
    def mask(self, value):
        try:
            self._image.set_mask(value)
            self.mask_changed.emit()
        except ValueError:
            raise ValueError("mask do not fit to image")

    def _image_changed(self):
        super()._image_changed()
        self._mask = None
        self.full_segmentation = None

    def load_profiles(self, file_path):
        pass

    def components_mask(self):
        return np.array([0] + [1] * self.segmentation.max(), dtype=np.uint8)

    def get_project_info(self) -> ProjectTuple:
        algorithm_name = self.last_executed_algorithm
        if algorithm_name:
            algorithm_val = {"name": algorithm_name, "values": self.get(f"algorithms.{algorithm_name}")}
        else:
            algorithm_val = {}
        return ProjectTuple(self.image.file_path, self.image, self.segmentation, self.full_segmentation,
                            self.mask, self.segmentation_history, algorithm_val)

    def save_project(self, file_path):
        dkt = dict()
        dkt["segmentation"] = self.segmentation
        algorithm_name = self.last_executed_algorithm
        dkt["algorithm_parameters"] = {"name": algorithm_name, "values": self.get(f"algorithms.{algorithm_name}")}
        dkt["mask"] = self.mask
        dkt["full_segmentation"] = self.full_segmentation
        dkt["history"] = self.segmentation_history
        dkt["image"] = self.image
        save_project(file_path, **dkt)

    def load_project(self, file_path):
        project_tuple = load_project(file_path)
        im = project_tuple.image
        im.file_path = file_path
        self.image = im
        self.mask = project_tuple.mask
        self.segmentation = project_tuple.segmentation
        self.full_segmentation = project_tuple.full_segmentation
        self.segmentation_history = project_tuple.history
        self.undo_segmentation_history = []
        algorithm_name = project_tuple.algorithm_parameters["name"]
        self.last_executed_algorithm = algorithm_name
        self.set(f"algorithms.{algorithm_name}", project_tuple.algorithm_parameters["values"])

    def save_cmap(self, file_path: str, cmap_profile: CmapProfile):
        save_cmap(file_path, self.image, cmap_profile)

    @property
    def segmentation_pipelines(self) -> typing.Dict[str, SegmentationPipeline]:
        return self.get("segmentation_pipelines", dict())

    @property
    def segmentation_profiles(self) -> typing.Dict[str, SegmentationProfile]:
        return self.get("segmentation_profiles", dict())

    @property
    def batch_plans(self) -> typing.Dict[str, CalculationPlan]:
        return self.get("batch_plans", dict())

    @property
    def statistic_profiles(self) -> typing.Dict[str, StatisticProfile]:
        return self.get("statistic_profiles", dict())

def save_labeled_image(file_path, settings):
    pass
