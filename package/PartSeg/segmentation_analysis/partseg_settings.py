import typing
from copy import deepcopy
from qtpy.QtWidgets import QMessageBox, QWidget
from qtpy.QtCore import Signal

from PartSeg.tiff_image import Image
from PartSeg.utils.analysis.calculation_plan import CalculationPlan
from PartSeg.utils.analysis.io_utils import ProjectTuple, MaskInfo
from PartSeg.utils.analysis.statistics_calculation import StatisticProfile
from PartSeg.utils.algorithm_describe_base import SegmentationProfile
from PartSeg.utils.analysis.analysis_utils import HistoryElement, SegmentationPipeline
from PartSeg.utils.analysis.save_hooks import PartEncoder, part_hook
from PartSeg.utils.analysis.save_functions import save_project
from PartSeg.utils.analysis.load_functions import load_project
from ..project_utils_qt.settings import BaseSettings, SaveSettingsDescription
from PartSeg.utils.json_hooks import ProfileDict
import numpy as np

MASK_COLORS = {"white": np.array((255, 255, 255)), "black": np.array((0, 0, 0)), "red": np.array((255, 0, 0)),
               "green": np.array((0, 255, 0)), "blue": np.array((0, 0, 255))}


class PartSettings(BaseSettings):
    """
    last_executed_algorithm - parameter for caring last used algorithm
    """
    mask_changed = Signal()
    json_encoder_class = PartEncoder
    decode_hook = staticmethod(part_hook)
    last_executed_algorithm: str
    save_locations_keys = ["open_directory", "save_directory", "export_directory", "batch_plan_directory",
                           "multiple_open_directory"]

    def __init__(self, json_path):
        super().__init__(json_path)
        self._mask = None
        self.full_segmentation = None
        self.segmentation_history: typing.List[HistoryElement] = []
        self.undo_segmentation_history: typing.List[HistoryElement] = []
        self.last_executed_algorithm = ""
        self.segmentation_pipelines_dict = ProfileDict()
        self.segmentation_profiles_dict = ProfileDict()
        self.batch_plans_dict = ProfileDict()
        self.statistic_profiles_dict = ProfileDict()

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

    def get_project_info(self) -> ProjectTuple:
        algorithm_name = self.last_executed_algorithm
        if algorithm_name:
            algorithm_val = {"name": algorithm_name, "values": deepcopy(self.get(f"algorithms.{algorithm_name}"))}
        else:
            algorithm_val = {}
        return ProjectTuple(self.image.file_path, self.image.substitute(), self.segmentation, self.full_segmentation,
                            self.mask, self.segmentation_history[:], algorithm_val)

    def set_project_info(self, data: typing.Union[ProjectTuple, MaskInfo]):
        if isinstance(data, ProjectTuple):
            if self.image.file_path == data.image.file_path and self.image.shape == data.image.shape:
                if data.segmentation is not None:
                    try:
                        self.image.fit_array_to_image(data.segmentation)
                        self.mask = data.mask
                    except:
                        self.image = data.image.substitute()
                else:
                    self.mask = data.mask
            else:
                self.image = data.image.substitute()
            self.segmentation = data.segmentation
            self.full_segmentation = data.full_segmentation
            self.segmentation_history = data.history[:]
            if data.algorithm_parameters:
                self.last_executed_algorithm = data.algorithm_parameters["name"]
                self.set(f"algorithms.{self.last_executed_algorithm}", deepcopy(data.algorithm_parameters["values"]))
                self.algorithm_changed.emit()
        if isinstance(data, MaskInfo):
            self.mask = data.mask_array

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

    def get_save_list(self) -> typing.List[SaveSettingsDescription]:
        return super().get_save_list() + [
            SaveSettingsDescription("segmentation_pipeline_save.json", self.segmentation_pipelines_dict),
            SaveSettingsDescription("segmentation_profiles_save.json", self.segmentation_profiles_dict),
            SaveSettingsDescription("statistic_profiles_save.json", self.statistic_profiles_dict),
            SaveSettingsDescription("batch_plans_save.json", self.batch_plans_dict),
        ]

    @staticmethod
    def verify_image(image: Image, silent=True) -> typing.Union[Image, bool]:
        if image.is_time:
            if image.is_stack:
                if silent:
                    raise ValueError("Do not support time and stack image")
                else:
                    wid = QWidget()
                    QMessageBox.warning(wid, "image error", "Do not support time and stack image")
                    return False
            if silent:
                return image.swap_time_and_stack()
            else:
                wid = QWidget()
                res = QMessageBox.question(wid,
                    "Not supported",
                    "Time data are currently not supported. Maybe You would like to treat time as z-stack",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if res == QMessageBox.Yes:
                    return image.swap_time_and_stack()
                return False
        return True

    @property
    def segmentation_pipelines(self) -> typing.Dict[str, SegmentationPipeline]:
        return self.segmentation_pipelines_dict.get(self.current_segmentation_dict, dict())

    @property
    def segmentation_profiles(self) -> typing.Dict[str, SegmentationProfile]:
        return self.segmentation_profiles_dict.get(self.current_segmentation_dict, dict())

    @property
    def batch_plans(self) -> typing.Dict[str, CalculationPlan]:
        return self.batch_plans_dict.get(self.current_segmentation_dict, dict())

    @property
    def statistic_profiles(self) -> typing.Dict[str, StatisticProfile]:
        return self.statistic_profiles_dict.get(self.current_segmentation_dict, dict())
