import typing
from copy import deepcopy

import numpy as np
from qtpy.QtCore import Signal

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.analysis_utils import SegmentationPipeline
from PartSegCore.analysis.calculation_plan import CalculationPlan
from PartSegCore.analysis.io_utils import MaskInfo, ProjectTuple
from PartSegCore.analysis.load_functions import load_metadata
from PartSegCore.analysis.measurement_calculation import MeasurementProfile
from PartSegCore.analysis.save_hooks import PartEncoder
from PartSegCore.io_utils import HistoryElement
from PartSegCore.json_hooks import ProfileDict
from PartSegCore.roi_info import ROIInfo

from ..common_backend.base_settings import BaseSettings, SaveSettingsDescription

MASK_COLORS = {
    "white": np.array((255, 255, 255)),
    "black": np.array((0, 0, 0)),
    "red": np.array((255, 0, 0)),
    "green": np.array((0, 255, 0)),
    "blue": np.array((0, 0, 255)),
}


class PartSettings(BaseSettings):
    """
    last_executed_algorithm - parameter for caring last used algorithm
    """

    compare_segmentation_change = Signal(ROIInfo)
    json_encoder_class = PartEncoder
    load_metadata = staticmethod(load_metadata)
    last_executed_algorithm: str
    save_locations_keys = [
        "open_directory",
        "save_directory",
        "export_directory",
        "batch_plan_directory",
        "multiple_open_directory",
    ]

    def __init__(self, json_path):
        super().__init__(json_path)
        self._mask = None
        self.compare_segmentation = None
        self.segmentation_pipelines_dict = ProfileDict()
        self.segmentation_profiles_dict = ProfileDict()
        self.batch_plans_dict = ProfileDict()
        self.measurement_profiles_dict = ProfileDict()

    def fix_history(self, algorithm_name, algorithm_values):
        """
        set new algorithm parameters to

        :param str algorithm_name:
        :param dict algorithm_values:
        """
        self.history[self.history_index + 1] = self.history[self.history_index + 1].replace_(
            segmentation_parameters={"algorithm_name": algorithm_name, "values": algorithm_values}
        )

    @staticmethod
    def cmp_history_element(el1: HistoryElement, el2: HistoryElement):
        return el1.mask_property == el2.mask_property

    def set_segmentation_to_compare(self, segmentation: ROIInfo):
        self.compare_segmentation = segmentation
        self.compare_segmentation_change.emit(segmentation)

    @property
    def use_physical_unit(self):
        return self.get("use_physical_unit", False)

    def set_use_physical_unit(self, value):
        self.set("use_physical_unit", value)

    def _image_changed(self):
        super()._image_changed()
        self._mask = None

    def get_project_info(self) -> ProjectTuple:
        algorithm_name = self.last_executed_algorithm
        if algorithm_name:
            algorithm_val = {
                "algorithm_name": algorithm_name,
                "values": deepcopy(self.get(f"algorithms.{algorithm_name}")),
            }
        else:
            algorithm_val = {}
        return ProjectTuple(
            file_path=self.image.file_path,
            image=self.image.substitute(),
            roi=self.roi,
            roi_info=self.roi_info,
            additional_layers=self.additional_layers,
            mask=self.mask,
            history=self.history[: self.history_index + 1],
            algorithm_parameters=algorithm_val,
        )

    def set_project_info(self, data: typing.Union[ProjectTuple, MaskInfo]):
        if isinstance(data, ProjectTuple):
            if self.image.file_path == data.image.file_path and self.image.shape == data.image.shape:
                if data.roi is not None:
                    try:
                        self.image.fit_array_to_image(data.roi)
                        self.mask = data.mask
                    except ValueError:
                        self.image = data.image.substitute()
                else:
                    self.mask = data.mask
            else:
                self.image = data.image.substitute(mask=data.mask)
            self.roi = data.roi
            self._additional_layers = data.additional_layers
            self.set_history(data.history[:])
            if data.algorithm_parameters:
                self.last_executed_algorithm = data.algorithm_parameters["algorithm_name"]
                self.set(f"algorithms.{self.last_executed_algorithm}", deepcopy(data.algorithm_parameters["values"]))
                self.algorithm_changed.emit()
        elif isinstance(data, MaskInfo):
            self.mask = data.mask_array

    def get_save_list(self) -> typing.List[SaveSettingsDescription]:
        return super().get_save_list() + [
            SaveSettingsDescription("segmentation_pipeline_save.json", self.segmentation_pipelines_dict),
            SaveSettingsDescription("segmentation_profiles_save.json", self.segmentation_profiles_dict),
            SaveSettingsDescription("statistic_profiles_save.json", self.measurement_profiles_dict),
            SaveSettingsDescription("batch_plans_save.json", self.batch_plans_dict),
        ]

    @property
    def segmentation_pipelines(self) -> typing.Dict[str, SegmentationPipeline]:
        return self.segmentation_pipelines_dict.get(self._current_roi_dict, {})

    @property
    def segmentation_profiles(self) -> typing.Dict[str, ROIExtractionProfile]:
        return self.segmentation_profiles_dict.get(self._current_roi_dict, {})

    @property
    def batch_plans(self) -> typing.Dict[str, CalculationPlan]:
        return self.batch_plans_dict.get(self._current_roi_dict, {})

    @property
    def measurement_profiles(self) -> typing.Dict[str, MeasurementProfile]:
        return self.measurement_profiles_dict.get(self._current_roi_dict, {})
