import typing
import warnings
from copy import deepcopy

from qtpy.QtCore import Signal

from PartSeg.common_backend.base_settings import BaseSettings, SaveSettingsDescription
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.analysis_utils import SegmentationPipeline
from PartSegCore.analysis.calculation_plan import CalculationPlan
from PartSegCore.analysis.io_utils import MaskInfo, ProjectTuple
from PartSegCore.analysis.load_functions import load_metadata
from PartSegCore.analysis.measurement_calculation import MeasurementProfile
from PartSegCore.io_utils import PointsInfo
from PartSegCore.json_hooks import PartSegEncoder
from PartSegCore.project_info import HistoryElement
from PartSegCore.roi_info import ROIInfo
from PartSegCore.utils import EventedDict, ProfileDict


class PartSettings(BaseSettings):
    """
    last_executed_algorithm - parameter for caring last used algorithm
    """

    compare_segmentation_change = Signal(ROIInfo)
    roi_profiles_changed = Signal()
    roi_pipelines_changed = Signal()
    measurement_profiles_changed = Signal()
    batch_plans_changed = Signal()
    json_encoder_class = PartSegEncoder
    load_metadata = staticmethod(load_metadata)
    last_executed_algorithm: str
    save_locations_keys: typing.ClassVar[typing.List[str]] = [
        "open_directory",
        "save_directory",
        "export_directory",
        "batch_plan_directory",
        "multiple_open_directory",
    ]

    def __init__(self, json_path, profile_name="default"):
        super().__init__(json_path, profile_name)
        self._mask = None
        self.compare_segmentation = None
        self._segmentation_pipelines_dict = ProfileDict(klass={"*": {"*": SegmentationPipeline}})
        self._segmentation_profiles_dict = ProfileDict(klass={"*": {"*": ROIExtractionProfile}})
        self._batch_plans_dict = ProfileDict(klass={"*": {"*": CalculationPlan}})
        self._measurement_profiles_dict = ProfileDict(klass={"*": {"*": MeasurementProfile}})
        self._segmentation_profiles_dict.connect("", self.roi_profiles_changed.emit, maxargs=0)
        self._segmentation_pipelines_dict.connect("", self.roi_pipelines_changed.emit, maxargs=0)
        self._measurement_profiles_dict.connect("", self.measurement_profiles_changed.emit, maxargs=0)
        self._batch_plans_dict.connect("", self.batch_plans_changed.emit, maxargs=0)

    def fix_history(self, algorithm_name, algorithm_values):
        """
        set new algorithm parameters to

        :param str algorithm_name:
        :param dict algorithm_values:
        """
        self.history[self.history_index + 1] = self.history[self.history_index + 1].copy(
            update={"roi_extraction_parameters": {"algorithm_name": algorithm_name, "values": algorithm_values}}
        )

    @staticmethod
    def cmp_history_element(el1: HistoryElement, el2: HistoryElement):
        return el1.mask_property == el2.mask_property

    def set_segmentation_to_compare(self, segmentation: ROIInfo):
        self.compare_segmentation = segmentation
        self.compare_segmentation_change.emit(segmentation)

    def _image_changed(self):
        super()._image_changed()
        self._mask = None

    def get_project_info(self) -> ProjectTuple:
        if algorithm_name := self.last_executed_algorithm:
            value = self.get_algorithm(f"algorithms.{algorithm_name}")
            if isinstance(value, EventedDict):
                value = value.as_dict_deep()
            algorithm_val = {
                "algorithm_name": algorithm_name,
                "values": deepcopy(value),
            }
        else:
            algorithm_val = {}
        return ProjectTuple(
            file_path=self.image.file_path,
            image=self.image.substitute(),
            roi_info=self.roi_info,
            additional_layers=self.additional_layers,
            mask=self.mask,
            history=self.history[: self.history_index + 1],
            algorithm_parameters=algorithm_val,
            points=self.points,
        )

    def set_project_info(self, data: typing.Union[ProjectTuple, MaskInfo, PointsInfo]):
        if isinstance(data, MaskInfo):
            self.mask = data.mask_array
            return
        if isinstance(data, PointsInfo):
            self.points = data.points
            return
        if not isinstance(data, ProjectTuple):
            return
        if (
            self.image.file_path == data.image.file_path
            and self.image.shape == data.image.shape
            and self.image.channels == data.image.channels
        ):
            if data.roi_info.roi is not None:
                try:
                    self.image.fit_array_to_image(data.roi_info.roi)
                    self.mask = data.mask
                except ValueError:
                    self.image = data.image.substitute()
            else:
                self.mask = data.mask
        else:
            self.image = data.image.substitute(mask=data.mask)
        self.roi = data.roi_info
        self._additional_layers = data.additional_layers
        self.additional_layers_changed.emit()
        self.set_history(data.history[:])
        if data.algorithm_parameters:
            self.last_executed_algorithm = data.algorithm_parameters["algorithm_name"]
            self.set_algorithm(
                f"algorithms.{self.last_executed_algorithm}", deepcopy(data.algorithm_parameters["values"])
            )
            self.algorithm_changed.emit()

    def get_save_list(self) -> typing.List[SaveSettingsDescription]:
        return [
            *super().get_save_list(),
            SaveSettingsDescription("segmentation_pipeline_save.json", self._segmentation_pipelines_dict),
            SaveSettingsDescription("segmentation_profiles_save.json", self._segmentation_profiles_dict),
            SaveSettingsDescription("statistic_profiles_save.json", self._measurement_profiles_dict),
            SaveSettingsDescription("batch_plans_save.json", self._batch_plans_dict),
        ]

    @property
    def segmentation_pipelines(self) -> typing.Dict[str, SegmentationPipeline]:
        warnings.warn("segmentation_pipelines is deprecated, use roi_pipelines", DeprecationWarning, stacklevel=2)
        return self.roi_pipelines

    @property
    def roi_pipelines(self) -> typing.Dict[str, SegmentationPipeline]:
        return self._segmentation_pipelines_dict.get(self._current_roi_dict, EventedDict())

    @property
    def segmentation_profiles(self) -> typing.Dict[str, ROIExtractionProfile]:
        warnings.warn("segmentation_profiles is deprecated, use roi_profiles", DeprecationWarning, stacklevel=2)
        return self.roi_profiles

    @property
    def roi_profiles(self) -> typing.Dict[str, ROIExtractionProfile]:
        return self._segmentation_profiles_dict.get(self._current_roi_dict, EventedDict())

    @property
    def batch_plans(self) -> typing.Dict[str, CalculationPlan]:
        return self._batch_plans_dict.get(self._current_roi_dict, EventedDict())

    @property
    def measurement_profiles(self) -> typing.Dict[str, MeasurementProfile]:
        return self._measurement_profiles_dict.get(self._current_roi_dict, EventedDict())
