import dataclasses
import typing
from collections import defaultdict
from copy import copy, deepcopy
from os import path

import numpy as np
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QMessageBox

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.io_utils import PointsInfo
from PartSegCore.mask.io_functions import MaskProjectTuple, load_metadata
from PartSegCore.project_info import HistoryElement, HistoryProblem
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.algorithm_base import SegmentationResult
from PartSegImage import Image
from PartSegImage.image import minimal_dtype, reduce_array

from ..common_backend.base_settings import BaseSettings


class StackSettings(BaseSettings):
    load_metadata = staticmethod(load_metadata)
    components_change_list = Signal([int, list])
    save_locations_keys = [
        "save_batch",
        "save_components_directory",
        "save_segmentation_directory",
        "open_segmentation_directory",
        "load_image_directory",
        "batch_directory",
        "multiple_open_directory",
    ]

    def set_segmentation_result(self, result: SegmentationResult):
        if self._parent and np.max(result.roi) == 0:  # pragma: no cover
            QMessageBox.information(
                self._parent,
                "No result",
                "Segmentation contains no component, check parameters, especially chosen channel.",
            )
        if result.info_text and self._parent is not None:  # pragma: no cover
            QMessageBox().information(self._parent, "Algorithm info", result.info_text)
        parameters_dict = defaultdict(lambda: deepcopy(result.parameters))
        self._additional_layers = result.additional_layers
        self.additional_layers_changed.emit()
        self.last_executed_algorithm = result.parameters.algorithm
        self.set(f"algorithms.{result.parameters.algorithm}", result.parameters.values)
        self.set_segmentation(result.roi, True, [], parameters_dict)

    def __init__(self, json_path):
        super().__init__(json_path)
        self.chosen_components_widget = None
        self.keep_chosen_components = False
        self.components_parameters_dict: typing.Dict[int, ROIExtractionProfile] = {}

    @Slot(int)
    def set_keep_chosen_components(self, val: bool):
        self.keep_chosen_components = val

    def file_save_name(self):
        return path.splitext(path.basename(self.image.file_path))[0]

    def get_file_names_for_save_result(self, dir_path):
        components = self.chosen_components()
        file_name = self.file_save_name()
        res = []
        for i in components:
            res.append(path.join(dir_path, f"{file_name}_component{i}.tif"))
            res.append(path.join(dir_path, f"{file_name}_component{i}_mask.tif"))
        return res

    def chosen_components(self) -> typing.List[int]:
        """
        Needs instance of :py:class:`PartSeg.segmentation_mask.main_window.ChosenComponents` on variable
        Py:attr:`chosen_components_widget` (or something implementing its interface)

        :return: list of chosen components
        """
        if self.chosen_components_widget is not None:
            return sorted(self.chosen_components_widget.get_chosen())

        raise RuntimeError("chosen_components_widget do not initialized")  # pragma: no cover

    def component_is_chosen(self, val: int) -> bool:
        """
        Needs instance of :py:class:`PartSeg.segmentation_mask.main_window.ChosenComponents` on variable
        Py:attr:`chosen_components_widget` (or something implementing its interface)

        :return: if given component is selected
        """
        if self.chosen_components_widget is not None:
            return self.chosen_components_widget.get_state(val)

        raise RuntimeError("chosen_components_widget do not idealized")  # pragma: no cover

    def components_mask(self) -> np.ndarray:
        """
        Needs instance of :py:class:`PartSeg.segmentation_mask.main_window.ChosenComponents` on variable
        Py:attr:`chosen_components_widget` (or something implementing its interface)

        :return: boolean mask if component is selected
        """
        if self.chosen_components_widget is not None:
            return self.chosen_components_widget.get_mask()

        raise RuntimeError("chosen_components_widget do not initialized")  # pragma: no cover

    def get_project_info(self) -> MaskProjectTuple:
        """
        Get all information about current project

        :return: object with all project details.
        :rtype: MaskProjectTuple
        """
        return MaskProjectTuple(
            file_path=self.image.file_path,
            image=self.image.substitute(),
            mask=self.mask,
            roi_info=self.roi_info,
            selected_components=self.chosen_components(),
            roi_extraction_parameters=copy(self.components_parameters_dict),
            history=self.history[: self.history_index + 1],
            points=self.points,
        )

    def set_project_info(self, data: typing.Union[MaskProjectTuple, PointsInfo]):
        """signals = self.signalsBlocked()
        if data.segmentation is not None:
            self.blockSignals(True)"""

        if isinstance(data, PointsInfo):
            self.points = data.points
            return
        if isinstance(data.image, Image) and (
            self.image_path != data.image.file_path or self.image_shape != data.image.shape
        ):
            # This line clean segmentation also
            self.image = data.image
        # self.blockSignals(signals)
        state = self.get_project_info()
        # TODO Remove repetition this and set_segmentation code

        components = list(sorted(data.roi_info.bound_info))
        for i in components:
            _skip = data.roi_extraction_parameters[int(i)]  # noqa: F841
        self.mask = data.mask
        if self.keep_chosen_components:
            if not self.compare_history(data.history) and self.chosen_components():
                raise HistoryProblem("Incompatible history")
            state2 = self.transform_state(
                state,
                data.roi_info,
                data.roi_extraction_parameters,
                data.selected_components,
                self.keep_chosen_components,
            )
            self.chosen_components_widget.set_chose(
                list(sorted(state2.roi_extraction_parameters.keys())), state2.selected_components
            )
            self.roi = state2.roi_info
            self.components_parameters_dict = state2.roi_extraction_parameters
        else:
            self.set_history(data.history)
            self.chosen_components_widget.set_chose(
                list(sorted(data.roi_extraction_parameters.keys())), data.selected_components
            )
            self.roi = data.roi_info
            self.components_parameters_dict = data.roi_extraction_parameters

    @staticmethod
    def _clip_data_array(mask_array: np.ndarray, data_array: np.ndarray) -> np.ndarray:
        """
        From `data_array` fet only information masked with `mask_array`

        :param np.ndarray mask_array: masking array
        :param np.ndarray data_array: data array
        :return: array with selected data
        :rtype: np.ndarray
        """
        res = np.copy(data_array)
        res[mask_array == 0] = 0
        return res

    @classmethod
    def transform_state(
        cls,
        state: MaskProjectTuple,
        new_roi_info: ROIInfo,
        new_roi_extraction_parameters: typing.Dict[int, typing.Optional[ROIExtractionProfile]],
        list_of_components: typing.List[int],
        save_chosen: bool = True,
    ) -> MaskProjectTuple:
        """

        :param MaskProjectTuple state: state to be transformed
        :param ROIInfo new_roi_info: roi description
        :param typing.Dict[int, typing.Optional[ROIExtractionProfile]] new_roi_extraction_parameters:
            Parameters used to extract roi
        :param typing.List[int] list_of_components: list of components from new_roi which should be selected
        :param bool save_chosen: if save currently selected components
        :return: new state
        """

        # TODO Refactor
        if not save_chosen or state.roi_info.roi is None or len(state.selected_components) == 0:
            return dataclasses.replace(
                state,
                roi_info=new_roi_info,
                selected_components=list_of_components,
                roi_extraction_parameters=new_roi_extraction_parameters,
            )
        if list_of_components is None:
            list_of_components = []
        if new_roi_extraction_parameters is None:
            new_roi_extraction_parameters = defaultdict(lambda: None)
        segmentation_count = len(state.roi_info.bound_info)
        new_segmentation_count = len(new_roi_info.bound_info)
        segmentation_dtype = minimal_dtype(segmentation_count + new_segmentation_count)
        roi_base = reduce_array(state.roi_info.roi, state.selected_components, dtype=segmentation_dtype)
        annotation_base = {
            i: state.roi_info.annotations.get(x) for i, x in enumerate(state.selected_components, start=1)
        }
        alternative_base = {
            name: cls._clip_data_array(roi_base, array) for name, array in state.roi_info.alternative.items()
        }
        components_parameters_dict = {
            i: state.roi_extraction_parameters[val] for i, val in enumerate(sorted(state.selected_components), 1)
        }

        base_chose = list(annotation_base.keys())

        if new_segmentation_count == 0:
            return dataclasses.replace(
                state,
                roi_info=ROIInfo(roi=roi_base, annotations=annotation_base, alternative=alternative_base),
                selected_components=base_chose,
                roi_extraction_parameters=components_parameters_dict,
            )

        new_segmentation = np.copy(new_roi_info.roi)
        new_segmentation[roi_base > 0] = 0
        left_component_list = np.unique(new_segmentation.flat)
        if left_component_list[0] == 0:
            left_component_list = left_component_list[1:]
        new_segmentation = reduce_array(new_segmentation, dtype=segmentation_dtype)
        roi_base[new_segmentation > 0] = new_segmentation[new_segmentation > 0] + len(base_chose)
        for name, array in new_roi_info.alternative.items():
            if name in alternative_base:
                alternative_base[name][new_segmentation > 0] = array[new_segmentation > 0]
            else:
                alternative_base[name] = cls._clip_data_array(new_segmentation, array)
        for i, el in enumerate(left_component_list, start=len(base_chose) + 1):
            annotation_base[i] = new_roi_info.annotations.get(el)
            if el in list_of_components:
                base_chose.append(i)
            components_parameters_dict[i] = new_roi_extraction_parameters[el]

        roi_info = ROIInfo(roi=roi_base, annotations=annotation_base, alternative=alternative_base)

        return dataclasses.replace(
            state,
            roi_info=roi_info,
            selected_components=base_chose,
            roi_extraction_parameters=components_parameters_dict,
        )

    def compare_history(self, history: typing.List[HistoryElement]):
        # TODO check dict comparision
        if len(history) != self.history_size():
            return False
        return not any(
            el2.mask_property != el1.mask_property or el2.roi_extraction_parameters != el1.roi_extraction_parameters
            for el1, el2 in zip(self.history, history)
        )

    def set_segmentation(
        self, new_segmentation_data: np.ndarray, save_chosen=True, list_of_components=None, segmentation_parameters=None
    ):
        new_segmentation_data = self.image.fit_array_to_image(new_segmentation_data)
        if list_of_components is None:
            list_of_components = []
        if segmentation_parameters is None:
            segmentation_parameters = defaultdict(lambda: None)
        state = self.get_project_info()
        try:
            self.image.fit_array_to_image(new_segmentation_data)
        except ValueError:
            raise ValueError("Segmentation do not fit to image")
        if save_chosen:
            state2 = self.transform_state(
                state, ROIInfo(new_segmentation_data), segmentation_parameters, list_of_components, save_chosen
            )
            self.chosen_components_widget.set_chose(
                list(sorted(state2.roi_extraction_parameters.keys())), state2.selected_components
            )
            self.roi = state2.roi_info
            self.components_parameters_dict = state2.roi_extraction_parameters
        else:
            unique = np.unique(new_segmentation_data.flat)
            if unique[0] == 0:
                unique = unique[1:]
            selected_parameters = {i: segmentation_parameters[i] for i in unique}

            self.chosen_components_widget.set_chose(list(sorted(selected_parameters.keys())), list_of_components)
            self.roi = new_segmentation_data
            self.components_parameters_dict = segmentation_parameters


def get_mask(
    segmentation: typing.Optional[np.ndarray], mask: typing.Optional[np.ndarray], selected: typing.List[int]
) -> np.ndarray:
    """
    Calculate mask base on segmentation, current mask and list of chosen components.
    Its exclude selected components from mask.

    :param typing.Optional[np.ndarray] segmentation: segmentation array
    :param typing.Optional[np.ndarray] mask: current mask
    :param typing.List[int] selected: list of selected components which should be masked as non segmentation area
    :return: new mask
    :rtype: typing.Optional[np.ndarray]
    """
    if segmentation is None or len(selected) == 0:
        return mask
    segmentation = reduce_array(segmentation, selected)
    if mask is None:
        resp = np.ones(segmentation.shape, dtype=np.uint8)
    else:
        resp = np.copy(mask)
    resp[segmentation > 0] = 0
    return resp
