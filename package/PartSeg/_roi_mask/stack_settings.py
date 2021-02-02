import dataclasses
import typing
from collections import defaultdict
from copy import copy, deepcopy
from os import path

import numpy as np
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QMessageBox

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.io_utils import HistoryElement, HistoryProblem
from PartSegCore.mask.io_functions import MaskProjectTuple, load_metadata
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
        if self._parent and np.max(result.roi) == 0:
            QMessageBox.information(
                self._parent,
                "No result",
                "Segmentation contains no component, check parameters, especially chosen channel.",
            )
        if result.info_text and self._parent is not None:
            QMessageBox().information(self._parent, "Algorithm info", result.info_text)
        parameters_dict = defaultdict(lambda: deepcopy(result.parameters))
        self._additional_layers = result.additional_layers
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

        raise RuntimeError("chosen_components_widget do not initialized")

    def component_is_chosen(self, val: int) -> bool:
        """
        Needs instance of :py:class:`PartSeg.segmentation_mask.main_window.ChosenComponents` on variable
        Py:attr:`chosen_components_widget` (or something implementing its interface)

        :return: if given component is selected
        """
        if self.chosen_components_widget is not None:
            return self.chosen_components_widget.get_state(val)

        raise RuntimeError("chosen_components_widget do not idealized")

    def components_mask(self) -> np.ndarray:
        """
        Needs instance of :py:class:`PartSeg.segmentation_mask.main_window.ChosenComponents` on variable
        Py:attr:`chosen_components_widget` (or something implementing its interface)

        :return: boolean mask if component is selected
        """
        if self.chosen_components_widget is not None:
            return self.chosen_components_widget.get_mask()

        raise RuntimeError("chosen_components_widget do not initialized")

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
            roi=self.roi,
            roi_info=self.roi_info,
            selected_components=self.chosen_components(),
            roi_extraction_parameters=copy(self.components_parameters_dict),
            history=self.history[: self.history_index + 1],
        )

    def set_project_info(self, data: MaskProjectTuple):
        """signals = self.signalsBlocked()
        if data.segmentation is not None:
            self.blockSignals(True)"""
        if isinstance(data.image, Image) and (
            self.image_path != data.image.file_path or self.image_shape != data.image.shape
        ):
            # This line clean segmentation also
            self.image = data.image
        # self.blockSignals(signals)
        state = self.get_project_info()
        # TODO Remove repetition this and set_segmentation code

        components = np.unique(data.roi)
        if components[0] == 0 or components[0] is None:
            components = components[1:]
        for i in components:
            _skip = data.roi_extraction_parameters[int(i)]  # noqa: F841
        self.mask = data.mask
        if self.keep_chosen_components:
            if not self.compare_history(data.history) and self.chosen_components():
                raise HistoryProblem("Incompatible history")
            state2 = self.transform_state(
                state,
                data.roi,
                data.roi_extraction_parameters,
                data.selected_components,
                self.keep_chosen_components,
            )
            self.chosen_components_widget.set_chose(
                list(sorted(state2.roi_extraction_parameters.keys())), state2.selected_components
            )
            self.roi = state2.roi
            self.components_parameters_dict = state2.roi_extraction_parameters
        else:
            self.set_history(data.history)
            self.chosen_components_widget.set_chose(
                list(sorted(data.roi_extraction_parameters.keys())), data.selected_components
            )
            self.roi = data.roi
            self.components_parameters_dict = data.roi_extraction_parameters

    @staticmethod
    def transform_state(
        state: MaskProjectTuple,
        new_segmentation_data: np.ndarray,
        segmentation_parameters: typing.Dict,
        list_of_components: typing.List[int],
        save_chosen: bool = True,
    ) -> MaskProjectTuple:

        # TODO Refactor
        if list_of_components is None:
            list_of_components = []
        if segmentation_parameters is None:
            segmentation_parameters = defaultdict(lambda: None)
        segmentation_count = 0 if state.roi is None else len(np.unique(state.roi.flat))
        new_segmentation_count = 0 if new_segmentation_data is None else len(np.unique(new_segmentation_data.flat))
        segmentation_dtype = minimal_dtype(segmentation_count + new_segmentation_count)
        if save_chosen and state.roi is not None:
            segmentation = reduce_array(state.roi, state.selected_components, dtype=segmentation_dtype)
            components_parameters_dict = {}
            for i, val in enumerate(sorted(state.selected_components), 1):
                components_parameters_dict[i] = state.roi_extraction_parameters[val]
            base_chose = list(range(1, len(state.selected_components) + 1))
        else:
            segmentation = None
            base_chose = []
            components_parameters_dict = {}
        if new_segmentation_data is not None:
            state.image.fit_array_to_image(new_segmentation_data)
            num = np.max(new_segmentation_data)
            if segmentation is not None:
                new_segmentation = np.copy(new_segmentation_data)
                new_segmentation[segmentation > 0] = 0
                new_segmentation = reduce_array(new_segmentation, dtype=segmentation_dtype)
                segmentation[new_segmentation > 0] = new_segmentation[new_segmentation > 0] + len(base_chose)

                components_size = np.bincount(new_segmentation.flat)

                base_index = len(base_chose) + 1
                chosen_components = base_chose[:]
                components_list = base_chose[:]
                for i, val in enumerate(components_size[1:], 1):
                    if val > 0:
                        if i in list_of_components:
                            chosen_components.append(base_index)
                        components_list.append(base_index)
                        components_parameters_dict[base_index] = segmentation_parameters[i]
                        base_index += 1

                return dataclasses.replace(
                    state,
                    roi=segmentation,
                    selected_components=chosen_components,
                    roi_extraction_parameters=components_parameters_dict,
                )

            for i in range(1, num + 1):
                components_parameters_dict[i] = segmentation_parameters[i]

            return dataclasses.replace(
                state,
                roi=new_segmentation_data,
                selected_components=list_of_components,
                roi_extraction_parameters=components_parameters_dict,
            )

        return dataclasses.replace(
            state,
            roi=segmentation,
            selected_components=base_chose,
            roi_extraction_parameters=components_parameters_dict,
        )

    def compare_history(self, history: typing.List[HistoryElement]):
        # TODO check dict comparision
        if len(history) != self.history_size():
            return False
        for el1, el2 in zip(self.history, history):
            if el2.mask_property != el1.mask_property or el2.segmentation_parameters != el1.segmentation_parameters:
                return False
        return True

    def set_project_data(self, data: MaskProjectTuple, save_chosen=True):
        if isinstance(data.image, Image):
            self.image = data.image
        if data.roi is not None:
            if not self.compare_history(data.history) and data.selected_components:
                raise HistoryProblem("Incompatible history")
            self.set_history(data.history)
            self.mask = data.mask
            self.set_segmentation(data.roi, save_chosen, data.selected_components, data.roi_extraction_parameters)

    def set_segmentation(
        self, new_segmentation_data, save_chosen=True, list_of_components=None, segmentation_parameters=None
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
                state, new_segmentation_data, segmentation_parameters, list_of_components, save_chosen
            )
            self.chosen_components_widget.set_chose(
                list(sorted(state2.roi_extraction_parameters.keys())), state2.selected_components
            )
            self.roi = state2.roi
            self.components_parameters_dict = state2.roi_extraction_parameters
        else:
            unique = np.unique(new_segmentation_data.flat)
            if unique[0] == 0:
                unique = unique[1:]
            selected_parameters = {i: segmentation_parameters[i] for i in unique}

            self.chosen_components_widget.set_chose(list(sorted(selected_parameters.keys())), list_of_components)
            self.roi = new_segmentation_data
            self.components_parameters_dict = segmentation_parameters


def get_mask(segmentation: typing.Optional[np.ndarray], mask: typing.Optional[np.ndarray], selected: typing.List[int]):
    """
    Calculate mask base on segmentation, current mask and list of chosen components.

    :param typing.Optional[np.ndarray] segmentation: segmentation array
    :param typing.Optional[np.ndarray] mask: current mask
    :param typing.List[int] selected: list of selected components
    :return: new mask
    :rtype: typing.Optional[np.ndarray]
    """
    if segmentation is None or len(selected) == 0:
        return None if mask is None else mask
    segmentation = reduce_array(segmentation, selected)
    if mask is None:
        resp = np.ones(segmentation.shape, dtype=np.uint8)
    else:
        resp = np.copy(mask)
    resp[segmentation > 0] = 0
    return resp
