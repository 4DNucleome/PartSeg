import typing
from collections import defaultdict
from copy import copy, deepcopy
from typing import List
import numpy as np
from os import path

from qtpy.QtWidgets import QMessageBox, QWidget
from qtpy.QtCore import Signal, Slot

from PartSeg.tiff_image import Image
from PartSeg.utils.algorithm_describe_base import SegmentationProfile
from PartSeg.utils.segmentation.algorithm_base import SegmentationResult
from ..project_utils_qt.settings import BaseSettings
from PartSeg.utils.mask.io_functions import load_stack_segmentation, save_components, \
    SegmentationTuple


class StackSettings(BaseSettings):
    components_change_list = Signal([int, list])
    save_locations_keys = ["save_batch", "save_components_directory", "save_segmentation_directory",
                           "open_segmentation_directory", "load_image_directory", "batch_directory",
                           "multiple_open_directory"]

    def __init__(self, json_path):
        super().__init__(json_path)
        self.chosen_components_widget = None
        self.keep_chosen_components = False
        self.components_parameters_dict: typing.Dict[int, SegmentationProfile] = {}

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

    def set_segmentation_old(self, segmentation, components):
        num = segmentation.max()
        self.chosen_components_widget.set_chose(range(1, num + 1), components)
        self.image.fit_array_to_image(segmentation)
        self.segmentation = segmentation

    def save_components(self, dir_path, range_changed=None, step_changed=None):
        save_components(self.image, self.chosen_components_widget.get_chosen(), self.segmentation, dir_path,
                        range_changed=range_changed, step_changed=step_changed)

    def load_segmentation(self, file_path: str, range_changed=None, step_changed=None):
        self.segmentation, metadata = load_stack_segmentation(file_path,
                                                              range_changed=range_changed, step_changed=step_changed)
        num = self.segmentation.max()
        self.components_change_list.emit(num, list(metadata["components"]))
        # self.chosen_components_widget.set_chose(range(1, num + 1), metadata["components"])

    def chosen_components(self) -> List[int]:
        if self.chosen_components_widget is not None:
            return sorted(self.chosen_components_widget.get_chosen())
        else:
            raise RuntimeError("chosen_components_widget do not initialized")

    def component_is_chosen(self, val: int) -> bool:
        if self.chosen_components_widget is not None:
            return self.chosen_components_widget.get_state(val)
        else:
            raise RuntimeError("chosen_components_widget do not idealized")

    def components_mask(self) -> np.ndarray:
        if self.chosen_components_widget is not None:
            return self.chosen_components_widget.get_mask()
        else:
            raise RuntimeError("chosen_components_widget do not initialized")

    def get_project_info(self) -> SegmentationTuple:
        return SegmentationTuple(self.image.file_path, self.image.substitute(), self.segmentation,
                                 self.chosen_components(), copy(self.components_parameters_dict))

    def set_project_info(self, data: SegmentationTuple):
        """signals = self.signalsBlocked()
        if data.segmentation is not None:
            self.blockSignals(True)"""
        if data.image is not None and \
                (self.image.file_path != data.image.file_path or self.image.shape != data.image.shape):
            self.image = data.image
        # self.blockSignals(signals)
        state = self.get_project_info()
        # TODO Remove reperition this and set_segmentation code
        if self.keep_chosen_components:
            state2 = self.transform_state(state, data.segmentation, data.segmentation_parameters, data.chosen_components,
                                          self.keep_chosen_components)
            self.chosen_components_widget.set_chose(list(sorted(state2.segmentation_parameters.keys())),
                                                    state2.chosen_components)
            self.segmentation = state2.segmentation
            self.components_parameters_dict = state2.segmentation_parameters
        else:
            self.chosen_components_widget.set_chose(list(sorted(data.segmentation_parameters.keys())),
                                                    data.chosen_components)
            self.segmentation = data.segmentation
            self.components_parameters_dict = data.segmentation_parameters

    @staticmethod
    def transform_state(state: SegmentationTuple, new_segmentation_data: np.ndarray, segmentation_parameters: typing.Dict,
                        list_of_components: List[int],
                        save_chosen: bool=True) -> SegmentationTuple:

        if list_of_components is None:
            list_of_components = []
        if segmentation_parameters is None:
            segmentation_parameters = defaultdict(lambda: None)
        if save_chosen and state.segmentation is not None:
            segmentation = np.zeros(state.segmentation.shape, dtype=state.segmentation.dtype)
            components_parameters_dict = {}
            for i, val in enumerate(sorted(state.chosen_components), 1):
                segmentation[state.segmentation == val] = i
                components_parameters_dict[i] = state.segmentation_parameters[val]
            base_chose = list(range(1, len(state.chosen_components) + 1))
        else:
            segmentation = None
            base_chose = []
            components_parameters_dict = {}
        if new_segmentation_data is not None:
            state.image.fit_array_to_image(new_segmentation_data)
            num = new_segmentation_data.max()
            if segmentation is not None:
                new_segmentation = np.copy(new_segmentation_data)
                new_segmentation[segmentation > 0] = 0
                components_size = np.bincount(new_segmentation.flat)
                base_index = len(base_chose) + 1
                chosen_components = base_chose[:]
                components_list = base_chose[:]
                for i, val in enumerate(components_size[1:], 1):
                    if val > 0:
                        segmentation[new_segmentation == i] = base_index
                        if i in list_of_components:
                            chosen_components.append(base_index)
                        components_list.append(base_index)
                        components_parameters_dict[base_index] = segmentation_parameters[i]
                        base_index += 1
                return state._replace(segmentation=segmentation, chosen_components=chosen_components,
                                      segmentation_parameters=components_parameters_dict)
            else:
                for i in range(1, num + 1):
                    components_parameters_dict[i] = segmentation_parameters[i]
                return state._replace(segmentation=new_segmentation_data, chosen_components=list_of_components,
                                      segmentation_parameters=components_parameters_dict)
        else:
            return state._replace(segmentation=segmentation, chosen_components=base_chose,
                                  segmentation_parameters=components_parameters_dict)

    def set_segmentation(self, new_segmentation_data, save_chosen=True, list_of_components=None,
                         segmentation_parameters=None):
        if list_of_components is None:
            list_of_components = []
        if segmentation_parameters is None:
            segmentation_parameters = defaultdict(lambda: None)
        state = self.get_project_info()
        if save_chosen:
            state2 = self.transform_state(state, new_segmentation_data, segmentation_parameters,
                                          list_of_components,
                                          save_chosen)
            self.chosen_components_widget.set_chose(list(sorted(state2.segmentation_parameters.keys())),
                                                    state2.chosen_components)
            self.segmentation = state2.segmentation
            self.components_parameters_dict = state2.segmentation_parameters
        else:
            self.chosen_components_widget.set_chose(list(sorted(segmentation_parameters.keys())),
                                                    list_of_components)
            self.segmentation = new_segmentation_data
            self.components_parameters_dict = segmentation_parameters

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
                                           "Time data are currently not supported. "
                                           "Maybe You would like to treat time as z-stack",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if res == QMessageBox.Yes:
                    return image.swap_time_and_stack()
                return False
        return True


def get_mask(segmentation: typing.Optional[np.ndarray], chosen: List[int]):
    if segmentation is None or len(chosen) == 0:
        return None
    resp = np.ones(segmentation.shape, dtype=np.uint8)
    for i in chosen:
        resp[segmentation == i] = 0
    return resp
