from typing import List
import numpy as np
from os import path
from PyQt5.QtCore import pyqtSignal

from project_utils_qt.settings import BaseSettings
from partseg_utils.segmentation.segment import cut_with_mask, save_catted_list
from deprecation import deprecated

from segmentation_mask.io_functions import save_stack_segmentation, load_stack_segmentation, save_components

class StackSettings(BaseSettings):
    components_change_list = pyqtSignal([int, list])

    def __init__(self, json_path):
        super().__init__(json_path)
        self.chosen_components_widget = None

    """@property
    def batch_directory(self):
        # TODO update batch widget to use new style settings
        return self.get("io.batch_directory", self.get("io.load_image_directory", ""))

    @batch_directory.setter
    def batch_directory(self, val):
        self.set("io.batch_directory", val)"""

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

    def set_segmentation(self, segmentation, metadata):
        num = segmentation.max()
        self.chosen_components_widget.set_chose(range(1, num + 1), metadata["components"])
        self.segmentation = segmentation

    @deprecated()
    def save_result(self, dir_path: str):
        # TODO remove
        res_img = cut_with_mask(self.segmentation, self._image, only=self.chosen_components())
        res_mask = cut_with_mask(self.segmentation, self.segmentation, only=self.chosen_components())
        res_mask = [(int(n), np.array((v > 0).astype(np.uint8))) for n,v in res_mask]
        file_name = self.file_save_name()
        save_catted_list(res_img, dir_path, prefix=f"{file_name}_component")
        save_catted_list(res_mask, dir_path, prefix=f"{file_name}_component", suffix="_mask")

    def save_components(self, dir_path, range_changed=None, step_changed=None):
        save_components(self.image, self.chosen_components_widget.get_chosen(), self.segmentation, dir_path,
                        range_changed=range_changed, step_changed=step_changed)

    def save_segmentation(self, file_path: str, range_changed=None, step_changed=None):
        print("[save_segmentation]", self.chosen_components())
        save_stack_segmentation(file_path, self.segmentation, self.chosen_components(), self.image.file_path,
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
