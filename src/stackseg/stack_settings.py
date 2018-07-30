import json
import typing
from typing import List

import numpy as np
from os import path, makedirs
from partseg.io_functions import save_stack_segmentation, load_stack_segmentation
from project_utils.settings import ViewSettings, ProfileDict, ProfileEncoder
from stackseg.stack_algorithm.segment import cut_with_mask, save_catted_list

default_colors = ['BlackRed', 'BlackGreen', 'BlackBlue', 'BlackMagenta']


class StackSettings(ViewSettings):
    def __init__(self):
        super().__init__()
        self.chosen_components_widget = None
        self.current_segmentation_dict = "default"
        self.segmentation_dict: typing.Dict[str, ProfileDict] = {self.current_segmentation_dict: ProfileDict()}

    def set(self, key_path, value):
        self.segmentation_dict[self.current_segmentation_dict].set(key_path, value)

    def get(self, key_path, default):
        return self.segmentation_dict[self.current_segmentation_dict].get(key_path, default)

    def dump(self, file_path):
        if not path.exists(path.dirname(file_path)):
            makedirs(path.dirname(file_path))
        dump_view = self.dump_view_profiles()
        dump_seg = json.dumps(self.segmentation_dict, cls=ProfileEncoder)
        with open(file_path, 'w') as ff:
            json.dump(
                {"view_profiles": dump_view,
                 "segment_profile": dump_seg,
                 "image_spacing": self.image_spacing
                 },
                ff)

    def load(self, file_path):
        try:
            with open(file_path, 'r') as ff:
                data = json.load(ff)
            try:
                self.load_view_profiles(data["view_profiles"])
            except KeyError:
                pass
            try:
                for k, v in json.loads(data["segment_profile"]).items():
                    self.segmentation_dict[k] = ProfileDict()
                    self.segmentation_dict[k].my_dict = v
            except KeyError:
                pass
            try:
                self.image_spacing = data["image_spacing"]
            except KeyError:
                pass
        except json.decoder.JSONDecodeError:
            pass

    @property
    def batch_directory(self):
        # TODO update batch widget to use new style settings
        return self.get("io.batch_directory", self.get("io.load_image_directory", ""))

    @batch_directory.setter
    def batch_directory(self, val):
        self.set("io.batch_directory", val)

    def save_result(self, dir_path: str):
        res_img = cut_with_mask(self.segmentation, self._image, only=self.chosen_components())
        res_mask = cut_with_mask(self.segmentation, self.segmentation, only=self.chosen_components())
        file_name = path.splitext(path.basename(self.image_path))[0]
        save_catted_list(res_img, dir_path, prefix=f"{file_name}_component")
        save_catted_list(res_mask, dir_path, prefix=f"{file_name}_component", suffix="_mask")

    def save_segmentation(self, file_path: str):
        save_stack_segmentation(file_path, self.segmentation, self.chosen_components(), self._image_path)

    def load_segmentation(self, file_path: str):
        self.segmentation, metadata = load_stack_segmentation(file_path)
        num = self.segmentation.max()
        self.chosen_components_widget.set_chose(range(1, num + 1), metadata["components"])

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
