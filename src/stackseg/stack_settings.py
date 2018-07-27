import copy
import json
import typing
from typing import List

import numpy as np
from matplotlib import pyplot
from os import path, makedirs
from partseg.io_functions import save_stack_segmentation, load_stack_segmentation
from qt_import import QObject, pyqtSignal
from stackseg.stack_algorithm.segment import cut_with_mask, save_catted_list
from project_utils.image_operations import normalize_shape

default_colors = ['BlackRed', 'BlackGreen', 'BlackBlue', 'BlackMagenta']


class ImageSettings(QObject):
    """
    :type _image: np.ndarray
    """
    image_changed = pyqtSignal([np.ndarray], [int], [str])
    segmentation_changed = pyqtSignal(np.ndarray)

    def __init__(self):
        super(ImageSettings, self).__init__()
        self.open_directory = None
        self.save_directory = None
        self._image = None
        self._image_path = ""
        self.segmentation = None
        self.has_channels = False
        self.image_spacing = 70, 70, 210
        self._segmentation = None
        self.sizes = []
        self.chosen_colormap = pyplot.colormaps()
        # self.fixed_range = 0, 255

    @property
    def segmentation(self) -> np.ndarray:
        return self._segmentation

    @segmentation.setter
    def segmentation(self, val: np.ndarray):
        self._segmentation = val
        if val is not None:
            self.sizes = np.bincount(val.flat)
            self.segmentation_changed.emit(val)
        else:
            self.sizes = []

    def set_segmentation(self, segmentation, metadata):
        self.segmentation = segmentation
        num = self.segmentation.max()
        self.chosen_components_widget.set_chose(range(1, num + 1), metadata["components"])

    @property
    def batch_directory(self):
        return self.open_directory

    @batch_directory.setter
    def batch_directory(self, val):
        self.open_directory = val

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, tuple):
            file_path = value[1]
            value = value[0]
        else:
            file_path = None
        value = np.squeeze(value)
        self._image = normalize_shape(value)

        if file_path is not None:
            self._image_path = file_path
            self.image_changed[str].emit(self._image_path)
        if self._image.shape[-1] < 10:
            self.has_channels = True
        else:
            self.has_channels = False

        self._image_changed()

        self.image_changed.emit(self._image)
        self.image_changed[int].emit(self.channels)

    def _image_changed(self):
        pass

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        self._image_path = value
        self.image_changed[str].emmit(self._image_path)

    @property
    def channels(self):
        if self._image is None:
            return 0
        if len(self._image.shape) == 4:
            return self._image.shape[-1]
        else:
            return 1

    def get_chanel(self, chanel_num):
        if self.has_channels:
            return self._image[..., chanel_num]
        return self._image

    def get_information(self, *pos):
        return self._image[pos]


class ProfileDict(object):
    def __init__(self):
        self.my_dict = dict()

    def set(self, key_path: typing.Union[list, str], value):
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict

        for i, key in enumerate(key_path[:-1]):
            try:
                curr_dict = curr_dict[key]
            except KeyError:
                for key in key_path[i:-1]:
                    curr_dict[key] = dict()
                    curr_dict = curr_dict[key]
                    break
        curr_dict[key_path[-1]] = value


    def get(self, key_path: typing.Union[list, str], default):
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict
        for i, key in enumerate(key_path):
            try:
                curr_dict = curr_dict[key]
            except KeyError:
                val = copy.deepcopy(default)
                self.set(key_path, val)
                return val
        return curr_dict


class ProfileEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ProfileDict):
            return o.my_dict
        return super().default(o)


class ViewSettings(ImageSettings):
    def __init__(self):
        super().__init__()
        self.color_map = []
        self.current_dict = "default"
        self.profile_dict : typing.Dict[str, ProfileDict] = {self.current_dict: ProfileDict()}

    def _image_changed(self):
        for i in range(len(self.color_map), self.channels):
            self.color_map.append(default_colors[i % len(default_colors)])

    def change_profile(self, name):
        self.current_dict = name
        if self.current_dict not in self.profile_dict:
            self.profile_dict = {self.current_dict: ProfileDict()}

    def set_in_profile(self, key_path, value):
        self.profile_dict[self.current_dict].set(key_path, value)

    def get_from_profile(self, key_path, default):
        return self.profile_dict[self.current_dict].get(key_path, default)

    def dump_view_profiles(self):
        return json.dumps(self.profile_dict, cls=ProfileEncoder)

    def load_view_profiles(self, data):
        dicts:dict = json.loads(data)
        for k, v in dicts.items():
            self.profile_dict[k] = ProfileDict()
            self.profile_dict[k].my_dict = v


class StackSettings(ViewSettings):

    def dump(self, file_path):
        if not path.exists(path.dirname(file_path)):
            makedirs(path.dirname(file_path))
        with open(file_path, 'w') as ff:
            json.dump({"view_profiles": self.dump_view_profiles()}, ff)

    def load(self, file_path):
        with open(file_path, 'r') as ff:
            data = json.load(ff)
        try:
            self.load_view_profiles(data["view_profiles"])
        except KeyError:
            pass


    def __init__(self):
        super().__init__()
        self.chosen_components_widget = None

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
