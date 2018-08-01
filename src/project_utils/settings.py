import json
import logging
import typing

from qt_import import QObject, pyqtSignal
from .image_operations import normalize_shape
from matplotlib import pyplot
import copy
import numpy as np
from .custom_colormaps import default_colors
from os import path, makedirs


class ImageSettings(QObject):
    """
    :type _image: np.ndarray
    """
    image_changed = pyqtSignal([np.ndarray], [int], [str])
    segmentation_changed = pyqtSignal(np.ndarray)

    def __init__(self):
        super(ImageSettings, self).__init__()
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

        num = segmentation.max()
        self.chosen_components_widget.set_chose(range(1, num + 1), metadata["components"])
        self.segmentation = segmentation

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, tuple):
            file_path = value[1]
            value = value[0]
            # if len(value) > 3:
            #    self.set_segmentation(value[2], value[3])
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
                for key2 in key_path[i:-1]:
                    curr_dict[key2] = dict()
                    curr_dict = curr_dict[key2]
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
        self.current_profile_dict = "default"
        self.profile_dict: typing.Dict[str, ProfileDict] = {self.current_profile_dict: ProfileDict()}

    def _image_changed(self):
        for i in range(len(self.color_map), self.channels):
            self.color_map.append(default_colors[i % len(default_colors)])

    def change_profile(self, name):
        self.current_profile_dict = name
        if self.current_profile_dict not in self.profile_dict:
            self.profile_dict = {self.current_profile_dict: ProfileDict()}

    def set_in_profile(self, key_path, value):
        self.profile_dict[self.current_profile_dict].set(key_path, value)

    def get_from_profile(self, key_path, default):
        return self.profile_dict[self.current_profile_dict].get(key_path, default)

    def dump_view_profiles(self):
        return json.dumps(self.profile_dict, cls=ProfileEncoder)

    def load_view_profiles(self, data):
        dicts: dict = json.loads(data)
        for k, v in dicts.items():
            self.profile_dict[k] = ProfileDict()
            self.profile_dict[k].my_dict = v


class BaseSettings(ViewSettings):
    def __init__(self):
        super().__init__()
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
                logging.error('error in load "view_profiles"')
            try:
                for k, v in json.loads(data["segment_profile"]).items():
                    self.segmentation_dict[k] = ProfileDict()
                    self.segmentation_dict[k].my_dict = v
            except KeyError:
                logging.error('error in load "segment_profile"')
            try:
                self.image_spacing = data["image_spacing"]
            except KeyError:
                logging.error('error in load "image_spacing"')
        except json.decoder.JSONDecodeError:
            pass