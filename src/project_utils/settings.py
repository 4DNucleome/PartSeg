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
        self.has_channels = False
        self.image_spacing = 70, 70, 210
        self._segmentation = None
        self.sizes = []
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
        self.segmentation = None

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
            except KeyError as e:
                if default is not None:
                    val = copy.deepcopy(default)
                    self.set(key_path, val)
                    return val
                else:
                    raise e
        return curr_dict


class ProfileEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ProfileDict):
            return {"__ProfileDict__": True, **o.my_dict}
        return super().default(o)

def profile_hook(_, dkt):
    if "__ProfileDict__" in dkt:
        del dkt["__ProfileDict__"]
        res = ProfileDict()
        res.my_dict = dkt
        return res
    return dkt



class ViewSettings(ImageSettings):
    colormap_changes = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.color_map = []
        self.border_val = []
        # self.chosen_colormap = pyplot.colormaps()
        self.current_profile_dict = "default"
        self.profile_dict: typing.Dict[str, ProfileDict] = {self.current_profile_dict: ProfileDict()}

    @property
    def chosen_colormap(self):
        return self.get_from_profile("colormaps", pyplot.colormaps())

    @chosen_colormap.setter
    def chosen_colormap(self, val):
        self.set_in_profile("colormaps", val)
        self.colormap_changes.emit()

    @property
    def available_colormaps(_):
        return pyplot.colormaps()

    def change_profile(self, name):
        self.current_profile_dict = name
        if self.current_profile_dict not in self.profile_dict:
            self.profile_dict = {self.current_profile_dict: ProfileDict()}

    def set_in_profile(self, key_path, value):
        self.profile_dict[self.current_profile_dict].set(key_path, value)

    def get_from_profile(self, key_path, default=None):
        return self.profile_dict[self.current_profile_dict].get(key_path, default)

    def dump_view_profiles(self):
        # return json.dumps(self.profile_dict, cls=ProfileEncoder)
        return self.profile_dict

    def load_view_profiles(self, dicts):
        for k, v in dicts.items():
            self.profile_dict[k] = v # ProfileDict()
            # self.profile_dict[k].my_dict = v


class BaseSettings(ViewSettings):
    json_encoder_class = ProfileEncoder
    decode_hook = profile_hook

    def __init__(self):
        super().__init__()
        self.current_segmentation_dict = "default"
        self.segmentation_dict: typing.Dict[str, ProfileDict] = {self.current_segmentation_dict: ProfileDict()}

    def set(self, key_path, value):
        self.segmentation_dict[self.current_segmentation_dict].set(key_path, value)

    def get(self, key_path, default=None):
        return self.segmentation_dict[self.current_segmentation_dict].get(key_path, default)

    def dump_part(self, file_path, path_in_dict, names=None):
        data = self.get(path_in_dict)
        if names is not None:
            data = dict([(name, data[name]) for name in names ])
        with open(file_path, 'w') as ff:
            json.dump(data, ff, cls=self.json_encoder_class, indent=2)

    def load_part(self, file_path):
        with open(file_path, 'r') as ff:
           return json.load(ff, object_hook=self.decode_hook)

    def dump(self, file_path):
        if not path.exists(path.dirname(file_path)):
            makedirs(path.dirname(file_path))
        dump_view = self.dump_view_profiles()
        with open(file_path, 'w') as ff:
            json.dump(
                {"view_profiles": dump_view,
                 "segment_profile": self.segmentation_dict,
                 "image_spacing": self.image_spacing
                 },
                ff, cls=self.json_encoder_class, indent=2)

    def load(self, file_path):
        try:
            with open(file_path, 'r') as ff:
                data = json.load(ff, object_hook=self.decode_hook)
            try:
                self.load_view_profiles(data["view_profiles"])
            except KeyError:
                logging.error('error in load "view_profiles"')
            except AttributeError:
                logging.error('error in load "view_profiles"')
            try:
                for k, v in data["segment_profile"].items():
                    self.segmentation_dict[k] = v #ProfileDict()
                    print(self.segmentation_dict[k].my_dict)
                    #self.segmentation_dict[k].my_dict = v
            except KeyError:
                logging.error('error in load "segment_profile"')
            except AttributeError:
                logging.error('error in load "segment_profile"')
            try:
                self.image_spacing = data["image_spacing"]
            except KeyError:
                logging.error('error in load "image_spacing"')
        except json.decoder.JSONDecodeError:
            pass