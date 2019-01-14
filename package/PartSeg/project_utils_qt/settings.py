import json
import sys
import typing

from PyQt5.QtCore import QObject, pyqtSignal

from ..partseg_utils.color_image.color_image_base import color_maps
from ..partseg_utils.image_operations import RadiusType
import copy
import numpy as np
from os import path, makedirs
from ..partseg_utils.class_generator import ReadonlyClassEncoder, readonly_hook
from PartSeg.tiff_image import Image
from datetime import datetime


class ImageSettings(QObject):
    """
    :type _image: Image
    noise_removed - for image cleaned by algorithm
    """
    image_changed = pyqtSignal([Image], [int], [str])
    segmentation_changed = pyqtSignal(np.ndarray)
    noise_remove_image_part_changed = pyqtSignal()

    def __init__(self):
        super(ImageSettings, self).__init__()
        self._image: Image = None
        self._image_path = ""
        self._image_spacing = 210, 70, 70
        self._segmentation = None
        self._noise_removed = None
        self.sizes = []
        self.gauss_3d = True
        # self.fixed_range = 0, 255

    @property
    def noise_remove_image_part(self):
        return self._noise_removed

    @noise_remove_image_part.setter
    def noise_remove_image_part(self, val):
        self._noise_removed = val
        self.noise_remove_image_part_changed.emit()

    @property
    def image_spacing(self):
        return self._image.spacing

    @image_spacing.setter
    def image_spacing(self, value):
        assert (len(value) in [2, 3])
        if len(value) == 2:
            self._image.set_spacing([self._image.spacing[0]] + list(value))
        else:
            self._image.set_spacing(value)

    @property
    def segmentation(self) -> np.ndarray:
        return self._segmentation

    @segmentation.setter
    def segmentation(self, val: np.ndarray):
        try:
            self.image.fit_array_to_image(val)
        except ValueError:
            raise ValueError("Segmentation do not fit to image")
        self._segmentation = val
        if val is not None:
            self.sizes = np.bincount(val.flat)
            self.segmentation_changed.emit(val)
        else:
            self.sizes = []

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value: Image):
        self._image = value
        if value.file_path is not None:
            self.image_changed[str].emit(value.file_path)
        self._image_changed()
        self._segmentation = None
        self.sizes = []

        self.image_changed.emit(self._image)
        self.image_changed[int].emit(self._image.channels)

    @property
    def has_channels(self):
        return self._image.channels > 1

    def _image_changed(self):
        pass

    @property
    def image_path(self):
        return self._image.file_path

    @image_path.setter
    def image_path(self, value):
        self._image_path = value
        self.image_changed[str].emmit(self._image_path)

    @property
    def channels(self):
        if self._image is None:
            return 0
        return self._image.channels

    def get_chanel(self, chanel_num):
        return self._image.get_channel(chanel_num)

    def get_information(self, *pos):
        return self._image[pos]

    def components_mask(self):
        return np.array([0] + [1] * self.segmentation.max(), dtype=np.uint8)


class ProfileDict(object):
    def __init__(self):
        self.my_dict = dict()

    def update(self, ob=None, *args, **kwargs):
        if isinstance(ob, ProfileDict):
            self.my_dict.update(ob.my_dict)
            self.my_dict.update(*args, **kwargs)
        elif ob is None:
            self.my_dict.update(**kwargs)
        else:
            self.my_dict.update(ob, **kwargs)

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
                    print(f"{key_path}: {curr_dict.items()}", file=sys.stderr)
                    raise e
        return curr_dict


class ProfileEncoder(ReadonlyClassEncoder):
    def default(self, o):
        if isinstance(o, ProfileDict):
            return {"__ProfileDict__": True, **o.my_dict}
        if isinstance(o, RadiusType):
            return {"__RadiusType__": True, "value": o.value}
        return super().default(o)


def profile_hook(dkt):
    if "__ProfileDict__" in dkt:
        del dkt["__ProfileDict__"]
        res = ProfileDict()
        res.my_dict = dkt
        return res
    if "__RadiusType__" in dkt:
        return RadiusType(dkt["value"])
    return readonly_hook(dkt)


class ViewSettings(ImageSettings):
    colormap_changes = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.color_map = []
        self.border_val = []
        self.current_profile_dict = "default"
        self.view_settings_dict: typing.Dict[str, ProfileDict] = {self.current_profile_dict: ProfileDict()}

    @property
    def chosen_colormap(self):
        return self.get_from_profile("colormaps", ["BlackBlue", "BlackGreen", "BlackMagenta", "BlackRed", "gray"])

    @chosen_colormap.setter
    def chosen_colormap(self, val):
        self.set_in_profile("colormaps", val)
        self.colormap_changes.emit()

    @property
    def available_colormaps(self):
        return list(color_maps.keys())

    def _image_changed(self):
        super()._image_changed()
        self.border_val = self.image.get_ranges()

    def change_profile(self, name):
        self.current_profile_dict = name
        if self.current_profile_dict not in self.view_settings_dict:
            self.view_settings_dict = {self.current_profile_dict: ProfileDict()}

    def set_in_profile(self, key_path, value):
        """function for saving information used in visualization"""
        self.view_settings_dict[self.current_profile_dict].set(key_path, value)

    def get_from_profile(self, key_path, default=None):
        """function for getting information used in visualization"""
        return self.view_settings_dict[self.current_profile_dict].get(key_path, default)

    def dump_view_profiles(self):
        # return json.dumps(self.profile_dict, cls=ProfileEncoder)
        return self.view_settings_dict

    def load_view_profiles(self, dicts):
        for k, v in dicts.items():
            self.view_settings_dict[k] = v  # ProfileDict()
            # self.profile_dict[k].my_dict = v


class SaveSettingsDescription(typing.NamedTuple):
    file_name: str
    values: typing.Union[dict, ProfileDict]


class BaseSettings(ViewSettings):
    json_encoder_class = ProfileEncoder
    decode_hook = staticmethod(profile_hook)

    def get_save_list(self) -> typing.List[SaveSettingsDescription]:
        return [SaveSettingsDescription("segmentation_settings.json", self.segmentation_dict),
                SaveSettingsDescription("view_settings.json", self.view_settings_dict)]

    def __init__(self, json_path):
        super().__init__()
        self.current_segmentation_dict = "default"
        self.segmentation_dict: typing.Dict[str, ProfileDict] = {self.current_segmentation_dict: ProfileDict()}
        self.json_folder_path = json_path
        self.last_executed_algorithm = ""

    def set(self, key_path, value):
        """function for saving general state (not visualization) """
        self.segmentation_dict[self.current_segmentation_dict].set(key_path, value)

    def get(self, key_path, default=None):
        """function for getting general state (not visualization) """
        return self.segmentation_dict[self.current_segmentation_dict].get(key_path, default)

    def dump_part(self, file_path, path_in_dict, names=None):
        data = self.get(path_in_dict)
        if names is not None:
            data = dict([(name, data[name]) for name in names])
        with open(file_path, 'w') as ff:
            json.dump(data, ff, cls=self.json_encoder_class, indent=2)

    def load_part(self, file_path):
        with open(file_path, 'r') as ff:
            return json.load(ff, object_hook=self.decode_hook)

    def dump(self, folder_path=None):
        if folder_path is None:
            folder_path = self.json_folder_path
        if not path.exists(folder_path):
            makedirs(folder_path)
        errors_list = []
        for el in self.get_save_list():
            try:
                dump_string = json.dumps(el.values, cls=self.json_encoder_class, indent=2)
                with open(path.join(folder_path, el.file_name), 'w') as ff:
                    ff.write(dump_string)
            except Exception as e:
                errors_list.append((e, path.join(folder_path, el.file_name)))
        if errors_list:
            print(errors_list, file=sys.stderr)
        return errors_list

    def load(self, folder_path=None):
        if folder_path is None:
            folder_path = self.json_folder_path
        errors_list = []
        for el in self.get_save_list():
            file_path = path.join(folder_path, el.file_name)
            if not path.exists(file_path):
                continue
            try:
                with open(file_path, 'r') as ff:
                    data = json.load(ff, object_hook=self.decode_hook)
                el.values.update(data)
            except Exception as e:
                timestamp = datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
                base_path, ext = path.splitext(file_path)
                import os
                os.rename(file_path, base_path + "_" + timestamp + ext)
                errors_list.append(e)
        if errors_list:
            print(errors_list, file=sys.stderr)
        return errors_list
