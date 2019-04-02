import json
import sys
import typing
from pathlib import Path

from qtpy.QtCore import QObject, Signal

from PartSeg.utils.io_utils import ProjectInfoBase
from PartSeg.utils.json_hooks import ProfileDict, ProfileEncoder, profile_hook, check_loaded_dict
from ..utils.color_image.color_image_base import color_maps
import numpy as np
from os import path, makedirs
from PartSeg.tiff_image import Image
from datetime import datetime


class ImageSettings(QObject):
    """
    :type _image: Image
    noise_removed - for image cleaned by algorithm
    """
    image_changed = Signal([Image], [int], [str])
    segmentation_changed = Signal(np.ndarray)
    segmentation_clean = Signal()
    noise_remove_image_part_changed = Signal()

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

    def is_image_2d(self):
        return self._image is None or self._image.is_2d

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
        if val is not None:
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
            self.segmentation_clean.emit()

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value: Image):
        if value is None:
            return
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


class ViewSettings(ImageSettings):
    colormap_changes = Signal()

    def __init__(self):
        super().__init__()
        self.color_map = []
        self.border_val = []
        self.current_profile_dict = "default"
        self.view_settings_dict = ProfileDict()

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
        self.border_val = self.image.get_ranges()
        super()._image_changed()

    def change_profile(self, name):
        self.current_profile_dict = name
        if self.current_profile_dict not in self.view_settings_dict:
            self.view_settings_dict = {self.current_profile_dict: ProfileDict()}

    def set_in_profile(self, key_path, value):
        """function for saving information used in visualization"""
        self.view_settings_dict.set(f"{self.current_profile_dict}.{key_path}", value)

    def get_from_profile(self, key_path, default=None):
        """function for getting information used in visualization"""
        return self.view_settings_dict.get(f"{self.current_profile_dict}.{key_path}", default)

    def dump_view_profiles(self):
        # return json.dumps(self.profile_dict, cls=ProfileEncoder)
        return self.view_settings_dict


class SaveSettingsDescription(typing.NamedTuple):
    file_name: str
    values: typing.Union[dict, ProfileDict]


class BaseSettings(ViewSettings):
    json_encoder_class = ProfileEncoder
    decode_hook = staticmethod(profile_hook)
    algorithm_changed = Signal()
    save_locations_keys = []

    def get_save_list(self) -> typing.List[SaveSettingsDescription]:
        return [SaveSettingsDescription("segmentation_settings.json", self.segmentation_dict),
                SaveSettingsDescription("view_settings.json", self.view_settings_dict)]

    def __init__(self, json_path):
        super().__init__()
        self.current_segmentation_dict = "default"
        self.segmentation_dict = ProfileDict()
        self.json_folder_path = json_path
        self.last_executed_algorithm = ""

    def get_path_history(self) -> typing.List[str]:
        res = self.get("io.history", [])
        for name in self.save_locations_keys:
            val = self.get("io." + name,  str(Path.home()))
            if val not in res:
                res = res + [val]
        return res

    def add_path_history(self, dir_path: str):
        history = self.get("io.history", [])
        if dir_path not in history:
            self.set("io.history", history[-9:] + [dir_path])

    def set(self, key_path, value):
        """function for saving general state (not visualization) """
        self.segmentation_dict.set(f"{self.current_segmentation_dict}.{key_path}", value)

    def get(self, key_path, default=None):
        """function for getting general state (not visualization) """
        return self.segmentation_dict.get(f"{self.current_segmentation_dict}.{key_path}", default)

    def dump_part(self, file_path, path_in_dict, names=None):
        data = self.get(path_in_dict)
        if names is not None:
            data = dict([(name, data[name]) for name in names])
        with open(file_path, 'w') as ff:
            json.dump(data, ff, cls=self.json_encoder_class, indent=2)

    def load_part(self, file_path):
        with open(file_path, 'r') as ff:
            data = json.load(ff, object_hook=self.decode_hook)
        bad_key = []
        if isinstance(data, dict):
            if not check_loaded_dict(data):
                for k, v in data.items():
                    if not check_loaded_dict(v):
                        bad_key.append(k)
                for el in bad_key:
                    del data[el]
        elif isinstance(data, ProfileDict):
            if not data.verify_data():
                bad_key = data.filter_data()
        return data, bad_key

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
            error = False
            try:
                with open(file_path, 'r') as ff:
                    data: ProfileDict = json.load(ff, object_hook=self.decode_hook)
                    if not data.verify_data():
                        errors_list.append((file_path, data.filter_data()))
                        error = True
                el.values.update(data)
            except Exception as e:
                error = True
                errors_list.append((file_path, e))
            finally:
                if error:
                    timestamp = datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
                    base_path, ext = path.splitext(file_path)
                    import os
                    os.rename(file_path, base_path + "_" + timestamp + ext)

        if errors_list:
            print(errors_list, file=sys.stderr)
        return errors_list

    def get_project_info(self) -> ProjectInfoBase:
        raise NotImplementedError

    def set_project_info(self, data: ProjectInfoBase):
        raise NotImplementedError

    @staticmethod
    def verify_image(image: Image, silent=True) -> typing.Union[Image, bool]:
        raise NotImplementedError
