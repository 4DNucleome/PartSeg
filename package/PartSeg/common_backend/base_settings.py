import json
import os
import os.path
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import napari.utils.theme
import numpy as np
from napari.utils import Colormap
from napari.utils.theme import template as napari_template
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QMessageBox, QWidget

from PartSeg.common_backend.partially_const_dict import PartiallyConstDict
from PartSegCore import register
from PartSegCore.color_image import default_colormap_dict, default_label_dict
from PartSegCore.color_image.base_colors import starting_colors
from PartSegCore.io_utils import load_metadata_base
from PartSegCore.json_hooks import ProfileDict, ProfileEncoder, check_loaded_dict
from PartSegCore.project_info import AdditionalLayerDescription, HistoryElement, ProjectInfoBase
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.algorithm_base import SegmentationResult
from PartSegImage import Image

try:
    from napari.qt import get_stylesheet
except ImportError:  # pragma: no cover
    from napari.resources import get_stylesheet

if hasattr(napari.utils.theme, "get_theme"):

    def get_theme(name: str) -> dict:
        theme = napari.utils.theme.get_theme(name)
        if "canvas" in theme and theme["canvas"] != "black":
            theme["canvas"] = "black"
            napari.utils.theme.register_theme(name, theme)
        return theme


else:  # pragma: no cover

    def get_theme(name: str) -> dict:
        theme = napari.utils.theme.palettes[name]
        theme["canvas"] = "black"
        return theme


DIR_HISTORY = "io.dir_location_history"
FILE_HISTORY = "io.files_open_history"


class ImageSettings(QObject):
    """
    Base class for all PartSeg settings. Keeps information about current Image.
    """

    image_changed = Signal([Image], [int], [str])
    image_spacing_changed = Signal()
    """:py:class:`Signal` ``([Image], [int], [str])`` emitted when image has changed"""
    roi_changed = Signal(ROIInfo)
    """
    :py:class:`.Signal`
    emitted when roi has changed
    """
    roi_clean = Signal()
    additional_layers_changed = Signal()

    def __init__(self):
        super().__init__()
        self._image: Optional[Image] = None
        self._image_path = ""
        self._roi_info = ROIInfo(None)
        self._additional_layers = {}
        self._parent: Optional[QWidget] = None

    def set_parent(self, parent: QWidget):
        self._parent = parent

    @property
    def full_segmentation(self):  # pragma: no cover
        raise AttributeError("full_segmentation not supported")

    @full_segmentation.setter
    def full_segmentation(self, val):  # pragma: no cover # pylint: disable=R0201
        raise AttributeError("full_segmentation not supported")

    @property
    def noise_remove_image_part(self):  # pragma: no cover
        raise AttributeError("noise_remove_image_part not supported")

    @noise_remove_image_part.setter
    def noise_remove_image_part(self, val):  # pragma: no cover # pylint: disable=R0201
        raise AttributeError("noise_remove_image_part not supported")

    @property
    def additional_layers(self) -> Dict[str, AdditionalLayerDescription]:
        return self._additional_layers

    @additional_layers.setter
    def additional_layers(self, val):  # pragma: no cover  # pylint: disable=R0201
        raise AttributeError("additional_layers assign not supported")

    @property
    def image_spacing(self):
        """:py:meth:`Image.spacing` proxy"""
        if self._image is not None:
            return self._image.spacing
        return ()

    def is_image_2d(self):
        """:py:meth:`Image.is_2d` proxy"""
        return self._image is None or self._image.is_2d

    @image_spacing.setter
    def image_spacing(self, value):
        if len(value) not in [2, 3]:  # pragma: no cover
            raise ValueError(f"value parameter should have length 2 or 3. Current length is {len(value)}.")
        if len(value) == 2:
            self._image.set_spacing(tuple([self._image.spacing[0]] + list(value)))
        else:
            self._image.set_spacing(value)
        self.image_spacing_changed.emit()

    @property
    def segmentation(self) -> np.ndarray:  # pragma: no cover
        """current roi"""
        warnings.warn("segmentation parameter is renamed to roi", DeprecationWarning)
        return self.roi

    @property
    def roi(self) -> np.ndarray:
        """current roi"""
        return self._roi_info.roi

    @property
    def segmentation_info(self) -> ROIInfo:  # pragma: no cover
        warnings.warn("segmentation info parameter is renamed to roi", DeprecationWarning)
        return self.roi_info

    @property
    def roi_info(self) -> ROIInfo:
        return self._roi_info

    @roi.setter
    def roi(self, val: Union[np.ndarray, ROIInfo]):
        if val is None:
            self._roi_info = ROIInfo(val)
            self._additional_layers = {}
            self.roi_clean.emit()
            return
        try:
            if isinstance(val, np.ndarray):
                self._roi_info = ROIInfo(self.image.fit_array_to_image(val))
            else:
                self._roi_info = val.fit_to_image(self.image)
        except ValueError:  # pragma: no cover
            raise ValueError("roi do not fit to image")
        self._additional_layers = {}
        self.roi_changed.emit(self._roi_info)

    @property
    def sizes(self):
        return self._roi_info.sizes

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
        self._roi_info = ROIInfo(None)

        self.image_changed.emit(self._image)
        self.image_changed[int].emit(self._image.channels)

    @property
    def has_channels(self):
        return self.channels > 1

    def _image_changed(self):
        """Reimplement hook for change of main image"""

    @property
    def image_path(self):
        if self.image is not None:
            return self._image.file_path
        return ""

    @property
    def image_shape(self):
        # TODO analyse and decide if channels should be part of shape
        if self.image is not None:
            return self._image.shape
        return ()

    @image_path.setter
    def image_path(self, value):
        self._image_path = value
        self.image_changed[str].emmit(self._image_path)

    @property
    def channels(self):
        if self._image is None:
            return 0
        return self._image.channels

    def components_mask(self):
        return np.array([0] + [1] * np.max(self.roi), dtype=np.uint8)


class ColormapDict(PartiallyConstDict[Colormap]):
    """
    Dict for mixing custom colormap with predefined ones
    """

    if os.path.basename(sys.argv[0]) in ["sphinx-build", "sphinx-build.exe"]:
        const_item_dict = {}
    else:
        const_item_dict = default_colormap_dict
    """
    Non removable items for this dict. Current value is :py:data:`default_colormap_dict`
    """

    @property
    def colormap_removed(self):
        """
        Signal that colormap is removed form dict
        """
        return self.item_removed

    @property
    def colormap_added(self):
        """
        Signal that colormap is added to dict
        """
        return self.item_added


class LabelColorDict(PartiallyConstDict[list]):
    """
    Dict for mixing custom label colors with predefined ones`
    """

    const_item_dict = default_label_dict
    """Non removable items for this dict. Current value is :py:data:`default_label_dict`"""

    def get_array(self, key: str) -> np.ndarray:
        """Get labels as numpy array"""
        return np.array(self[key][0], dtype=np.uint8)


class ViewSettings(ImageSettings):
    colormap_changes = Signal()
    labels_changed = Signal()
    theme_changed = Signal()
    profile_data_changed = Signal(str, object)
    """Signal about changes in stored data (set with set_in_profile)"""

    def __init__(self):
        super().__init__()
        self.color_map = []
        self.border_val = []
        self.current_profile_dict = "default"
        self.view_settings_dict = ProfileDict()
        self.colormap_dict = ColormapDict(self.get_from_profile("custom_colormap", {}))
        self.label_color_dict = LabelColorDict(self.get_from_profile("custom_label_colors", {}))
        self.cached_labels: Optional[Tuple[str, np.ndarray]] = None

    @property
    def theme_name(self) -> str:
        return self.get_from_profile("theme", "light")

    @property
    def style_sheet(self):
        theme = get_theme(self.theme_name)
        # TODO understand qss overwrite mechanism
        return napari_template("\n".join(register.qss_list) + get_stylesheet() + "\n".join(register.qss_list), **theme)

    @theme_name.setter
    def theme_name(self, value: str):
        if value not in napari.utils.theme.palettes:
            raise ValueError(f"Unsupported theme {value}. Supported one: {self.theme_list()}")
        if value == self.theme_name:
            return
        self.set_in_profile("theme", value)
        self.theme_changed.emit()

    @staticmethod
    def theme_list():
        return list(napari.utils.theme.palettes.keys())

    @property
    def chosen_colormap(self):
        data = self.get_from_profile("colormaps", starting_colors[:])
        res = [x for x in data if x in self.colormap_dict]
        if len(res) != data:
            if not res:
                res = starting_colors[:]
            self.set_in_profile("colormaps", res)
        return res

    @chosen_colormap.setter
    def chosen_colormap(self, val):
        self.set_in_profile("colormaps", val)
        self.colormap_changes.emit()

    @property
    def current_labels(self):
        return self.get_from_profile("labels_used", "default")

    @current_labels.setter
    def current_labels(self, val):
        if val not in self.label_color_dict:
            raise ValueError(f"Unknown label scheme name '{val}'")
        self.set_in_profile("labels_used", val)
        self.labels_changed.emit()

    @property
    def label_colors(self):
        key = self.current_labels
        if key not in self.label_color_dict:
            key = "default"

        if not (self.cached_labels and key == self.cached_labels[0]):
            self.cached_labels = key, self.label_color_dict.get_array(key)

        return self.cached_labels[1]

    def chosen_colormap_change(self, name, visibility):
        colormaps = set(self.chosen_colormap)
        if visibility:
            colormaps.add(name)
        else:
            try:
                colormaps.remove(name)
            except KeyError:
                pass
        # TODO update sorting rule
        self.chosen_colormap = list(sorted(colormaps, key=self.colormap_dict.get_position))

    def get_channel_info(self, view: str, num: int, default: Optional[str] = None) -> str:
        cm = self.chosen_colormap
        if default is None:
            default = cm[num % len(cm)]
        resp = self.get_from_profile(f"{view}.cmap{num}", default)
        if resp not in self.colormap_dict:
            resp = cm[num % len(cm)]
            self.set_in_profile(f"{view}.cmap{num}", resp)
        return resp

    def set_channel_info(self, view: str, num, value: str):
        self.set_in_profile(f"{view}.cmap{num}", value)

    @property
    def available_colormaps(self):
        return list(self.colormap_dict.keys())

    def _image_changed(self):
        self.border_val = self.image.get_ranges()
        super()._image_changed()

    def change_profile(self, name):
        self.current_profile_dict = name
        if self.current_profile_dict not in self.view_settings_dict:
            self.view_settings_dict = {self.current_profile_dict: ProfileDict()}

    def set_in_profile(self, key_path, value):
        """
        Function for saving information used in visualization. This is accessor to
        :py:meth:`~.ProfileDict.set` of inner variable.

        :param key_path: dot separated path
        :param value: value to store. The value need to be json serializable."""
        self.view_settings_dict.set(f"{self.current_profile_dict}.{key_path}", value)
        self.profile_data_changed.emit(key_path, value)

    def get_from_profile(self, key_path, default=None):
        """
        Function for getting information used in visualization. This is accessor to
        :py:meth:`~.ProfileDict.get` of inner variable.

        :param key_path: dot separated path
        :param default: default value if key is missed
        """
        return self.view_settings_dict.get(f"{self.current_profile_dict}.{key_path}", default)

    def dump_view_profiles(self):
        return self.view_settings_dict


class SaveSettingsDescription(NamedTuple):
    file_name: str
    values: Union[dict, ProfileDict]


class BaseSettings(ViewSettings):
    """
    :ivar json_folder_path: default location for saving/loading settings data
    :ivar last_executed_algorithm: name of last executed algorithm.
    :cvar save_locations_keys: list of names of distinct save location.
        location are stored in "io"

    """

    mask_changed = Signal()
    points_changed = Signal()
    mask_representation_changed = Signal()
    request_load_files = Signal(list)
    """:py:class:`~.Signal` mask changed signal"""
    json_encoder_class = ProfileEncoder
    load_metadata = staticmethod(load_metadata_base)
    algorithm_changed = Signal()
    """:py:class:`~.Signal` emitted when current algorithm should be changed"""
    save_locations_keys = []
    data_changed = Signal(str, object)

    def __init__(self, json_path):
        super().__init__()
        self._current_roi_dict = "default"
        self._roi_dict = ProfileDict()
        self.json_folder_path = json_path
        self.last_executed_algorithm = ""
        self.history: List[HistoryElement] = []
        self.history_index = -1
        self.last_executed_algorithm = ""
        self._points = None

    def _image_changed(self):
        super()._image_changed()
        self.points = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value if value is not None else None
        self.points_changed.emit()

    def set_segmentation_result(self, result: SegmentationResult):
        if result.info_text and self._parent is not None:
            QMessageBox().information(self._parent, "Algorithm info", result.info_text)

        self._additional_layers = result.additional_layers
        self.last_executed_algorithm = result.parameters.algorithm
        self.set(f"algorithms.{result.parameters.algorithm}", result.parameters.values)
        try:
            roi = self.image.fit_array_to_image(result.roi)
            alternative_list = {
                k: self.image.fit_array_to_image(v) for k, v in result.alternative_representation.items()
            }
        except ValueError:  # pragma: no cover
            raise ValueError("roi do not fit to image")
        self._roi_info = ROIInfo(roi, result.roi_annotation, alternative_list)
        self.roi_changed.emit(self._roi_info)

    def _load_files_call(self, files_list: List[str]):
        self.request_load_files.emit(files_list)

    def mask_representation_changed_emit(self):
        self.mask_representation_changed.emit()

    def add_history_element(self, elem: HistoryElement) -> None:
        self.history_index += 1
        if self.history_index < len(self.history) and self.cmp_history_element(elem, self.history[self.history_index]):
            self.history[self.history_index] = elem
        else:
            self.history = self.history[: self.history_index]
            self.history.append(elem)

    def history_size(self) -> int:
        return self.history_index + 1

    def history_redo_size(self) -> int:
        if self.history_index + 1 == len(self.history):
            return 0
        return len(self.history[self.history_index + 1 :])

    def history_redo_clean(self) -> None:
        self.history = self.history[: self.history_size()]

    def history_current_element(self) -> HistoryElement:
        return self.history[self.history_index]

    def history_next_element(self) -> HistoryElement:
        return self.history[self.history_index + 1]

    def history_pop(self) -> Optional[HistoryElement]:
        if self.history_index != -1:
            self.history_index -= 1
            return self.history[self.history_index + 1]
        return None

    def set_history(self, history: List[HistoryElement]):
        self.history = history
        self.history_index = len(self.history) - 1

    def get_history(self) -> List[HistoryElement]:
        return self.history[: self.history_index + 1]

    @staticmethod
    def cmp_history_element(el1, el2):
        return False

    @property
    def mask(self):
        return self._image.mask

    @mask.setter
    def mask(self, value):
        try:
            self._image.set_mask(value)
            self.mask_changed.emit()
        except ValueError:
            raise ValueError("mask do not fit to image")

    def get_save_list(self) -> List[SaveSettingsDescription]:
        """List of files in which program save the state."""
        return [
            SaveSettingsDescription("segmentation_settings.json", self._roi_dict),
            SaveSettingsDescription("view_settings.json", self.view_settings_dict),
        ]

    def get_path_history(self) -> List[str]:
        """
        return list containing last 10 elements added with :py:meth:`.add_path_history` and
        last opened in each category form :py:attr:`save_location_keys`
        """
        res = self.get(DIR_HISTORY, [])[:]
        for name in self.save_locations_keys:
            val = self.get("io." + name, str(Path.home()))
            if val not in res:
                res = res + [val]
        return res

    def get_last_files(self) -> list:
        return self.get(FILE_HISTORY, [])

    @staticmethod
    def _add_elem_to_list(data_list: list, value: Any) -> list:
        try:
            data_list.remove(value)
        except ValueError:
            data_list = data_list[:9]
        return [value] + data_list

    def add_path_history(self, dir_path: str):
        """Save path in history of visited directories. Store only 10 last"""
        self.set(DIR_HISTORY, self._add_elem_to_list(self.get(DIR_HISTORY, []), dir_path))

    def add_load_files_history(self, file_path: List[str], load_method: str):
        self.set(FILE_HISTORY, self._add_elem_to_list(self.get(FILE_HISTORY, []), [file_path, load_method]))
        self.add_path_history(os.path.dirname(file_path[0]))

    def set(self, key_path: str, value):
        """
        function for saving general state (not visualization). This is accessor to
        :py:meth:`~.ProfileDict.set` of inner variable.

        :param key_path: dot separated path
        :param value: value to store. The value need to be json serializable.
        """
        self._roi_dict.set(f"{self._current_roi_dict}.{key_path}", value)
        self.data_changed.emit(key_path, value)

    def get(self, key_path: str, default=None):
        """
        Function for getting general state (not visualization). This is accessor to
        :py:meth:`~.ProfileDict.get` of inner variable.

        :param key_path: dot separated path
        :param default: default value if key is missed
        """
        return self._roi_dict.get(f"{self._current_roi_dict}.{key_path}", default)

    def dump_part(self, file_path, path_in_dict, names=None):
        data = self.get(path_in_dict)
        if names is not None:
            data = {name: data[name] for name in names}
        with open(file_path, "w") as ff:
            json.dump(data, ff, cls=self.json_encoder_class, indent=2)

    def load_part(self, file_path):
        data = self.load_metadata(file_path)
        bad_key = []
        if isinstance(data, dict) and not check_loaded_dict(data):
            for k, v in data.items():
                if not check_loaded_dict(v):
                    bad_key.append(k)
            for el in bad_key:
                del data[el]
        elif isinstance(data, ProfileDict) and not data.verify_data():
            bad_key = data.filter_data()
        return data, bad_key

    def dump(self, folder_path: Optional[str] = None):
        """
        Save current application settings to disc.

        :param folder_path: path to directory in which data should be saved.
            If is None then use :py:attr:`.json_folder_path`
        """
        if folder_path is None:
            folder_path = self.json_folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        errors_list = []
        for el in self.get_save_list():
            try:
                dump_string = json.dumps(el.values, cls=self.json_encoder_class, indent=2)
                with open(os.path.join(folder_path, el.file_name), "w") as ff:
                    ff.write(dump_string)
            except Exception as e:
                errors_list.append((e, os.path.join(folder_path, el.file_name)))
        if errors_list:
            print(errors_list, file=sys.stderr)
        return errors_list

    def load(self, folder_path: Optional[str] = None):
        """
        Load settings state from given directory

        :param folder_path: path to directory in which data should be saved.
            If is None then use :py:attr:`.json_folder_path`
        """
        if folder_path is None:
            folder_path = self.json_folder_path
        errors_list = []
        for el in self.get_save_list():
            file_path = os.path.join(folder_path, el.file_name)
            if not os.path.exists(file_path):
                continue
            error = False
            try:
                data: ProfileDict = self.load_metadata(file_path)
                if not data.verify_data():
                    errors_list.append((file_path, data.filter_data()))
                    error = True
                el.values.update(data)
            except Exception as e:
                error = True
                errors_list.append((file_path, e))
            finally:
                if error:
                    timestamp = datetime.today().strftime("%Y-%m-%d_%H_%M_%S")
                    base_path, ext = os.path.splitext(file_path)
                    os.rename(file_path, base_path + "_" + timestamp + ext)

        if errors_list:
            print(errors_list, file=sys.stderr)
        return errors_list

    def get_project_info(self) -> ProjectInfoBase:
        """Get all information needed to save project"""
        raise NotImplementedError

    def set_project_info(self, data: ProjectInfoBase):
        """Set project info"""
        raise NotImplementedError

    @staticmethod
    def verify_image(image: Image, silent=True) -> Union[Image, bool]:
        if image.is_time:
            if image.is_stack:
                raise TimeAndStackException()
            if silent:
                return image.swap_time_and_stack()
            raise SwapTimeStackException()
        return True


class SwapTimeStackException(Exception):
    """
    Exception which inform that current image shape is not supported,
    but can be if time and stack axes were swapped
    """


class TimeAndStackException(Exception):
    """
    Exception which inform that current image has both time
    and stack dat which is not supported
    """
