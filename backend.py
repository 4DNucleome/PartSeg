# coding=utf-8
from __future__ import print_function, division
import matplotlib
from matplotlib import pyplot
import numpy as np
import json
import auto_fit as af
import logging
from enum import Enum
from calculation_plan import CalculationPlan
GAUSS = "Gauss"

from utils import class_to_dict, dict_set_class
from segment import SegmentationProfile, Segment, UPPER
from statistics_calculation import StatisticProfile, calculate_volume_surface
from image_operations import gaussian, dilate

UNITS_DICT = {
    "Volume": "{}^3",
    "Mass": "pixel sum",
    "Border Volume": "{}^3",
    "Border Surface": "{}^2",
    "Border Surface Opening": "{}^2",
    "Border Surface Closing": "{}^2",
    "Pixel min": "pixel brightness",
    "Pixel max": "pixel brightness",
    "Pixel mean": "pixel brightness",
    "Pixel median": "pixel brightness",
    "Pixel std": "pixel brightness",
    "Mass to Volume": "pixel sum/{}^3",
    "Volume to Border Surface": "{}",
    "Volume to Border Surface Opening": "{}",
    "Volume to Border Surface Closing": "{}",
    "Moment of inertia": "",
    "Noise_std": "pixel brightness"
}

UNITS_LIST = ["mm", u"µm", "nm", "pm"]


class MaskChange(Enum):
    prev_seg = 1
    next_seg = 2


class SegmentationSettings(object):
    def __init__(self):
        self._threshold = 0
        self._threshold_list = []
        self._threshold_type = UPPER
        self._threshold_layer_separate = False
        self._current_layer = 0
        self._use_gauss = False
        self._gauss_radius = 1
        self._image = None
        self._mask = None
        self._file_path = ""
        self._gauss_image = None

    def set_new_data(self, image, file_path):
        self._image = image
        self._file_path = file_path

    @property
    def threshold(self):
        if self._threshold_layer_separate:
            return self._threshold_list
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """
        :type value: int | list[int]
        :param value:
        :return:
        """
        if isinstance(value, list):
            self._threshold_list = value
        if self._threshold_layer_separate:
            self._threshold_list[self._current_layer] = value
        else:
            self._threshold = value

    @property
    def threshold_type(self):
        return self._threshold_type

    @threshold_type.setter
    def threshold_type(self, value):
        self._threshold_type = value

    @property
    def use_gauss(self):
        return self._use_gauss

    @use_gauss.setter
    def use_gauss(self, value):
        self._use_gauss = value

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
        self._gauss_image = gaussian(value, self._gauss_radius)

    @property
    def segment_image(self):
        if self._use_gauss:
            return self._gauss_image
        return self._image

    @property
    def gauss_radius(self):
        return self._gauss_radius

    @gauss_radius.setter
    def gauss_radius(self, value):
        self._gauss_radius = value
        self._gauss_image = gaussian(self._image, self._gauss_radius)

    @property
    def mask(self):
        return self._mask


class Settings(object):
    """
    :type segmentation_profiles_dict: dict[str, SegmentationProfile]
    :type statistics_profile_dict: dict[str, StatisticProfile]
    :type threshold: int
    :type threshold_list: list[int]
    :type threshold_type: str
    :type minimum_size: int
    :type image: np.ndarray
    :type image_change_callback: list[(() -> None) | (() -> None, object)]
    :type batch_plans: dict[str, CalculationPlan]
    """
    def __init__(self, settings_path):
        self.color_map_name = "cubehelix"
        self.color_map = matplotlib.cm.get_cmap(self.color_map_name)
        self.callback_colormap = []
        self.callback_colormap_list = []
        self.callback_change_layer = []
        self.chosen_colormap = pyplot.colormaps()
        self.segmentation_profiles_dict = dict()
        self.profiles_list_changed_callback = []
        self.use_gauss = False
        self.use_draw_result = False
        self.draw_callback = []
        self.threshold = 33000
        self.threshold_list = []
        self.threshold_type = UPPER
        self.threshold_layer_separate = False
        self.minimum_size = 100
        self.overlay = 0.7
        self.mask_overlay = 0.7
        self.power_norm = 1
        self.image = None
        self.min_value = 0
        self.max_value = 0
        self.original_image = None
        self.image_clean_profile = None
        self.gauss_image = None
        self.mask = None
        self.gauss_radius = 1
        self.image_change_callback = []
        self.threshold_change_callback = []
        self.threshold_type_change_callback = []
        self.minimum_size_change_callback = []
        self.metadata_changed_callback = []
        self.layer_num = 0
        self.open_directory = None
        self.batch_directory = None
        self.open_filter = None
        self.save_directory = None
        self.save_filter = None
        self.export_directory = None
        self.export_filter = None
        self.spacing = [5, 5, 30]
        self.voxel_size = [5, 5, 30]
        self.size_unit = "nm"
        self.advanced_menu_geometry = None
        self.file_path = ""
        self.protect = False
        # TODO read more about zstd compression
        self.prev_segmentation_settings = []
        self.next_segmentation_settings = []
        self.mask_dilate_radius = 0
        self.scale_factor = 0.97
        self.statistics_profile_dict = dict()
        self.statistic_dirs = None
        self.batch_plans = {}
        try:
            self.load(settings_path)
        except ValueError as e:
            logging.error("Saved profile problem: {}".format(e))

    def change_profile(self, name):
        print("%%%%%%%% {}".format(name))
        prof = self.segmentation_profiles_dict[str(name)]
        dict_set_class(self, prof.get_parameters(), *SegmentationProfile.SEGMENTATION_PARAMETERS)
        for fun in self.threshold_change_callback:
            fun()

    def add_profiles_list_callback(self, callback):
        self.profiles_list_changed_callback.append(callback)

    def get_profile_dict(self):
        return class_to_dict(self, "threshold", "threshold_list", "threshold_type", "minimum_size", "use_gauss",
                             "gauss_radius", "threshold_layer_separate")

    def dump_profiles(self, file_path):
        profiles_list = [x.get_parameters() for x in self.segmentation_profiles_dict.values()]
        with open(file_path, "w") as ff:
            json.dump(profiles_list, ff)

    def load_profiles(self, file_path):
        with open(file_path, "r") as ff:
            profiles_list = json.load(ff)
            for prof in profiles_list:
                self.segmentation_profiles_dict[prof["name"]] = SegmentationProfile(**prof)
        for fun in self.threshold_change_callback:
            fun()

    def dump_statistics(self, file_path):
        res = [x.get_parameters()
               for x in self.statistics_profile_dict.values()]

        json_str = json.dumps(res)
        with open(file_path, 'w') as ff:
            ff.write(json_str)

    def load_statistics(self, file_path):
        with open(file_path, 'r') as ff:
            statistics_list = json.load(ff)
            for stat in statistics_list:
                self.statistics_profile_dict[stat["name"]] = StatisticProfile(settings=self, **stat)

    def dump_calculation_plans(self, file_path):
        json_str = json.dumps([x.get_parameters() for x in self.batch_plans.values()])
        with open(file_path, "w") as ff:
            ff.write(json_str)

    def load_calculation_plans(self, file_path):
        with open(file_path, "r") as ff:
            calculation_plans = json.load(ff)
        for plan in calculation_plans:
            calc_plan = CalculationPlan.dict_load(plan)
            self.batch_plans[calc_plan.name] = calc_plan

    def dump(self, file_path):
        important_data = \
            class_to_dict(self, "open_directory", "open_filter", "save_directory", "save_filter", "spacing",
                          "voxel_size", "size_unit", "threshold", "color_map_name", "overlay", "minimum_size",
                          "gauss_radius", "export_filter", "export_directory", "scale_factor", "statistic_dirs",
                          "chosen_colormap", "batch_directory")
        # TODO Batch plans dump
        important_data["profiles"] = [x.get_parameters() for x in self.segmentation_profiles_dict.values()]
        important_data["statistics"] = \
            [class_to_dict(x, "name", "chosen_fields", "reversed_brightness", "use_gauss_image")
             for x in self.statistics_profile_dict.values()]
        important_data["batch_plans"] = [x.get_parameters() for x in self.batch_plans.values()]
        json_str = json.dumps(important_data)
        with open(file_path, "w") as ff:
            ff.write(json_str)

    def load(self, file_path):
        if file_path is None:
            return
        try:
            with open(file_path, "r") as ff:
                important_data = json.load(ff)
            dict_set_class(self, important_data, "open_directory", "open_filter", "save_directory", "save_filter",
                           "spacing", "voxel_size", "size_unit", "threshold", "color_map_name", "overlay",
                           "minimum_size", "gauss_radius", "export_filter", "export_directory", "scale_factor",
                           "statistic_dirs", "chosen_colormap", "batch_directory")
            # TODO Batch plans load
            chosen_colormap = set(self.chosen_colormap)
            avail_colormap = set(pyplot.colormaps())
            self.chosen_colormap = list(sorted(chosen_colormap & avail_colormap))
            self.color_map = matplotlib.cm.get_cmap(self.color_map_name)
            for prof in important_data["profiles"]:
                self.segmentation_profiles_dict[prof["name"]] = SegmentationProfile(**prof)
            for stat in important_data["statistics"]:
                self.statistics_profile_dict[stat["name"]] = StatisticProfile(settings=self, **stat)
            for plan in important_data["batch_plans"]:
                calc_plan = CalculationPlan.dict_load(plan)
                self.batch_plans[calc_plan.name] = calc_plan
        except IOError:
            logging.warning("No configuration file")
            pass
        except KeyError as e:
            logging.warning("Bad configuration: {}".format(e))

    def change_segmentation_mask(self, segment, order, save_draw):
        """
        :type segment: Segment
        :type order: MaskChange
        :type save_draw: bool
        :return:
        """
        save_fields = ["threshold", "threshold_list", "threshold_type", "threshold_layer_separate",
                       "minimum_size", "use_gauss", "use_draw_result", "mask_dilate_radius", "mask", "gauss_radius"]
        if order == MaskChange.prev_seg and len(self.prev_segmentation_settings) == 0:
            return

        current_mask = segment.get_segmentation()
        seg_settings = class_to_dict(self, *save_fields)
        seg_settings["draw_points"] = tuple(map(list, np.nonzero(np.array(segment.draw_canvas == 1))))
        seg_settings["erase_points"] = tuple(map(list, np.nonzero(np.array(segment.draw_canvas == 2))))
        save_draw_bck = np.copy(segment.draw_canvas)
        if order == MaskChange.next_seg:
            self.prev_segmentation_settings.append(seg_settings)
            if self.mask_dilate_radius > 0:
                self.mask = dilate(current_mask, self.mask_dilate_radius)
            else:
                self.mask = current_mask
            if len(self.next_segmentation_settings) > 0:
                new_seg = self.next_segmentation_settings.pop()
                new_seg["mask"] = self.mask
            else:
                new_seg = None
            save_fields = save_fields[:-1]
        else:
            self.next_segmentation_settings.append(seg_settings)
            new_seg = self.prev_segmentation_settings.pop()
        segment.draw_canvas[...] = 0
        if new_seg is not None:
            dict_set_class(self, new_seg, *save_fields)
            segment.draw_canvas[tuple(map(lambda x: np.array(x, dtype=np.uint32), new_seg["draw_points"]))] = 1
            segment.draw_canvas[tuple(map(lambda x: np.array(x, dtype=np.uint32), new_seg["erase_points"]))] = 2
        if save_draw:
            segment.draw_canvas[save_draw_bck > 0] = save_draw_bck[save_draw_bck > 0]
        for fun in self.threshold_change_callback:
            fun()
        for fun in self.callback_colormap:
            fun()
        self.advanced_settings_changed()

    def change_colormap(self, new_color_map=None):
        """
        :type new_color_map: str | none
        :param new_color_map: name of new colormap
        :return:
        """
        if new_color_map is not None:
            self.color_map_name = new_color_map
            self.color_map = matplotlib.cm.get_cmap(new_color_map)
        for fun in self.callback_colormap:
            fun()

    def add_colormap_list_callback(self, callback):
        self.callback_colormap_list.append(callback)

    def add_colormap_callback(self, callback):
        self.callback_colormap.append(callback)

    def remove_colormap_callback(self, callback):
        try:
            self.callback_colormap.remove(callback)
        except ValueError:
            pass

    def add_profile(self, profile):
        """
        :type profile: SegmentationProfile
        :return:
        """
        # if not overwrite and name in self.profiles:
        #    raise ValueError("Profile with this name already exists")
        self.segmentation_profiles_dict[profile.name] = profile
        for fun in self.profiles_list_changed_callback:
            fun()

    def delete_profile(self, name):
        name = str(name)
        del self.segmentation_profiles_dict[name]

    def get_profile(self, name):
        name = str(name)
        return self.segmentation_profiles_dict[name]

    @property
    def colormap_list(self):
        return self.chosen_colormap

    @property
    def available_colormap_list(self):
        return pyplot.colormaps()

    def add_image(self, image, file_path, mask=None, new_image=True, original_image=None):
        self.image = image
        self.min_value = image.min()
        self.max_value = image.max()
        if original_image is None:
            self.original_image = image
        else:
            self.original_image = original_image
        self.gauss_image = gaussian(self.image, self.gauss_radius)
        self.mask = mask
        self.file_path = file_path
        if new_image:
            self.threshold_list = []
            self.threshold_layer_separate = False
            self.prev_segmentation_settings = []
            self.next_segmentation_settings = []
            self.image_clean_profile = None
        self.image_changed_fun()

    def image_changed_fun(self):
        for fun in self.image_change_callback:
            if isinstance(fun, tuple) and fun[1] == str:
                fun[0](self.image, self.file_path)
                continue
            elif isinstance(fun, tuple) and fun[1] == GAUSS:
                fun[0](self.image, self.gauss_image)
                continue
            fun(self.image)

    def changed_gauss_radius(self):
        self.gauss_image = gaussian(self.image, self.gauss_radius)
        for fun in self.image_change_callback:
            if isinstance(fun, tuple) and fun[1] == GAUSS:
                fun[0](self.image, self.gauss_image, True)
                continue
            elif isinstance(fun, tuple) and fun[1] == str:
                continue
            fun(self.image)

    def add_image_callback(self, callback):
        self.image_change_callback.append(callback)

    def change_threshold(self, new_threshold):
        if self.protect:
            return
        if self.threshold_layer_separate:
            if self.threshold_list[self.layer_num] == new_threshold:
                return
            self.threshold_list[self.layer_num] = new_threshold
        else:
            if self.threshold == new_threshold:
                return
            self.threshold = new_threshold
        for fun in self.threshold_change_callback:
            fun()

    def add_change_layer_callback(self, callback):
        self.callback_change_layer.append(callback)

    def change_layer(self, val):
        self.layer_num = val
        self.protect = True
        if self.threshold_layer_separate:
            for fun in self.callback_change_layer:
                fun(self.threshold_list[val])
        else:
            for fun in self.callback_change_layer:
                fun(self.threshold)
        # for fun in self.threshold_change_callback:
        #     fun()
        self.protect = False

    def change_threshold_type(self, new_type):
        print(new_type)
        if new_type == "Upper threshold:":
            self.threshold_type = UPPER
        else:
            self.threshold_type = "Lower"
        for fun in self.threshold_change_callback:
            fun()
        for fun in self.threshold_type_change_callback:
            fun()

    def change_layer_threshold(self, layer_threshold):
        self.threshold_layer_separate = layer_threshold
        if layer_threshold and self.threshold_list == []:
            self.threshold_list = [self.threshold] * self.image.shape[0]
        for fun in self.threshold_change_callback:
            fun()

    def change_gauss(self, use_gauss):
        self.use_gauss = bool(use_gauss)
        for fun in self.threshold_change_callback:
            fun()

    def add_threshold_callback(self, callback):
        self.threshold_change_callback.append(callback)

    def add_threshold_type_callback(self, callback):
        self.threshold_type_change_callback.append(callback)

    def change_min_size(self, new_min_size):
        self.minimum_size = new_min_size
        for fun in self.minimum_size_change_callback:
            fun()

    def add_min_size_callback(self, callback):
        self.minimum_size_change_callback.append(callback)

    def get_profile_list(self):
        return self.segmentation_profiles_dict.keys()

    def set_available_colormap(self, cmap_list):
        self.chosen_colormap = cmap_list
        for fun in self.callback_colormap_list:
            fun()

    def change_draw_use(self, use_draw):
        self.use_draw_result = use_draw
        for fun in self.draw_callback:
            fun()

    def add_draw_callback(self, callback):
        self.draw_callback.append(callback)

    def add_metadata_changed_callback(self, callback):
        self.metadata_changed_callback.append(callback)

    def advanced_settings_changed(self):
        for fun in self.threshold_type_change_callback:
            fun()
        for fun in self.metadata_changed_callback:
            fun()

    def metadata_changed(self):
        for fun in self.metadata_changed_callback:
            fun()


def calculate_statistic_from_image(img, mask, settings):
    """
    :type img: np.ndarray
    :type mask: np.ndarray
    :type settings: Settings
    :return: dict[str,object]
    """
    def pixel_volume(x):
        return x[0] * x[1] * x[2]
    res = dict()
    voxel_size = settings.voxel_size
    res["Volume"] = np.count_nonzero(mask) * pixel_volume(settings.voxel_size)
    res["Mass"] = np.sum(img)
    # res["Border Volume"] = np.count_nonzero(border_im)

    res["Border Surface"] = calculate_volume_surface(mask, voxel_size)
    # res["Border Surface Opening"] = calculate_volume_surface(opening_smooth(mask), voxel_size)
    # res["Border Surface Closing"] = calculate_volume_surface(closing_smooth(mask), voxel_size)
    img_mask = mask > 0  # img > 0
    res["Pixel max"] = np.max(img)
    if np.any(img_mask):
        res["Pixel min"] = np.min(img[img_mask])
        res["Pixel mean"] = np.mean(img[img_mask])
        res["Pixel median"] = np.median(img[img_mask])
        res["Pixel std"] = np.std(img[img_mask])
    else:
        res["Pixel min"] = 0
        res["Pixel mean"] = 0
        res["Pixel median"] = 0
        res["Pixel std"] = 0
    try:
        res["Mass to Volume"] = res["Mass"] / res["Volume"]
        res["Volume to Border Surface"] = res["Volume"] / res["Border Surface"]
        # res["Volume to Border Surface Opening"] = res["Volume"] / res["Border Surface Opening"]
        # res["Volume to Border Surface Closing"] = res["Volume"] / res["Border Surface Closing"]
    except ZeroDivisionError:
        pass
    if len(img.shape) == 3:
        res["Moment of inertia"] = af.calculate_density_momentum(img, voxel_size)
    return res


def get_segmented_data(image, settings, segment, with_std=False, mask_morph=None, div_scale=1):
    """
    :param image: image from witch data should be catted
    :param settings: program settings
    :param segment: segment backend for program
    :param with_std: bool switch that enable returning information about noise outside segmented data
    :param mask_morph: function that do morphological operation on mask
    :type image: np.ndarray
    :type settings: Settings
    :type segment: Segment
    :type with_std: bool
    :type mask_morph: None | (np.ndarray) -> np.ndarray
    :type div_scale: float
    :return:
    """
    segmentation = segment.get_segmentation()
    if mask_morph is not None:
        logging.debug("No None morph_fun")
        segmentation = mask_morph(segmentation)
    else:
        logging.debug("None morph_fun")
    full_segmentation = segment.get_full_segmentation()
    image = image.astype(np.float)
    if settings.threshold_type == UPPER:
        noise_mean = np.mean(image[full_segmentation == 0])
        image = noise_mean - image
    image /= div_scale
    noise_std = np.std(image[full_segmentation == 0])
    image[segmentation == 0] = 0  # min(image[segmentation > 0].min(), 0)
    image[image < 0] = 0
    if with_std:
        return image, segmentation, noise_std
    return image, segmentation
