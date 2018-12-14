# coding=utf-8
from __future__ import print_function, division

import json
import logging
from enum import Enum
from functools import reduce

import matplotlib
import numpy as np
from matplotlib import pyplot

from partseg_old.batch_processing.calculation_plan import CalculationPlan

GAUSS = "Gauss"

from partseg_utils import class_to_dict, dict_set_class
from partseg_old.segment import SegmentationProfile, Segment, UPPER, fill_holes_in_mask, fill_2d_holes_in_mask
from partseg_old.statistics_calculation import StatisticProfile, calculate_volume_surface
from partseg_utils.image_operations import gaussian, dilate, erode
from scipy.ndimage.interpolation import zoom
from partseg_utils.autofit import calculate_density_momentum


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
        self.use_gauss = True
        self.use_draw_result = False
        self.draw_callback = []
        self.threshold = 37000
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
        self.voxel_size = [1, 1, 1]
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
        self.leave_biggest = False
        self.normalize_range = (0,2**16-1,False)
        try:
            self.load(settings_path)
        except ValueError as e:
            logging.error("Saved profile problem: {}".format(e))

    @property
    def spacing(self):
        return self.voxel_size[:self.image.ndim]

    @spacing.setter
    def spacing(self, value):
        self.voxel_size = value

    def rescale_image(self, scale_factor):
        sx, sy, sz = self.spacing
        if len(scale_factor) == 2:
            self.spacing = sx / scale_factor[1], sy / scale_factor[0], 1
        else:
            self.spacing = sx / scale_factor[2], sy / scale_factor[1], sz / scale_factor[0]

        scale_min_size = reduce(lambda x,y: x*y, scale_factor)
        new_image = zoom(self.image, scale_factor)
        new_mask = None
        new_threshold = None
        if len(self.threshold_list) != 0:
            new_threshold = zoom(np.array(self.threshold_list), scale_factor[0])
        if self.mask is not None:
            new_mask = zoom(self.image, scale_factor)
        if np.all(new_image[0] == 0):
            new_image = new_image[1:]
            if new_mask is not None:
                new_mask = new_mask[1:]
            if new_threshold is not None:
                new_threshold = new_threshold[1:]
        if np.all(new_image[-1] == 0):
            new_image = new_image[:-1]
            if new_mask is not None:
                new_mask = new_mask[:-1]
            if new_threshold is not None:
                new_threshold = new_threshold[:-1]
        self.minimum_size = self.minimum_size * scale_min_size
        self.metadata_changed()
        self.add_image(new_image, self.file_path, new_mask, threshold_list=new_threshold)


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

    def dump_profiles(self, file_path, export_names):
        export_names = set(export_names)
        profiles_list = [x.get_parameters() for n, x in self.segmentation_profiles_dict.items() if n in export_names]
        with open(file_path, "w") as ff:
            json.dump(profiles_list, ff)

    @staticmethod
    def load_profiles(file_path):
        res = dict()
        with open(file_path, "r") as ff:
            profiles_list = json.load(ff)
        for prof in profiles_list:
            res[prof["name"]] = SegmentationProfile(**prof)
        return res

    def add_profiles(self, profile_dict, import_names):
        for name, new_name in import_names:
            prof = profile_dict[name]
            prof.name = new_name
            self.segmentation_profiles_dict[new_name] = prof
        for fun in self.threshold_change_callback:
            fun()

    def dump_statistics(self, file_path, export_names):
        export_names = set(export_names)
        res = [x.get_parameters()
               for n, x in self.statistics_profile_dict.items() if n in export_names]

        json_str = json.dumps(res)
        with open(file_path, 'w') as ff:
            ff.write(json_str)

    @staticmethod
    def load_statistics(file_path):
        res = dict()
        with open(file_path, 'r') as ff:
            statistics_list = json.load(ff)
        for stat in statistics_list:
            res[stat["name"]] = StatisticProfile(**stat)
        return res

    def add_statistics(self, statistics, import_names):
        for name, new_name in import_names:
            stat = statistics[name]
            stat.name = new_name
            self.statistics_profile_dict[new_name] = stat

    def dump_calculation_plans(self, file_path, export_names):
        export_names = set(export_names)
        json_str = json.dumps([x.get_parameters() for n, x in self.batch_plans.items() if n in export_names])
        with open(file_path, "w") as ff:
            ff.write(json_str)

    @staticmethod
    def load_calculation_plans(file_path):
        res = dict()
        with open(file_path, "r") as ff:
            calculation_plans = json.load(ff)
        for plan in calculation_plans:
            calc_plan = CalculationPlan.dict_load(plan)
            res[calc_plan.name] = calc_plan
        return res

    def add_calculation_plans(self, calculation_plans, import_names):
        for name, new_name in import_names:
            calc_plan = calculation_plans[name]
            calc_plan.name = new_name
            self.batch_plans[calc_plan.name] = calc_plan

    def dump(self, file_path):
        important_data = \
            class_to_dict(self, "open_directory", "open_filter", "save_directory", "save_filter",
                          "voxel_size", "size_unit", "threshold", "threshold_type", "color_map_name", "overlay", "minimum_size",
                          "gauss_radius", "export_filter", "export_directory", "scale_factor", "statistic_dirs",
                          "chosen_colormap", "batch_directory", "use_gauss")
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
            try:
                dict_set_class(self, important_data, "open_directory", "open_filter", "save_directory", "save_filter",
                               "voxel_size", "size_unit", "threshold", "threshold_type", "color_map_name", "overlay",
                               "minimum_size", "gauss_radius", "export_filter", "export_directory", "scale_factor",
                               "statistic_dirs", "chosen_colormap", "batch_directory", "use_gauss")
            except Exception as e:
                print(e)
                pass
            # TODO Batch plans load
            chosen_colormap = set(self.chosen_colormap)
            avail_colormap = set(pyplot.colormaps())
            self.chosen_colormap = list(sorted(chosen_colormap & avail_colormap))
            self.color_map = matplotlib.cm.get_cmap(self.color_map_name)
            for prof in important_data["profiles"]:
                self.segmentation_profiles_dict[prof["name"]] = SegmentationProfile(**prof)
            for stat in important_data["statistics"]:
                self.statistics_profile_dict[stat["name"]] = StatisticProfile(**stat)
            for plan in important_data["batch_plans"]:
                calc_plan = CalculationPlan.dict_load(plan)
                self.batch_plans[calc_plan.name] = calc_plan
        except IOError:
            logging.warning("No configuration file")
            pass
        except KeyError as e:
            logging.warning("Bad configuration: {}".format(e))

    def change_segmentation_mask(self, segment, order, save_draw, fill_holes=False, fill_2d_holes=False):
        """
        :type segment: Segment
        :type order: MaskChange
        :type save_draw: bool
        :type fill_holes: bool
        :type fill_2d_holes: bool
        :return:
        """
        save_fields = ["threshold", "threshold_list", "threshold_type", "threshold_layer_separate",
                       "minimum_size", "use_gauss", "use_draw_result", "mask_dilate_radius", "mask", "gauss_radius"]
        if order == MaskChange.prev_seg and len(self.prev_segmentation_settings) == 0:
            return

        current_mask = segment.get_segmentation()
        current_mask = np.array(current_mask > 0)
        seg_settings = class_to_dict(self, *save_fields)
        if segment.draw_canvas is not None:
            seg_settings["draw_points"] = tuple(map(list, np.nonzero(np.array(segment.draw_canvas == 1))))
            seg_settings["erase_points"] = tuple(map(list, np.nonzero(np.array(segment.draw_canvas == 2))))
        save_draw_bck = np.copy(segment.draw_canvas)
        if order == MaskChange.next_seg:
            if fill_2d_holes:
                current_mask = fill_2d_holes_in_mask(current_mask)
            elif fill_holes:
                current_mask = fill_holes_in_mask(current_mask)
            self.prev_segmentation_settings.append(seg_settings)
            if self.mask_dilate_radius > 0:
                self.mask = dilate(current_mask, self.mask_dilate_radius)
            elif self.mask_dilate_radius < 0:
                self.mask = erode(current_mask, -self.mask_dilate_radius)
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
        if segment.draw_canvas is not None:
            segment.draw_canvas[...] = 0
        if new_seg is not None:
            dict_set_class(self, new_seg, *save_fields)
            if segment.draw_canvas is not None:
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

    def scale_image(self, scale_factor):
        if self.image is None:
            return
        if len(self.image.shape) == 2:
            self.image = zoom(self.image, scale_factor)
        else:
            self.image = zoom(self.image, (1, scale_factor, scale_factor))
        if self.mask is not None:
            if len(self.image.shape) == 2:
                self.mask = zoom(self.mask, scale_factor)
            else:
                self.mask = zoom(self.mask, (1, scale_factor, scale_factor))

        self.min_value = np.min(self.image)
        self.max_value = np.max(self.image)
        self.gauss_image = gaussian(self.image, self.gauss_radius)
        self.image_changed_fun()

    def add_image(self, image, file_path, mask=None, new_image=True, original_image=None, threshold_list=None):
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
        if threshold_list is not None:
            self.threshold_list = threshold_list
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
        res["Moment of inertia"] = calculate_density_momentum(img, voxel_size)
        #res["Diameter"] = calc_diam(img_mask, voxel_size)
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
