from __future__ import print_function, division
import matplotlib
from matplotlib import pyplot
import numpy as np
import SimpleITK as sitk
import json
import auto_fit as af
import logging
from enum import Enum
from collections import OrderedDict, namedtuple
UPPER = "Upper"
GAUSS = "Gauss"

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


class MaskChange(Enum):
    prev_seg = 1
    next_seg = 2


class DrawType(Enum):
    draw = 1
    erase = 2
    force_show = 3
    force_hide = 4


def class_to_dict(obj, *args):
    """
    Create dict which contains values of given fields
    :type obj: object
    :type args: list[str]
    :return:
    """
    res = dict()
    for name in args:
        res[name] = getattr(obj, name)
    return res


def dict_set_class(obj, dic, *args):
    """
    Set fields of given object based on values from dict.
    If *args contains no names all values from dict are used
    :type obj: object
    :type dic: dict[str,object]
    :param args: list[str]
    :return:
    """
    if len(args) == 0:
        li = dic.keys()
    else:
        li = args
    for name in li:
        try:
            getattr(obj, name)
            setattr(obj, name, dic[name])
        except AttributeError as ae:
            logging.warning(ae)


def gaussian(image, radius):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(layer), radius))
    return res


def dilate(image, radius):
    """
    :param image: image to apply gaussian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.GrayscaleDilate(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.GrayscaleDilate(sitk.GetImageFromArray(layer), radius))
    return res


def to_binary_image(image):
    return np.array(image > 0).astype(np.uint8)


def bisect(arr, val, comp):
    l = -1
    r = len(arr)
    while r - l > 1:
        e = (l + r) >> 1
        if comp(arr[e], val):
            l = e
        else:
            r = e
    return r

SettingsValue = namedtuple("SettingsValue", ["function_name", "help_message", "arguments"])
Leaf = namedtuple("Leaf", ["name", "dict"])
Node = namedtuple("Node", ["left", 'op', 'right'])


class StatisticProfile(object):

    STATISTIC_DICT = {
        "Volume": SettingsValue("calculate_volume", "Calculate volume of current segmentation", None),
        "Volume per component": SettingsValue("calculate_component_volume", "Calculate volume of each component "
                                              "of cohesion of current segmentation", None),
        "Mass": SettingsValue("calculate_mass", "Sum of pixel brightness for current segmentation", None),
        "Mass per component": SettingsValue("calculate_component_mass", "Sum of pixel brightness of each component of"
                                            " cohesion for current segmentation", None),
        "Border surface": SettingsValue("calculate_border_surface",
                                        "Calculating surface of current segmentation", None),
        "Maximum pixel brightness": SettingsValue(
            "maximum_brightness", "Calculate maximum brightness of pixel for current segmentation", None),
        "Minimum pixel brightness": SettingsValue(
            "minimum_brightness", "Calculate minimum brightness of pixel for current segmentation", None),
        "Median pixel brightness": SettingsValue(
            "median_brightness", "Calculate median brightness of pixel for current segmentation", None),
        "Mean pixel brightness": SettingsValue(
            "mean_brightness", "Calculate median brightness of pixel for current segmentation", None),
        "Standard deviation of pixel brightness": SettingsValue(
            "std_brightness", "Calculate  standard deviation of pixel brightness for current segmentation", None),
        "Standard deviation of Noise": SettingsValue(
            "std_noise", "Calculate standard deviation of pixel brightness outside current segmentation", None),
        "Moment of inertia": SettingsValue("moment_of_inertia", "Calculate moment of inertia for segmented structure."
                                           "Has one parameter thr (threshold). Only values above it are used "
                                           "in calculation", None),
        "Border Mass": SettingsValue("border_mass", "Calculate mass for elements in radius (in physical units)"
                                                    " from mask", {"radius": int}),
        "Border Volume": SettingsValue("border_volume", "Calculate volumes for elements in radius (in physical units)"
                                                        " from mask", {"radius": int})
    }
    PARAMETERS = ["name", "chosen_fields", "reversed_brightness", "use_gauss_image"]

    def __init__(self, name, chosen_fields, reversed_brightness, settings=None, use_gauss_image=False):
        self.name = name
        self.chosen_fields = []
        for cf_val in chosen_fields:
            user_name = cf_val[1]
            if isinstance(cf_val[0], str):
                tree = self.parse_statistic(cf_val[0])
            else:
                tree = self.rebuild_tree(cf_val[0])
            self.chosen_fields.append((tree, user_name, None))
        self.settings = settings
        self.reversed_brightness = reversed_brightness
        self.use_gauss_image = use_gauss_image
        self.name_prefix = ""

    def __str__(self):
        text = "Profile name: {}\n".format(self.name)
        text += "Reversed image [{}]\n".format(self.reversed_brightness)
        text += "Gaussed image [{}]\n".format(self.use_gauss_image)
        if self.name_prefix != "":
            text += "Name prefix: {}".format(self.name_prefix)
        text += "statistics list:\n"
        for el in self.chosen_fields:
            if el[2] is not None:
                text += "{}: {}\n".format(el[1], el[2])
            else:
                text += "{}\n".format(el[1])
        return text

    def get_parameters(self):
        return class_to_dict(self, *self.PARAMETERS)

    def rebuild_tree(self, l):
        if len(l) == 2:
            return Leaf(*l)
        else:
            return Node(self.rebuild_tree(l[0]), l[1], self.rebuild_tree(l[2]))

    def flat_tree(self, t):
        if isinstance(t, Leaf):
            res = ""
            if t.dict is not None and len(t.dict) > 0:
                for name, val in t.dict.items():
                    res += "{}={},".format(name, val)
                return "{}[{}]".format(t.name, res[:-1])
            return t.name
        elif isinstance(t, Node):
            if isinstance(t.left, Node):
                beg = "({})"
            else:
                beg = "{}"
            if isinstance(t.right, Node):
                end = "({})"
            else:
                end = "{}"
            return (beg+"{}"+end).format(self.flat_tree(t.left), t.op, self.flat_tree(t.right))

    @staticmethod
    def tokenize(text):
        special = ["(", ")", "[", "]", "/", "+", ","]
        res = []
        temp_str = ""
        for l in text:
            if l in special:
                if temp_str != "":
                    res.append(temp_str)
                    temp_str = ""
                res.append(l)
            else:
                temp_str += l
        if temp_str != "":
            res.append(temp_str)
        return res

    def build_tree(self, tokens):
        res = []
        final_res = res
        pos = 0
        while True:
            if pos == len(tokens):
                break
            if tokens[pos] == ")":
                pos += 1
                break
            if tokens[pos] == "/":
                final_res = [res[:], "/"]
                res = []
                final_res.append(res)
                pos += 1
            if tokens[pos] in "[],":
                pos += 1
                continue
            if tokens[pos] == "(":
                sub_tree, pos_shift = self.build_tree(tokens[pos+1:])
                pos += pos_shift+1
                res.extend(sub_tree)
                continue
            res.append(tokens[pos])
            pos += 1
        return final_res, pos

    def tree_to_dict_tree(self, tree):
        if isinstance(tree[0], list):
            left_tree = self.tree_to_dict_tree(tree[0])
            right_tree = self.tree_to_dict_tree(tree[2])
            return Node(left_tree, tree[1], right_tree)
        else:
            name = tree[0]
            base_stat = self.STATISTIC_DICT[name]
            d = dict()
            for el in tree[1:]:
                sp = el.split("=")
                d[sp[0]] = base_stat.arguments[sp[0]](sp[1])
            return Leaf(name, d)

    def parse_statistic(self, text):
        tokens = self.tokenize(text)

        tree, l = self.build_tree(tokens)
        return self.tree_to_dict_tree(tree)

    def calculate_tree(self, node, help_dict, kwargs):
        """
        :type node: Leaf | Node
        :type help_dict: dict
        :type kwargs: dict
        :return: float
        """
        if isinstance(node, Leaf):
            fun_name = self.STATISTIC_DICT[node.name][0]
            kw = dict(kwargs)
            kw.update(node.dict)
            hash_str = "{}: {}".format(fun_name, kw)
            if hash_str in help_dict:
                return help_dict[hash_str]
            fun = getattr(self, fun_name)
            val = fun(**kw)
            help_dict[hash_str] = val
            return val
        elif isinstance(node, Node):
            left_res = self.calculate_tree(node.left, help_dict, kwargs)
            right_res = self.calculate_tree(node.right, help_dict, kwargs)
            if node.op == "/":
                return left_res/right_res
        logging.error("Wrong statistics: {}".format(node))
        return 1

    def calculate(self, image, gauss_image, mask, full_mask, base_mask):
        result = OrderedDict()
        if self.use_gauss_image:
            image = gauss_image.astype(np.float)
        else:
            image = image.astype(np.float)
        if self.reversed_brightness:
            noise_mean = np.mean(image[full_mask == 0])
            image = noise_mean - image
        help_dict = dict()
        kw = {"image": image, "mask": mask, "base_mask": base_mask, "full_mask": full_mask}
        for tree, user_name, params in self.chosen_fields:
            try:
                result[user_name] = self.calculate_tree(tree, help_dict, kw)
            except ZeroDivisionError:
                result[user_name] = "Div by zero"
            except TypeError:
                result[user_name] = "None div"
        return result

    @staticmethod
    def pixel_volume(x):
        return x[0] * x[1] * x[2]

    def calculate_volume(self, mask, **_):
        print("Volume {}".format(np.max(mask)))
        return np.count_nonzero(mask) * self.pixel_volume(self.settings.voxel_size)

    def calculate_component_volume(self, mask, **_):
        return np.bincount(mask.flat)[1:] * self.pixel_volume(self.settings.voxel_size)

    @staticmethod
    def calculate_mass(mask, image, **_):
        if np.any(mask):
            return np.sum(image[mask > 0])
        return 0

    @staticmethod
    def calculate_component_mass(mask, image, **_):
        res = []
        for i in range(1, mask.max()+1):
            res.append(np.sum(image[mask == i]))
        return res

    def calculate_border_surface(self, mask, **_):
        return calculate_volume_surface(mask, self.settings.voxel_size)

    @staticmethod
    def maximum_brightness(mask, image, **_):
        if np.any(mask):
            return np.max(image[mask > 0])
        else:
            return None

    @staticmethod
    def minimum_brightness(mask, image, **_):
        if np.any(mask):
            return np.min(image[mask > 0])
        else:
            return None

    @staticmethod
    def median_brightness(mask, image, **_):
        if np.any(mask):
            return np.median(image[mask > 0])
        else:
            return None

    @staticmethod
    def std_brightness(mask, image, **_):
        if np.any(mask):
            return np.std(image[mask > 0])
        else:
            return None

    @staticmethod
    def mean_brightness(mask, image, **_):
        if np.any(mask):
            return np.mean(image[mask > 0])
        else:
            return None

    @staticmethod
    def std_noise(mask, base_mask, image, **_):
        if np.any(mask):
            if base_mask is not None:
                return np.std(image[(mask == 0) * (base_mask > 0)])
            else:
                return np.std(image[mask == 0])
        else:
            return None

    def moment_of_inertia(self, image, mask, **_):
        if image.ndim != 3:
            return None
        img = np.copy(image)
        img[mask == 0] = 0
        return af.calculate_density_momentum(img, self.settings.voxel_size,)

    def border_mask(self, base_mask, radius, **_):
        if base_mask is None:
            return None
        base_mask = np.array(base_mask > 0)
        base_mask = base_mask.astype(np.uint8)
        border = sitk.LabelContour(sitk.GetImageFromArray(base_mask))
        border.SetSpacing(self.settings.voxel_size)
        dilated_border = sitk.GetArrayFromImage(sitk.BinaryDilate(border, radius))
        dilated_border[base_mask == 0] = 0
        return dilated_border

    def border_mass(self, image, mask, **kwargs):
        border_mask = self.border_mask(**kwargs)
        if border_mask is None:
            return None
        final_mask = np.array((border_mask > 0) * (mask > 0))
        if np.any(final_mask):
            return np.sum(image[final_mask])
        return 0

    def border_volume(self, mask, **kwargs):
        border_mask = self.border_mask(**kwargs)
        if border_mask is None:
            return None
        final_mask = np.array((border_mask > 0) * (mask > 0))
        return np.count_nonzero(final_mask) * self.pixel_volume(self.settings.voxel_size)


class SegmentationProfile(object):
    PARAMETERS = ("name", "threshold", "threshold_list", "threshold_type", "minimum_size", "use_gauss", "gauss_radius",
                  "threshold_layer_separate")
    SEGMENTATION_PARAMETERS = ("threshold", "threshold_list", "threshold_type", "minimum_size", "use_gauss",
                               "gauss_radius", "threshold_layer_separate")

    def __init__(self, name, threshold, threshold_list, threshold_type, minimum_size, use_gauss, gauss_radius,
                 threshold_layer_separate):
        """
        :param name: str,
        :param threshold: int
        :param threshold_list: list[int]
        :param threshold_type: str
        :param minimum_size: int
        :param use_gauss: bool
        """
        self.name = name
        if threshold_layer_separate:
            self.threshold = np.median(threshold_list)
            self.threshold_list = threshold_list
        else:
            self.threshold = threshold
            self.threshold_list = []
        self.threshold_type = threshold_type
        self.minimum_size = minimum_size
        self.use_gauss = use_gauss
        self.gauss_radius = gauss_radius
        self.threshold_layer_separate = threshold_layer_separate

    def __str__(self):
        if self.name != "":
            text = "Name: {}\n".format(self.name)
        else:
            text = ""
        text += "{} threshold: ".format(self.threshold_type)
        if self.threshold_layer_separate:
            text += str(self.threshold_list)
        else:
            text += str(self.threshold)
            text += "\n"
        text += "Minimum object size: {}\n".format(self.minimum_size)
        text += "Use gauss [{}]\n".format("x" if self.use_gauss else " ")
        if self.use_gauss:
            text += "Gauss radius: {}".format(self.gauss_radius)
        return text

    def get_parameters(self):
        return class_to_dict(self, *self.PARAMETERS)


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


class Segment(object):
    """
    :type _segmented_image: np.ndarray
    :type segmentation_change_callback: list[() -> None | (list[int] -> None]
    """

    def __init__(self, settings):
        """
        :type settings: Settings
        """
        self._settings = settings
        self._image = None
        self.draw_canvas = None
        self.draw_counter = 0
        self.gauss_image = None
        self._threshold_image = None
        self._segmented_image = None
        self._finally_segment = None
        self._sizes_array = []
        self.segmentation_change_callback = []
        self._segmentation_changed = True
        self.protect = False
        self._settings.add_threshold_callback(self.threshold_updated)
        self._settings.add_min_size_callback(self.min_size_updated)
        self._settings.add_draw_callback(self.draw_update)

    def set_image(self, image):
        self._image = image
        self.gauss_image = gaussian(self._image, self._settings.gauss_radius)
        self._segmentation_changed = True
        self._finally_segment = np.zeros(image.shape, dtype=np.uint8)
        self.threshold_updated()

    def threshold_updated(self):
        if self.protect:
            return
        if self._image is None:
            return
        self._threshold_image = np.zeros(self._image.shape, dtype=np.uint8)
        if self._settings.use_gauss:
            image_to_threshold = self.gauss_image
        else:
            image_to_threshold = self._image
        # Define which threshold use
        if self._settings.threshold_type == UPPER:
            def get_mask(image, threshold):
                return image <= threshold
        else:
            def get_mask(image, threshold):
                return image >= threshold

        if self._settings.threshold_layer_separate:
            # print("Layer separate")
            for i in range(self._image.shape[0]):
                self._threshold_image[i][get_mask(image_to_threshold[i], self._settings.threshold_list[i])] = 1
        else:
            # print("normal")
            self._threshold_image[get_mask(image_to_threshold, self._settings.threshold)] = 1
        if self._settings.mask is not None:
            self._threshold_image *= (self._settings.mask > 0)
        self.draw_update()

    def draw_update(self, canvas=None):
        if self.protect:
            return
        if canvas is not None:
            self.draw_canvas[...] = canvas[...]
            return
        if self._settings.use_draw_result:
            threshold_image = np.copy(self._threshold_image)
            threshold_image[self.draw_canvas == 1] = 1
            threshold_image[self.draw_canvas == 2] = 0
        else:
            threshold_image = self._threshold_image
        connect = sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_image))
        self._segmented_image = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
        self._sizes_array = np.bincount(self._segmented_image.flat)
        self.min_size_updated()

    def min_size_updated(self):
        if self.protect:
            return
        ind = bisect(self._sizes_array[1:], self._settings.minimum_size, lambda x, y: x > y)
        # print(ind, self._sizes_array, self._settings.minimum_size)
        hide_set = set(np.unique(self._segmented_image[self.draw_canvas == DrawType.force_hide.value]))
        show_set = set(np.unique(self._segmented_image[self.draw_canvas == DrawType.force_show.value]))
        hide_set -= show_set
        show_set.discard(0)
        hide_set.discard(0)
        finally_segment = np.copy(self._segmented_image)
        finally_segment[finally_segment > ind] = 0
        for val in show_set:
            finally_segment[self._segmented_image == val] = val
        for val in hide_set:
            finally_segment[self._segmented_image == val] = 0
        if len(show_set) > 0 or len(hide_set) > 0:
            self._finally_segment = np.zeros(finally_segment.shape, dtype=finally_segment.dtype)
            for i, val in enumerate(np.unique(finally_segment)[1:], 1):
                self._finally_segment[finally_segment == val] = i
        else:
            self._finally_segment = finally_segment

        self._segmentation_changed = True
        for fun in self.segmentation_change_callback:
            if isinstance(fun, tuple):
                fun[0](self._sizes_array[1:ind+1])
                continue
            if callable(fun):
                fun()

    @property
    def segmentation_changed(self):
        """:rtype: bool"""
        return self._segmentation_changed

    def get_segmentation(self):
        self._segmentation_changed = False
        return self._finally_segment

    def get_size_array(self):
        return self._sizes_array

    def get_full_segmentation(self):
        return self._segmented_image

    def add_segmentation_callback(self, callback):
        self.segmentation_change_callback.append(callback)


def calculate_volume_surface(volume_mask, voxel_size):
    border_surface = 0
    surf_im = np.array(volume_mask).astype(np.uint8)
    border_surface += np.count_nonzero(np.logical_xor(surf_im[1:], surf_im[:-1])) * voxel_size[1] * voxel_size[2]
    border_surface += np.count_nonzero(np.logical_xor(surf_im[:, 1:], surf_im[:, :-1])) * voxel_size[0] * voxel_size[2]
    if len(surf_im.shape) == 3:
        border_surface += np.count_nonzero(np.logical_xor(surf_im[:, :, 1:], surf_im[:, :, :-1])) * voxel_size[0] * \
                          voxel_size[1]
    return border_surface


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
    # border_im = sitk.GetArrayFromImage(sitk.LabelContour(sitk.GetImageFromArray((mask > 0).astype(np.uint8))))
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


MaskCreate = namedtuple("MaskCreate", ['name'])
MaskUse = namedtuple("MaskUse", ['name'])
CmapProfile = namedtuple("CmapProfile", ["suffix", "gauss_type", "center_data", "rotation_axis", "cut_obsolete_are"])


class Operations(Enum):
    clean_mask = 1


class CalculationPlan(object):
    correct_name = {MaskCreate.__name__ : MaskCreate, MaskUse.__name__: MaskUse, CmapProfile.__name__: CmapProfile,
                    StatisticProfile.__name__: StatisticProfile, SegmentationProfile.__name__: SegmentationProfile}

    # TODO Batch plans dump and load
    def __init__(self):
        self.execution_list = []
        self.execution_tree = None
        self.segmentation_count = 0
        self.name = ""

    def clean(self):
        self.execution_list = []

    def add_step(self, step):
        text = self.get_el_name(step)
        self.execution_list.append(step)
        if isinstance(step, SegmentationProfile):
            self.segmentation_count += 1
        print(self.dict_dump())
        return text

    def __len__(self):
        return len(self.execution_list)

    def pop(self):
        el = self.execution_list.pop()
        if isinstance(el, SegmentationProfile):
            self.segmentation_count -= 1

    def is_segmentation(self):
        return self.segmentation_count > 0

    def build_execution_tree(self):
        pass

    def set_name(self, text):
        self.name = text

    def get_parameters(self):
        return self.dict_dump()

    def dict_dump(self):
        res = dict()
        res["name"] = self.name
        execution_list = []
        for el in self.execution_list:
            sub_dict = dict()
            sub_dict["type"] = el.__class__.__name__
            if issubclass(el.__class__, tuple):
                sub_dict["values"] = el.__dict__
            elif isinstance(el, StatisticProfile):
                sub_dict["values"] = el.get_parameters()
            elif isinstance(el, SegmentationProfile):
                sub_dict["values"] = el.get_parameters()
            else:
                raise ValueError("Not supported type")
            execution_list.append(sub_dict)
        res["execution_list"] = execution_list
        return res

    @staticmethod
    def dict_load(data_dict):
        res_plan = CalculationPlan()
        name = data_dict["name"]
        res_plan.set_name(name)
        execution_list = data_dict["execution_list"]
        for el in execution_list:
            res_plan.add_step(CalculationPlan.correct_name[el["type"]](**el["values"]))
        res_plan.build_execution_tree()
        return res_plan

    @staticmethod
    def get_el_name(el):
        """
        :param el: Plan element
        :return: str
        """
        if el.__class__.__name__ not in CalculationPlan.correct_name.keys():
            raise ValueError("Unknown type")
        if isinstance(el, Operations):
            if el == Operations.clean_mask:
                return "Clean mask"
        if isinstance(el, SegmentationProfile):
            return "Segmentation: {}".format(el.name)
        if isinstance(el, StatisticProfile):
            if el.name_prefix == "":
                return "Statistics: {}".format(el.name)
            else:
                "Statistics: {} with prefix: {}".format(el.name, el.name_prefix)
        if isinstance(el, MaskCreate):
            return "Create mask: {}".format(el.name)
        if isinstance(el, MaskUse):
            return "Use mask: {}".format(el.name)
        if isinstance(el, CmapProfile):
            if el.suffix == "":
                return "Camp save"
            else:
                return "Cmap save with suffix: {}".format(el.suffix)

        raise ValueError("Unknown type")