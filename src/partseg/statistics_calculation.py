from __future__ import division

import logging
import traceback
from collections import namedtuple, OrderedDict
from functools import reduce
from typing import Dict

import SimpleITK as sitk
import numpy as np

from partseg_utils import class_to_dict, autofit as af

SettingsValue = namedtuple("SettingsValue", ["function_name", "help_message", "arguments", "is_mask", "is_component"])

Leaf = namedtuple("Leaf", ["name", "dict"])
Node = namedtuple("Node", ["left", 'op', 'right'])


class StatisticProfile(object):

    STATISTIC_DICT = {
        "Volume": SettingsValue("calculate_volume", "Calculate volume of current segmentation", None, False, False),
        "Volume per component": SettingsValue("calculate_component_volume", "Calculate volume of each component "
                                              "of cohesion of current segmentation", None, False, True),
        "Mass": SettingsValue("calculate_mass", "Sum of pixel brightness for current segmentation", None, False, False),
        "Mass per component": SettingsValue("calculate_component_mass", "Sum of pixel brightness of each component of"
                                            " cohesion for current segmentation", None, False, True),
        "Border surface": SettingsValue("calculate_border_surface",
                                        "Calculating surface of current segmentation", None, False, False),
        "Maximum pixel brightness": SettingsValue(
            "maximum_brightness", "Calculate maximum brightness of pixel for current segmentation", None, False, False),
        "Minimum pixel brightness": SettingsValue(
            "minimum_brightness", "Calculate minimum brightness of pixel for current segmentation", None, False, False),
        "Median pixel brightness": SettingsValue(
            "median_brightness", "Calculate median brightness of pixel for current segmentation", None, False, False),
        "Mean pixel brightness": SettingsValue(
            "mean_brightness", "Calculate median brightness of pixel for current segmentation", None, False, False),
        "Standard deviation of pixel brightness": SettingsValue(
            "std_brightness", "Calculate  standard deviation of pixel brightness for current segmentation", None,
            False, False),
        "Standard deviation of Noise": SettingsValue(
            "std_noise", "Calculate standard deviation of pixel brightness outside current segmentation", None,
            False, False),
        "Moment of inertia": SettingsValue("moment_of_inertia", "Calculate moment of inertia for segmented structure."
                                           "Has one parameter thr (threshold). Only values above it are used "
                                           "in calculation", None, False, False),
        "Border Mass": SettingsValue("border_mass", "Calculate mass for elements in radius (in physical units)"
                                                    " from mask", {"radius": float}, False, False),
        "Border Volume": SettingsValue("border_volume", "Calculate volumes for elements in radius (in physical units)"
                                                        " from mask", {"radius": float}, False, False),
        "Components Number": SettingsValue("number_of_components", "Calculate number of connected components "
                                                                   "on segmentation", None, False, False),
        "Mask Volume": SettingsValue("mask_volume", "Volume of mask", None, True, False),
        "Mask Diameter": SettingsValue("mask_diameter", "Diameter of mask", None, True, False),
        "Segmentation Diameter": SettingsValue("segmentation_diameter", "Diameter of segmentation", None, False, False),
        "Segmentation component Diameter":
            SettingsValue("component_diameter", "Diameter of each segmentation component", None, False, False),
        "Segmentation first main axis length":
            SettingsValue("calculate_first_main_axis_length",
                          "Length of first main axis of segmentation", None, False, False),
        "Segmentation second main axis length":
            SettingsValue("calculate_second_main_axis_length",
                          "Length of second main axis of segmentation", None, False, False),
        "Segmentation third main axis length":
            SettingsValue("calculate_third_main_axis_length",
                          "Length of third main axis of segmentation", None, False, False),
        "Compactness":
            SettingsValue("calculate_compactness",
                          "Calculate compactness off segmentation (Border surface^1.5/volume)", None, False, False),
        "Sphericity": SettingsValue("calculate_sphericity", "volume/(diameter**3/8)", None, False, False)

    }
    PARAMETERS = ["name", "chosen_fields", "reversed_brightness", "use_gauss_image", "name_prefix"]

    def __init__(self, name, chosen_fields, reversed_brightness, use_gauss_image=False, name_prefix=""):
        self.name = name
        self.chosen_fields = []
        self._need_mask = False
        for cf_val in chosen_fields:
            user_name = cf_val[1]
            if isinstance(cf_val[0], str):
                tree = self.parse_statistic(cf_val[0])
            else:
                tree = self.rebuild_tree(cf_val[0])
            self.chosen_fields.append((tree, user_name, None))
            self._need_mask = self._need_mask or self.need_mask(tree)
        self.voxel_size = (1, 1, 1)
        self.reversed_brightness = reversed_brightness
        self.use_gauss_image = use_gauss_image
        self.name_prefix = name_prefix

    def to_dict(self):
        return {"name": self.name, "chosen_fields": self.chosen_fields,
                "reversed_brightness": self.reversed_brightness, "use_gauss_image":self.use_gauss_image}

    def need_mask(self, tree):
        if isinstance(tree, Leaf):
            return self.STATISTIC_DICT[tree.name].is_mask
        else:
            return self.need_mask(tree.left) or self.need_mask(tree.right)

    def __str__(self):
        text = "Profile name: {}\n".format(self.name)
        text += "Reversed image [{}]\n".format(self.reversed_brightness)
        text += "Gaussed image [{}]\n".format(self.use_gauss_image)
        if self.name_prefix != "":
            text += "Name prefix: {}\n".format(self.name_prefix)
        text += "statistics list:\n"
        for el in self.chosen_fields:
            if el[2] is not None:
                text += "{}: {}\n".format(el[1], el[2])
            else:
                text += "{}\n".format(el[1])
        return text

    def get_component_info(self):
        """
        :return: list[(str, bool)]
        """
        res = []
        for tree, name, _ in self.chosen_fields:
            res.append((self.name_prefix+name, self._is_component_statistic(tree)))
        return res

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

        tree, _ = self.build_tree(tokens)
        return self.tree_to_dict_tree(tree)

    def is_any_mask_statistic(self):
        for el in self.chosen_fields:
            if self._is_mask_statistic(el[0]):
                return True
        return False

    def _is_mask_statistic(self, node):
        if isinstance(node, Leaf):
            return self.STATISTIC_DICT[node.name].is_mask
        else:
            return self._is_mask_statistic(node.left) or self._is_mask_statistic(node.right)

    def _is_component_statistic(self, node):
        if isinstance(node, Leaf):
            return self.STATISTIC_DICT[node.name].is_component
        else:
            return self._is_component_statistic(node.left) or self._is_component_statistic(node.right)

    @staticmethod
    def hash_fun_call_name(fun_name, arguments):
        return "{}: {}".format(fun_name, arguments)

    def calculate_tree(self, node, help_dict, kwargs):
        """
        :type node: Leaf | Node
        :type help_dict: dict
        :type kwargs: dict
        :return: float
        """
        if isinstance(node, Leaf):
            fun_name = self.STATISTIC_DICT[node.name].function_name
            kw = dict(kwargs)
            kw.update(node.dict)
            hash_str = self.hash_fun_call_name(fun_name, node.dict)
            if hash_str in help_dict:
                return help_dict[hash_str]
            kw['help_dict'] = help_dict
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

    def calculate(self, image, gauss_image, segmentation, full_mask, base_mask, voxel_size):
        self.voxel_size = voxel_size
        if self._need_mask and base_mask is None:
            raise ValueError("Statistics need mask")
        result = OrderedDict()
        if self.use_gauss_image:
            image = gauss_image.astype(np.float)
        else:
            image = image.astype(np.float)
        if self.reversed_brightness:
            noise_mean = np.mean(image[full_mask == 0])
            image = noise_mean - image
        help_dict = dict()
        kw = {"image": image, "segmentation": segmentation, "base_mask": base_mask, "full_segmentation": full_mask}
        for tree, user_name, params in self.chosen_fields:
            try:
                result[self.name_prefix + user_name] = self.calculate_tree(tree, help_dict, kw)
            except ZeroDivisionError:
                result[self.name_prefix + user_name] = "Div by zero"
            except TypeError as e:
                logging.warning(e)
                print(traceback.print_exc())
                result[self.name_prefix + user_name] = "None div"
            except AttributeError as e:
                logging.warning(e)
                result[self.name_prefix + user_name] = "No attribute"
        return result

    @staticmethod
    def pixel_volume(spacing):
        return reduce((lambda x, y: x * y), spacing)

    def calculate_volume(self, segmentation, **_):
        return np.count_nonzero(segmentation) * self.pixel_volume(self.voxel_size)

    def calculate_component_volume(self, segmentation, **_):
        return np.bincount(segmentation.flat)[1:] * self.pixel_volume(self.voxel_size)

    def calculate_main_axis(self, segmentation: np.ndarray, image: np.ndarray):
        cut_img = np.copy(image)
        cut_img[segmentation == 0] = 0
        cut_img = np.swapaxes(cut_img, 0, 2)
        orientation_matrix, _ = af.find_density_orientation(cut_img, self.voxel_size, 1)
        center_of_mass = af.density_mass_center(cut_img, self.voxel_size)
        # positions = np.array(np.nonzero(np.swapaxes(segmentation, 0, 2)), dtype=np.float64)
        positions = np.array(np.nonzero(cut_img), dtype=np.float64)
        for i, v in enumerate(self.voxel_size):
            positions[i] *= v
            positions[i] -= center_of_mass[i]
        centered = np.dot(orientation_matrix.T, positions)
        size = np.max(centered, axis=1) - np.min(centered, axis=1)
        return size

    def get_main_axis_length(self, index, segmentation: np.ndarray, image: np.ndarray, help_dict: Dict,  **_):
        if "main_axis" not in help_dict:
            help_dict["main_axis"] = self.calculate_main_axis(segmentation, image)
        return help_dict["main_axis"][index]

    def calculate_first_main_axis_length(self, **kwargs):
        return self.get_main_axis_length(0, **kwargs)

    def calculate_second_main_axis_length(self, **kwargs):
        return self.get_main_axis_length(1, **kwargs)

    def calculate_third_main_axis_length(self, **kwargs):
        return self.get_main_axis_length(2, **kwargs)

    def calculate_compactness(self, **kwargs):
        help_dict = kwargs["help_dict"]
        border_hash_str = self.hash_fun_call_name('calculate_border_surface', {})
        if border_hash_str not in help_dict:
            border_surface = self.calculate_border_surface(**kwargs)
            help_dict[border_hash_str] = border_surface
        else:
            border_surface = help_dict[border_hash_str]

        volume_hash_str = self.hash_fun_call_name('calculate_volume', {})

        if volume_hash_str not in help_dict:
            volume = self.calculate_volume(**kwargs)
            help_dict[volume_hash_str] = volume
        else:
            volume = help_dict[volume_hash_str]
        return border_surface**1.5/volume

    def calculate_sphericity(self, **kwargs):
        help_dict = kwargs["help_dict"]

        volume_hash_str = self.hash_fun_call_name('calculate_volume', {})
        if volume_hash_str not in help_dict:
            volume = self.calculate_volume(**kwargs)
            help_dict[volume_hash_str] = volume
        else:
            volume = help_dict[volume_hash_str]

        diameter_hash_str = self.hash_fun_call_name('segmentation_diameter', {})
        if diameter_hash_str not in help_dict:
            diameter = self.segmentation_diameter(**kwargs)
            help_dict[volume_hash_str] = diameter
        else:
            diameter = help_dict[volume_hash_str]
        radius = diameter / 2
        return volume/radius**3


    @staticmethod
    def calculate_mass(segmentation, image, **_):
        if np.any(segmentation):
            return np.sum(image[segmentation > 0])
        return 0

    @staticmethod
    def calculate_component_mass(segmentation, image, **_):
        res = []
        for i in range(1, segmentation.max()+1):
            res.append(np.sum(image[segmentation == i]))
        return res

    def calculate_border_surface(self, segmentation, **_):
        return calculate_volume_surface(segmentation, self.voxel_size)

    @staticmethod
    def maximum_brightness(segmentation, image, **_):
        if np.any(segmentation):
            return np.max(image[segmentation > 0])
        else:
            return None

    @staticmethod
    def minimum_brightness(segmentation, image, **_):
        if np.any(segmentation):
            return np.min(image[segmentation > 0])
        else:
            return None

    @staticmethod
    def median_brightness(segmentation, image, **_):
        if np.any(segmentation):
            return np.median(image[segmentation > 0])
        else:
            return None

    @staticmethod
    def std_brightness(segmentation, image, **_):
        if np.any(segmentation):
            return np.std(image[segmentation > 0])
        else:
            return None

    @staticmethod
    def mean_brightness(segmentation, image, **_):
        if np.any(segmentation):
            return np.mean(image[segmentation > 0])
        else:
            return None

    @staticmethod
    def std_noise(full_segmentation, base_mask, image, **_):
        if np.any(full_segmentation):
            if base_mask is not None:
                return np.std(image[(full_segmentation == 0) * (base_mask > 0)])
            else:
                return np.std(image[full_segmentation == 0])
        else:
            return None

    def moment_of_inertia(self, image, segmentation, **_):
        if image.ndim != 3:
            return None
        img = np.copy(image)
        img[segmentation == 0] = 0
        return af.calculate_density_momentum(img, self.voxel_size,)

    def border_mask(self, base_mask, radius, **_):
        if base_mask is None:
            return None
        final_radius = [int(radius / x) for x in self.voxel_size[::-1]]
        base_mask = np.array(base_mask > 0)
        mask = base_mask.astype(np.uint8)
        eroded = sitk.GetArrayFromImage(sitk.BinaryErode(sitk.GetImageFromArray(mask), final_radius))
        mask[eroded > 0] = 0
        return mask

    def border_mass(self, image, segmentation, **kwargs):
        border_mask = self.border_mask(**kwargs)
        if border_mask is None:
            return None
        final_mask = np.array((border_mask > 0) * (segmentation > 0))
        if np.any(final_mask):
            return np.sum(image[final_mask])
        return 0

    def border_volume(self, segmentation, **kwargs):
        border_mask = self.border_mask(**kwargs)
        if border_mask is None:
            return None
        final_mask = np.array((border_mask > 0) * (segmentation > 0))
        return np.count_nonzero(final_mask) * self.pixel_volume(self.voxel_size)

    @staticmethod
    def number_of_components(segmentation, **_):
        return segmentation.max()

    def mask_volume(self, base_mask, **_):
        return np.count_nonzero(base_mask) * self.pixel_volume(self.voxel_size)

    def mask_diameter(self, base_mask, **_):
        return calc_diam(get_border(base_mask), self.voxel_size)

    def segmentation_diameter(self, segmentation, **_):
        return calc_diam(get_border(segmentation), self.voxel_size)

    def component_diameter(self, segmentation, **_):
        unique = np.unique(segmentation[segmentation > 0])
        return np.array([calc_diam(get_border(segmentation == i), self.voxel_size) for i in unique], dtype=np.float)


def calculate_volume_surface(volume_mask, voxel_size):
    border_surface = 0
    surf_im = np.array(volume_mask).astype(np.uint8)
    border_surface += np.count_nonzero(np.logical_xor(surf_im[1:], surf_im[:-1])) * voxel_size[1] * voxel_size[2]
    border_surface += np.count_nonzero(np.logical_xor(surf_im[:, 1:], surf_im[:, :-1])) * voxel_size[0] * voxel_size[2]
    if len(surf_im.shape) == 3:
        border_surface += np.count_nonzero(np.logical_xor(surf_im[:, :, 1:], surf_im[:, :, :-1])) * voxel_size[0] * \
                          voxel_size[1]
    return border_surface


def get_border(array):
    if array.dtype == np.bool:
        array = array.astype(np.uint8)
    return sitk.GetArrayFromImage(sitk.LabelContour(sitk.GetImageFromArray(array)))


def calc_diam(array, voxel_size):
    pos = np.transpose(np.nonzero(array)).astype(np.float)
    for i, val in enumerate(reversed(voxel_size)):
        pos[:, i] *= val
    diam = 0
    for i, p in enumerate(zip(pos[:-1])):
        tmp = np.array((pos[i+1:] - p) ** 2)
        diam = max(diam, np.max(np.sum(tmp, 1)))
    return np.sqrt(diam)
