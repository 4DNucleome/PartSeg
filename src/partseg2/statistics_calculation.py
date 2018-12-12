from __future__ import division

import logging
import sys
import traceback
from abc import ABC
from collections import OrderedDict
from enum import Enum
from functools import reduce
from math import pow
from typing import NamedTuple, Optional, Union, Dict, Callable, List

import SimpleITK as sitk
import numpy as np

from partseg_utils import class_to_dict, autofit as af
from partseg_utils.class_generator import BaseReadonlyClass
from partseg_utils.class_generator import enum_register
from partseg_utils.segmentation.algorithm_describe_base import AlgorithmDescribeBase, Register, AlgorithmProperty
from partseg_utils.universal_const import UNITS_LIST, UNIT_SCALE


class AreaType(Enum):
    Segmentation = 1
    Mask = 2
    Mask_without_segmentation = 3

    def __str__(self):
        return self.name.replace("_", " ")


class PerComponent(Enum):
    No = 1
    Yes = 2

    def __str__(self):
        return self.name.replace("_", " ")


enum_register.register_class(AreaType)
enum_register.register_class(PerComponent)


class SettingsValue(NamedTuple):
    function: Callable
    help_message: str
    arguments: Optional[dict]
    is_component: bool
    default_area: Optional[AreaType] = None


class Leaf(BaseReadonlyClass):
    name: str
    dict: Dict = dict()
    power: float = 1.0
    area: Optional[AreaType] = None
    per_component: Optional[PerComponent] = None

    def __str__(self):
        resp = self.name
        if self.area is not None:
            resp = str(self.area) + " " + resp
        if self.per_component is not None and self.per_component == PerComponent.Yes:
            resp += " per component "
        if len(self.dict) != 0:
            resp += "["
            for k, v in self.dict.items():
                resp += f"{k}={v} "
            else:
                resp = resp[:-1]
            resp += "]"
        if self.power != 1.0:
            resp += f" to the power {self.power}"
        return resp


class Node(BaseReadonlyClass):
    left: Union['Node', Leaf]
    op: str
    right: Union['Node', Leaf]

    def __str__(self):
        left_text = "(" + str(self.left) + ")" if isinstance(self.left, Node) else str(self.left)
        right_text = "(" + str(self.right) + ")" if isinstance(self.right, Node) else str(self.right)
        return left_text + self.op + right_text


class StatisticEntry(BaseReadonlyClass):
    name: str
    calculation_tree: Union[Node, Leaf]


class MethodBase(AlgorithmDescribeBase, ABC):
    text_info = "", ""

    @classmethod
    def get_name(cls):
        return cls.text_info[0]

    @classmethod
    def get_description(cls):
        return cls.text_info[1]

    @classmethod
    def is_component(cls):
        return False

    @classmethod
    def get_fields(cls):
        return []

    @staticmethod
    def calculate_property(**kwargs):
        raise NotImplementedError()

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(cls.get_name())


class StatisticProfile(object):
    PARAMETERS = ["name", "chosen_fields", "reversed_brightness", "use_gauss_image", "name_prefix"]

    def __init__(self, name, chosen_fields: List[StatisticEntry], name_prefix=""):
        self.name = name
        self.chosen_fields: List[StatisticEntry] = chosen_fields
        self._need_mask = False
        for cf_val in chosen_fields:
            self._need_mask = self._need_mask or self.need_mask(cf_val.calculation_tree)
        self.name_prefix = name_prefix

    def to_dict(self):
        return {"name": self.name, "chosen_fields": self.chosen_fields, "name_prefix": self.name_prefix}

    def need_mask(self, tree):
        if isinstance(tree, Leaf):
            return tree.area == AreaType.Mask or tree.area == AreaType.Mask_without_segmentation
        else:
            return self.need_mask(tree.left) or self.need_mask(tree.right)

    def _need_mask_without_segmentation(self, tree):
        if isinstance(tree, Leaf):
            return tree.area == AreaType.Mask_without_segmentation
        else:
            return self.need_mask(tree.left) or self.need_mask(tree.right)

    def __str__(self):
        text = "Profile name: {}\n".format(self.name)
        if self.name_prefix != "":
            text += "Name prefix: {}\n".format(self.name_prefix)
        text += "statistics list:\n"
        for el in self.chosen_fields:
            text += "{}\n".format(el.name)
        return text

    def get_component_info(self):
        """
        :return: list[(str, bool)]
        """
        res = []
        for el in self.chosen_fields:
            res.append((self.name_prefix + el.name, self._is_component_statistic(el.calculation_tree)))
        return res

    def get_parameters(self):
        return class_to_dict(self, *self.PARAMETERS)

    def rebuild_tree(self, l):
        if len(l) == 2:
            return Leaf(*l)
        else:
            return Node(left=self.rebuild_tree(l[0]), op=l[1], right=self.rebuild_tree(l[2]))

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
            return (beg + "{}" + end).format(self.flat_tree(t.left), t.op, self.flat_tree(t.right))

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

    def is_any_mask_statistic(self):
        for el in self.chosen_fields:
            if self.need_mask(el.calculation_tree):
                return True
        return False

    def _is_component_statistic(self, node):
        if isinstance(node, Leaf):
            return node.per_component == PerComponent.Yes
        else:
            return self._is_component_statistic(node.left) or self._is_component_statistic(node.right)

    def calculate_tree(self, node, help_dict, kwargs):
        """
        :type node: Leaf | Node
        :type help_dict: dict
        :type kwargs: dict
        :return: float
        """
        if isinstance(node, Leaf):
            method = STATISTIC_DICT[node.name]
            kw = dict(kwargs)
            kw.update(node.dict)
            hash_str = hash_fun_call_name(method, node.dict, node.area)
            if hash_str in help_dict:
                val = help_dict[hash_str]
            else:
                kw['help_dict'] = help_dict
                kw['_area'] = node.area
                if node.area == AreaType.Mask:
                    kw["area_array"] = kw["base_mask"]
                elif node.area == AreaType.Mask_without_segmentation:
                    kw["area_array"] = kw["mask_without_segmentation"]
                elif node.area == AreaType.Segmentation:
                    kw["area_array"] = kw["segmentation"]
                else:
                    raise ValueError(f"Unknown area type {node.area}")
                if node.per_component == PerComponent.Yes:
                    val = []
                    area_array = kw["area_array"]
                    for i in np.unique(area_array)[1:]:
                        kw["area_array"] = area_array == i
                        val.append(method.calculate_property(**kw))
                else:
                    val = method.calculate_property(**kw)
                help_dict[hash_str] = val
            if node.power != 1:
                return pow(val, node.power)
            return val
        elif isinstance(node, Node):
            left_res = self.calculate_tree(node.left, help_dict, kwargs)
            right_res = self.calculate_tree(node.right, help_dict, kwargs)
            if node.op == "/":
                return left_res / right_res
        logging.error("Wrong statistics: {}".format(node))
        return 1

    def calculate(self, image: np.ndarray, segmentation: np.ndarray, full_mask: np.ndarray, mask: np.ndarray,
                  voxel_size, result_units):
        if self._need_mask and mask is None:
            raise ValueError("Statistics need mask")
        result = OrderedDict()
        image = image.astype(np.float)
        help_dict = dict()
        result_scalar = UNIT_SCALE[UNITS_LIST.index(result_units)]
        kw = {"image": image, "segmentation": segmentation, "mask": mask, "full_segmentation": full_mask,
              "voxel_size": voxel_size, "result_scalar": result_scalar}
        for el in self.chosen_fields:
            if self._need_mask_without_segmentation(el.calculation_tree):
                mm = mask.copy()
                mm[segmentation > 0] = 0
                kw["mask_without_segmentation"] = mm
                break
        for el in self.chosen_fields:
            tree, user_name = el.calculation_tree, el.name
            try:
                result[self.name_prefix + user_name] = self.calculate_tree(tree, help_dict, kw)
            except ZeroDivisionError:
                result[self.name_prefix + user_name] = "Div by zero"
            except TypeError as e:
                print(e, file=sys.stderr)
                print(traceback.print_exc(), file=sys.stderr)
                result[self.name_prefix + user_name] = "None div"
            except AttributeError as e:
                print(e, file=sys.stderr)
                result[self.name_prefix + user_name] = "No attribute"
        return result


def calculate_main_axis(area_array: np.ndarray, image: np.ndarray, voxel_size):
    # TODO check if it produces good values
    cut_img = np.copy(image)
    cut_img[area_array == 0] = 0
    # cut_img = np.swapaxes(cut_img, 0, 2)
    orientation_matrix, _ = af.find_density_orientation(cut_img, voxel_size, 1)
    center_of_mass = af.density_mass_center(cut_img, voxel_size)
    positions = np.array(np.nonzero(cut_img), dtype=np.float64)
    for i, v in enumerate(voxel_size):
        positions[i] *= v
        positions[i] -= center_of_mass[i]
    centered = np.dot(orientation_matrix.T, positions)
    size = np.max(centered, axis=1) - np.min(centered, axis=1)
    return size


def get_main_axis_length(index: int, area_array: np.ndarray, image: np.ndarray, help_dict: Dict, voxel_size,
                         result_scalar, _area: AreaType, **_):
    hash_name = hash_fun_call_name(calculate_main_axis, {}, _area)
    if hash_name not in help_dict:
        help_dict[hash_name] = calculate_main_axis(area_array, image, [x * result_scalar for x in voxel_size])
        print("get_main_axis_length", help_dict[hash_name], voxel_size, image.shape)
    return help_dict[hash_name][index]


def calculate_first_main_axis_length(**kwargs):
    return get_main_axis_length(0, **kwargs)


def calculate_second_main_axis_length(**kwargs):
    return get_main_axis_length(1, **kwargs)


def calculate_third_main_axis_length(**kwargs):
    return get_main_axis_length(2, **kwargs)


def diameter(area_array, voxel_size, **_):
    return calc_diam(get_border(area_array), voxel_size)


def component_diameter(segmentation, voxel_size, **_):
    unique = np.unique(segmentation[segmentation > 0])
    return np.array([calc_diam(get_border(segmentation == i), voxel_size) for i in unique], dtype=np.float)


def calculate_compactness(**kwargs):
    help_dict = kwargs["help_dict"]
    border_hash_str = hash_fun_call_name(calculate_border_surface, {}, kwargs["_area"])
    if border_hash_str not in help_dict:
        border_surface = calculate_border_surface(**kwargs)
        help_dict[border_hash_str] = border_surface
    else:
        border_surface = help_dict[border_hash_str]

    volume_hash_str = hash_fun_call_name(calculate_volume, {}, kwargs["_area"])

    if volume_hash_str not in help_dict:
        volume = calculate_volume(**kwargs)
        help_dict[volume_hash_str] = volume
    else:
        volume = help_dict[volume_hash_str]
    return border_surface ** 1.5 / volume


def calculate_sphericity(**kwargs):
    help_dict = kwargs["help_dict"]

    volume_hash_str = hash_fun_call_name(calculate_volume, {}, kwargs["_area"])
    if volume_hash_str not in help_dict:
        volume = calculate_volume(**kwargs)
        help_dict[volume_hash_str] = volume
    else:
        volume = help_dict[volume_hash_str]

    diameter_hash_str = hash_fun_call_name(diameter, {}, kwargs["_area"])
    if diameter_hash_str not in help_dict:
        diameter_val = diameter(**kwargs)
        help_dict[volume_hash_str] = diameter_val
    else:
        diameter_val = help_dict[volume_hash_str]
    radius = diameter_val / 2
    return volume / radius ** 3


def border_volume(segmentation, voxel_size, result_scalar, **kwargs):
    border_mask_array = border_mask(**kwargs)
    if border_mask_array is None:
        return None
    final_mask = np.array((border_mask_array > 0) * (segmentation > 0))
    return np.count_nonzero(final_mask) * pixel_volume(voxel_size, result_scalar)


def border_mass(image, segmentation, **kwargs):
    border_mask_array = border_mask(**kwargs)
    if border_mask_array is None:
        return None
    final_mask = np.array((border_mask_array > 0) * (segmentation > 0))
    if np.any(final_mask):
        return np.sum(image[final_mask])
    return 0


def number_of_components(area_array, **_):
    return np.unique(area_array).size - 1


def moment_of_inertia(image, segmentation, voxel_size, **_):
    if image.ndim != 3:
        return None
    img = np.copy(image)
    img[segmentation == 0] = 0
    return af.calculate_density_momentum(img, voxel_size, )


def hash_fun_call_name(fun: Callable, arguments: Dict, area: AreaType):
    if hasattr(fun, "__module__"):
        fun_name = f"{fun.__module__}.{fun.__name__}"
    else:
        fun_name = fun.__name__
    return "{}: {} # {}".format(fun_name, arguments, area)


class Volume(MethodBase):
    text_info = "Volume", "Calculate volume of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return np.count_nonzero(area_array) * pixel_volume(voxel_size, result_scalar)


class Diameter(MethodBase):
    text_info = "Diameter", "Diameter of area (Very slow)"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return np.bincount(area_array.flat)[1:] * pixel_volume(voxel_size, result_scalar)


class PixelBrightnessSum(MethodBase):
    text_info = "Pixel Brightness Sum", "Sum of pixel brightness for current segmentation"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.sum(image[area_array > 0])
        return 0


class ComponentsNumber(MethodBase):
    text_info = "Components Number", "Calculate number of connected components on segmentation"

    @staticmethod
    def calculate_property(area_array, **_):
        return np.unique(area_array).size - 1

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(cls.get_name(), per_component=PerComponent.No)


class MaximumPixelBrightness(MethodBase):
    text_info = "Maximum pixel brightness", "Calculate maximum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.max(image[area_array > 0])
        else:
            return None


class MinimumPixelBrightness(MethodBase):
    text_info = "Minimum pixel brightness", "Calculate minimum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.min(image[area_array > 0])
        else:
            return None


class MeanPixelBrightness(MethodBase):
    text_info = "Mean pixel brightness", "Calculate mean pixel brightness  for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.mean(image[area_array > 0])
        else:
            return None


class StandardDeviationOfPixelBrightness(MethodBase):
    text_info = "Standard deviation of pixel brightness", \
                "Calculate standard deviation of pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.std(image[area_array > 0])
        else:
            return None


class MomentOfInertia(MethodBase):
    text_info = "Moment of inertia", "Calculate moment of inertia for segmented structure"

    @staticmethod
    def calculate_property(image, area_array, voxel_size, **_):
        if image.ndim != 3:
            return None
        img = np.copy(image)
        img[area_array == 0] = 0
        return af.calculate_density_momentum(img, voxel_size)


class LongestMainAxisLength(MethodBase):
    text_info = "Longest main axis length", "Length of first main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(0, **kwargs)


class MiddleMainAxisLength(MethodBase):
    text_info = "Middle main axis length", "Length of second main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(1, **kwargs)


class ShortestMainAxisLength(MethodBase):
    text_info = "Shortest main axis length", "Length of third main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(2, **kwargs)


class Compactness(MethodBase):
    text_info = "Compactness", "Calculate compactness off segmentation (Surface^1.5/volume)"

    @staticmethod
    def calculate_property(**kwargs):
        help_dict = kwargs["help_dict"]
        border_hash_str = hash_fun_call_name(Surface, {}, kwargs["_area"])
        if border_hash_str not in help_dict:
            border_surface = Surface.calculate_property(**kwargs)
            help_dict[border_hash_str] = border_surface
        else:
            border_surface = help_dict[border_hash_str]

        volume_hash_str = hash_fun_call_name(Volume, {}, kwargs["_area"])

        if volume_hash_str not in help_dict:
            volume = Volume.calculate_property(**kwargs)
            help_dict[volume_hash_str] = volume
        else:
            volume = help_dict[volume_hash_str]
        return border_surface ** 1.5 / volume


class Sphericity(MethodBase):
    text_info = "Sphericity", "volume/(diameter**3/8)"

    @staticmethod
    def calculate_property(**kwargs):
        help_dict = kwargs["help_dict"]
        volume_hash_str = hash_fun_call_name(Volume, {}, kwargs["_area"])
        if volume_hash_str not in help_dict:
            volume = Volume.calculate_property(**kwargs)
            help_dict[volume_hash_str] = volume
        else:
            volume = help_dict[volume_hash_str]

        diameter_hash_str = hash_fun_call_name(Diameter, {}, kwargs["_area"])
        if diameter_hash_str not in help_dict:
            diameter_val = Diameter.calculate_property(**kwargs)
            help_dict[volume_hash_str] = diameter_val
        else:
            diameter_val = help_dict[volume_hash_str]
        radius = diameter_val / 2
        return volume / radius ** 3


class Surface(MethodBase):
    text_info = "Surface", "Calculating surface of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return calculate_volume_surface(area_array, [x * result_scalar for x in voxel_size])


def border_mask(mask, distance, units, voxel_size, **_):
    if mask is None:
        return None
    units_scalar = UNIT_SCALE[UNITS_LIST.index(units)]
    final_radius = [int((distance / units_scalar) / x) for x in voxel_size]
    mask = np.array(mask > 0)
    mask = mask.astype(np.uint8)
    eroded = sitk.GetArrayFromImage(sitk.BinaryErode(sitk.GetImageFromArray(mask), final_radius))
    mask[eroded > 0] = 0
    return mask


class RimVolume(MethodBase):
    text_info = "Rim Volume", "Calculate volumes for elements in radius (in physical units) from mask"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 1000), property_type=float),
                AlgorithmProperty("units", "Units", "nm", possible_values=UNITS_LIST)]

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.get_name(), area=AreaType.Segmentation, per_component=PerComponent.No)

    @staticmethod
    def calculate_property(segmentation, voxel_size, result_scalar, **kwargs):
        border_mask_array = border_mask(voxel_size=voxel_size, result_scalar=result_scalar, **kwargs)
        if border_mask_array is None:
            return None
        final_mask = np.array((border_mask_array > 0) * (segmentation > 0))
        return np.count_nonzero(final_mask) * pixel_volume(voxel_size, result_scalar)


class RimPixelBrightnessSum(MethodBase):
    text_info = "Rim Pixel Brightness Sum", \
                "Calculate mass for components located within rim (in physical units) from mask"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 1000), property_type=float),
                AlgorithmProperty("units", "Units", "nm", possible_values=UNITS_LIST)]

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.get_name(), area=AreaType.Segmentation, per_component=PerComponent.No)

    @staticmethod
    def calculate_property(image, segmentation, **kwargs):
        border_mask_array = border_mask(**kwargs)
        if border_mask_array is None:
            return None
        final_mask = np.array((border_mask_array > 0) * (segmentation > 0))
        if np.any(final_mask):
            return np.sum(image[final_mask])
        return 0


def mean_brightness(area_array, image, **_):
    if np.any(area_array):
        return np.mean(image[area_array > 0])
    else:
        return None


def std_brightness(area_array, image, **_):
    if np.any(area_array):
        return np.std(image[area_array > 0])
    else:
        return None


def median_brightness(area_array, image, **_):
    if np.any(area_array):
        return np.median(image[area_array > 0])
    else:
        return None


def minimum_brightness(area_array, image, **_):
    if np.any(area_array):
        return np.min(image[area_array > 0])
    else:
        return None


def maximum_brightness(area_array, image, **_):
    if np.any(area_array):
        return np.max(image[area_array > 0])
    else:
        return None


def calculate_border_surface(area_array, voxel_size, **_):
    return calculate_volume_surface(area_array, voxel_size)


def calculate_component_mass(area_array, image, **_):
    res = []
    for i in range(1, area_array.max() + 1):
        res.append(np.sum(image[area_array == i]))
    return res


def calculate_mask_mass(base_mask, image, **_):
    if np.any(base_mask):
        return np.sum(image[base_mask > 0])
    return 0


def calculate_mass(segmentation, image, **_):
    if np.any(segmentation):
        return np.sum(image[segmentation > 0])
    return 0


def pixel_volume(spacing, result_scalar):
    return reduce((lambda x, y: x * y), [x * result_scalar for x in spacing])


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
        tmp = np.array((pos[i + 1:] - p) ** 2)
        diam = max(diam, np.max(np.sum(tmp, 1)))
    return np.sqrt(diam)


def calculate_volume(area_array, voxel_size, result_scalar, **_):
    return np.count_nonzero(area_array) * pixel_volume(voxel_size, result_scalar)


def calculate_component_volume(area_array, voxel_size, result_scalar, **_):
    return np.bincount(area_array.flat)[1:] * pixel_volume(voxel_size, result_scalar)


STATISTIC_DICT = Register(Volume, Diameter, PixelBrightnessSum, ComponentsNumber, MaximumPixelBrightness,
                          MinimumPixelBrightness, MeanPixelBrightness, StandardDeviationOfPixelBrightness,
                          MomentOfInertia, LongestMainAxisLength, MiddleMainAxisLength, ShortestMainAxisLength,
                          Compactness, Sphericity, Surface, RimVolume, RimPixelBrightnessSum)

STATISTIC_DICT2: Dict[str, SettingsValue] = OrderedDict({
    "Volume": SettingsValue(calculate_volume, "Calculate volume of current segmentation", None, False),
    "Diameter": SettingsValue(diameter, "Diameter of area", None, False),

    "Pixel Brightness Sum": SettingsValue(calculate_mass, "Sum of pixel brightness for current segmentation", None,
                                          False),
    "Components Number": SettingsValue(number_of_components, "Calculate number of connected components "
                                                             "on segmentation", None, False),
    "Volume per component": SettingsValue(calculate_component_volume, "Calculate volume of each component "
                                                                      "of current segmentation", None, True),
    "Pixel Brightness Sum per component": SettingsValue(calculate_component_mass,
                                                        "Sum of pixel brightness of each component of"
                                                        "segmentation", None, True),
    "Maximum pixel brightness": SettingsValue(
        maximum_brightness, "Calculate maximum brightness of pixel for current segmentation", None, False),
    "Minimum pixel brightness": SettingsValue(
        minimum_brightness, "Calculate minimum brightness of pixel for current segmentation", None, False),
    "Median pixel brightness": SettingsValue(
        median_brightness, "Calculate median brightness of pixel for current segmentation", None, False),
    "Mean pixel brightness": SettingsValue(
        mean_brightness, "Calculate median brightness of pixel for current segmentation", None, False),
    "Standard deviation of pixel brightness": SettingsValue(
        std_brightness, "Calculate  standard deviation of pixel brightness for current segmentation", None, False),
    "Moment of inertia":
        SettingsValue(moment_of_inertia, "Calculate moment of inertia for segmented structure. Has one parameter thr "
                                         "(threshold). Only values above it are used in calculation", None, False),
    "Rim Pixel Brightness Sum": SettingsValue(border_mass,
                                              "Calculate mass for components located within rim (in physical units)"
                                              " from mask", {"radius": int}, False),
    "Rim Volume": SettingsValue(border_volume, "Calculate volumes for elements in radius (in physical units)"
                                               " from mask", {"radius": int}, False),
    "Components Diameter":
        SettingsValue(component_diameter, "Diameter of each component", None, False),
    "Longest main axis length":
        SettingsValue(calculate_first_main_axis_length,
                      "Length of first main axis", None, False),
    "Middle main axis length":
        SettingsValue(calculate_second_main_axis_length,
                      "Length of second main axis", None, False),
    "Shortest main axis length":
        SettingsValue(calculate_third_main_axis_length,
                      "Length of third main axis", None, False),
    "Compactness":
        SettingsValue(calculate_compactness,
                      "Calculate compactness off segmentation (Surface^1.5/volume)", None, False),
    "Sphericity": SettingsValue(calculate_sphericity, "volume/(diameter**3/8)", None, False),
    "Surface": SettingsValue(calculate_border_surface,
                             "Calculating surface of current segmentation", None, False)
})
