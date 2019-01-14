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
from sympy import symbols
from ..partseg_utils import class_to_dict, autofit as af
from ..partseg_utils.border_rim import border_mask
from ..partseg_utils.class_generator import BaseReadonlyClass
from ..partseg_utils.class_generator import enum_register
from ..partseg_utils.segmentation.algorithm_describe_base import AlgorithmDescribeBase, Register, AlgorithmProperty
from ..partseg_utils.universal_const import UNITS_LIST, UNIT_SCALE


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
        return str(cls.get_starting_leaf())

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
        return Leaf(cls.text_info[0])

    @classmethod
    def get_units(cls, ndim):
        raise NotImplementedError()


def empty_fun(_a0=None, _a1=None):
    pass


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
                    kw["area_array"] = kw["mask"]
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
            unit = method.get_units(3)
            if node.power != 1:
                return pow(val, node.power), pow(unit, node.power)
            return val, unit
        elif isinstance(node, Node):
            left_res, left_unit = self.calculate_tree(node.left, help_dict, kwargs)
            right_res, right_unit = self.calculate_tree(node.right, help_dict, kwargs)
            if node.op == "/":
                return left_res / right_res, left_unit/right_unit
        logging.error("Wrong statistics: {}".format(node))
        return 1

    def calculate(self, image: np.ndarray, segmentation: np.ndarray, full_mask: np.ndarray, mask: np.ndarray,
                  voxel_size, result_units, range_changed=None, step_changed=None):
        if range_changed is None:
            range_changed = empty_fun
        if step_changed is None:
            step_changed = empty_fun
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
        range_changed(0, len(self.chosen_fields))
        for i, el in enumerate(self.chosen_fields):
            step_changed(i)
            tree, user_name = el.calculation_tree, el.name
            try:
                result[self.name_prefix + user_name] = self.calculate_tree(tree, help_dict, kw)
            except ZeroDivisionError:
                result[self.name_prefix + user_name] = "Div by zero", ""
            except TypeError as e:
                print(e, file=sys.stderr)
                print(traceback.print_exc(), file=sys.stderr)
                result[self.name_prefix + user_name] = "None div", ""
            except AttributeError as e:
                print(e, file=sys.stderr)
                result[self.name_prefix + user_name] = "No attribute", ""
        return result


def calculate_main_axis(area_array: np.ndarray, image: np.ndarray, voxel_size):
    # TODO check if it produces good values
    cut_img = np.copy(image)
    cut_img[area_array == 0] = 0
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
    return help_dict[hash_name][index]


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

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** ndim


class Diameter(MethodBase):
    text_info = "Diameter", "Diameter of area"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        pos = np.transpose(np.nonzero(get_border(area_array))).astype(np.float)
        for i, val in enumerate([x * result_scalar for x in voxel_size]):
            pos[:, i] *= val
        p1 = 0
        blocked_set = {p1}
        diam = 0
        while True:
            dist_array = np.sum(np.array((pos - pos[p1]) ** 2), 1)
            p2 = np.argmax(dist_array)
            diam = max(dist_array[p2], diam)
            mid_point = (pos[p1] + pos[p2])/2
            dist_array = np.sum(np.array((pos - mid_point) ** 2), 1)
            p1 = np.argmax(dist_array)
            if dist_array[p1] <= diam / 4 + 1 or p1 in blocked_set:
                return np.sqrt(diam)
            blocked_set.add(p1)
            print("step", np.sqrt(diam))

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class DiameterOld(MethodBase):
    text_info = "Diameter old", "Diameter of area (Very slow)"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return calc_diam(get_border(area_array), [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class PixelBrightnessSum(MethodBase):
    text_info = "Pixel Brightness Sum", "Sum of pixel brightness for current segmentation"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.sum(image[area_array > 0])
        return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class ComponentsNumber(MethodBase):
    text_info = "Components Number", "Calculate number of connected components on segmentation"

    @staticmethod
    def calculate_property(area_array, **_):
        return np.unique(area_array).size - 1

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(cls.text_info[0], per_component=PerComponent.No)

    @classmethod
    def get_units(cls, ndim):
        return symbols("count")


class MaximumPixelBrightness(MethodBase):
    text_info = "Maximum pixel brightness", "Calculate maximum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.max(image[area_array > 0])
        else:
            return None

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class MinimumPixelBrightness(MethodBase):
    text_info = "Minimum pixel brightness", "Calculate minimum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.min(image[area_array > 0])
        else:
            return None

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class MeanPixelBrightness(MethodBase):
    text_info = "Mean pixel brightness", "Calculate mean pixel brightness  for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.mean(image[area_array > 0])
        else:
            return None

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class StandardDeviationOfPixelBrightness(MethodBase):
    text_info = "Standard deviation of pixel brightness", \
                "Calculate standard deviation of pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.std(image[area_array > 0])
        else:
            return None

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class MomentOfInertia(MethodBase):
    text_info = "Moment of inertia", "Calculate moment of inertia for segmented structure"

    @staticmethod
    def calculate_property(image, area_array, voxel_size, **_):
        if image.ndim != 3:
            return None
        img = np.copy(image)
        img[area_array == 0] = 0
        return af.calculate_density_momentum(img, voxel_size)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 2 * symbols("Pixel_brightness")


class LongestMainAxisLength(MethodBase):
    text_info = "Longest main axis length", "Length of first main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(0, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class MiddleMainAxisLength(MethodBase):
    text_info = "Middle main axis length", "Length of second main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(1, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class ShortestMainAxisLength(MethodBase):
    text_info = "Shortest main axis length", "Length of third main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(2, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


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

    @classmethod
    def get_units(cls, ndim):
        return Surface.get_units(ndim)/Volume.get_units(ndim)


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
            help_dict[diameter_hash_str] = diameter_val
        else:
            diameter_val = help_dict[diameter_hash_str]
        radius = diameter_val / 2
        return volume / radius ** 3

    @classmethod
    def get_units(cls, ndim):
        return Volume.get_units(ndim) / Diameter.get_units(ndim)**3


class Surface(MethodBase):
    text_info = "Surface", "Calculating surface of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return calculate_volume_surface(area_array, [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")**2


class RimVolume(MethodBase):
    text_info = "Rim Volume", "Calculate volumes for elements in radius (in physical units) from mask"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 1000), property_type=float),
                AlgorithmProperty("units", "Units", "nm", possible_values=UNITS_LIST)]

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask, per_component=PerComponent.No)

    @staticmethod
    def calculate_property(segmentation, voxel_size, result_scalar, **kwargs):
        border_mask_array = border_mask(voxel_size=voxel_size, result_scalar=result_scalar, **kwargs)
        if border_mask_array is None:
            return None
        final_mask = np.array((border_mask_array > 0) * (segmentation > 0))
        return np.count_nonzero(final_mask) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")**3


class RimPixelBrightnessSum(MethodBase):
    text_info = "Rim Pixel Brightness Sum", \
                "Calculate mass for components located within rim (in physical units) from mask"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 1000), property_type=float),
                AlgorithmProperty("units", "Units", "nm", possible_values=UNITS_LIST)]

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask, per_component=PerComponent.No)

    @staticmethod
    def calculate_property(image, segmentation, **kwargs):
        border_mask_array = border_mask(**kwargs)
        if border_mask_array is None:
            return None
        final_mask = np.array((border_mask_array > 0) * (segmentation > 0))
        if np.any(final_mask):
            return np.sum(image[final_mask])
        return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


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
    for i, val in enumerate(voxel_size):
        pos[:, i] *= val
    diam = 0
    for i, p in enumerate(zip(pos[:-1])):
        tmp = np.array((pos[i + 1:] - p) ** 2)
        diam = max(diam, np.max(np.sum(tmp, 1)))
    return np.sqrt(diam)


STATISTIC_DICT = Register(Volume, Diameter, PixelBrightnessSum, ComponentsNumber, MaximumPixelBrightness,
                          MinimumPixelBrightness, MeanPixelBrightness, StandardDeviationOfPixelBrightness,
                          MomentOfInertia, LongestMainAxisLength, MiddleMainAxisLength, ShortestMainAxisLength,
                          Compactness, Sphericity, Surface, RimVolume, RimPixelBrightnessSum)
