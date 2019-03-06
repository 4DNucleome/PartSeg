from __future__ import division
import logging
import sys
import traceback
from abc import ABC
from collections import OrderedDict
from enum import Enum
from functools import reduce
# from math import pow
from typing import NamedTuple, Optional, Union, Dict, Callable, List

import SimpleITK as sitk
import numpy as np
from sympy import symbols
from math import pi

from .. import autofit as af
from ..border_rim import border_mask
from ..class_generator import BaseSerializableClass
from ..class_generator import enum_register
from ..segmentation.algorithm_describe_base import AlgorithmDescribeBase, Register, AlgorithmProperty
from ..universal_const import UNIT_SCALE, Units
from ..utils import class_to_dict


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


class Leaf(BaseSerializableClass):
    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    def __init__(self, name: str, dict: Dict = None, power: float = 1.0, area: Optional[AreaType] = None,
                 per_component: Optional[PerComponent] = None):...

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

    def get_unit(self, ndim):
        method = STATISTIC_DICT[self.name]
        if self.power != 1:
            return method.get_units(ndim)**self.power
        return method.get_units(ndim)


class Node(BaseSerializableClass):
    left: Union['Node', Leaf]
    op: str
    right: Union['Node', Leaf]

    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    def __init__(self, left: Union['Node', Leaf], op: str, right: Union['Node', Leaf]):...

    def __str__(self):
        left_text = "(" + str(self.left) + ")" if isinstance(self.left, Node) else str(self.left)
        right_text = "(" + str(self.right) + ")" if isinstance(self.right, Node) else str(self.right)
        return left_text + self.op + right_text

    def get_unit(self, ndim):
        if self.op == "/":
            return self.left.get_unit(ndim) / self.right.get_unit(ndim)
        raise ValueError(f"Unknown operator '{self.op}'")


class StatisticEntry(BaseSerializableClass):
    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    def __init__(self, name:str, calculation_tree: Union[Node, Leaf]): ...

    name: str
    calculation_tree: Union[Node, Leaf]

    def get_unit(self, unit: Units, ndim):
        return str(self.calculation_tree.get_unit(ndim)).format(str(unit))


class StatisticMethodBase(AlgorithmDescribeBase, ABC):
    """
    This is base class For all statistic calculation classes
    based on text_info[0] the Statistic name wil be generated, based_on text_info[1] the description is generated
    """
    text_info = "", ""

    @classmethod
    def get_name(cls):
        return str(cls.get_starting_leaf().name)

    @classmethod
    def get_description(cls):
        return cls.text_info[1]

    @classmethod
    def is_component(cls):
        """Return information if Need information about components"""
        return False

    @classmethod
    def get_fields(cls):
        """Additional fields needed by algorithm. like radius of dilation"""
        return []

    @staticmethod
    def calculate_property(**kwargs):
        """Main function for calculating statistic"""
        raise NotImplementedError()

    @classmethod
    def get_starting_leaf(cls):
        """This leaf is putted on default list"""
        return Leaf(cls.text_info[0])

    @classmethod
    def get_units(cls, ndim):
        """Return units for statistic. They are shown to user"""
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

    def get_component_info(self, unit: Units):
        """
        :return: list[((str, str), bool)]
        """
        res = []
        # Fixme remove binding to 3 dimensions
        for el in self.chosen_fields:
            res.append(((self.name_prefix + el.name, el.get_unit(unit, 3)),
                        self._is_component_statistic(el.calculation_tree)))
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
            hash_str = hash_fun_call_name(method, node.dict, node.area, node.per_component)
            if hash_str in help_dict:
                val = help_dict[hash_str]
            else:
                kw['help_dict'] = help_dict
                kw['_area'] = node.area
                kw['_per_component'] = node.per_component
                kw['_cache'] = True
                if node.area == AreaType.Mask:
                    kw["area_array"] = kw["mask"]
                elif node.area == AreaType.Mask_without_segmentation:
                    kw["area_array"] = kw["mask_without_segmentation"]
                elif node.area == AreaType.Segmentation:
                    kw["area_array"] = kw["segmentation"]
                else:
                    raise ValueError(f"Unknown area type {node.area}")
                if node.per_component == PerComponent.Yes:
                    kw['_cache'] = False
                    val = []
                    area_array = kw["area_array"]
                    for i in np.unique(area_array)[1:]:
                        kw["area_array"] = area_array == i
                        val.append(method.calculate_property(**kw))
                else:
                    val = method.calculate_property(**kw)
                help_dict[hash_str] = val
            unit = method.get_units(3) if kw["image"].shape[0] > 1 else method.get_units(3)
            if node.power != 1:
                return pow(val, node.power), pow(unit, node.power)
            return val, unit
        elif isinstance(node, Node):
            left_res, left_unit = self.calculate_tree(node.left, help_dict, kwargs)
            right_res, right_unit = self.calculate_tree(node.right, help_dict, kwargs)
            if node.op == "/":
                return left_res / right_res, left_unit / right_unit
        logging.error("Wrong statistics: {}".format(node))
        return 1

    def calculate(self, image: np.ndarray, segmentation: np.ndarray, full_mask: np.ndarray, mask: np.ndarray,
                  voxel_size, result_units: Units, range_changed=None, step_changed=None):
        if range_changed is None:
            range_changed = empty_fun
        if step_changed is None:
            step_changed = empty_fun
        if self._need_mask and mask is None:
            raise ValueError("Statistics need mask")
        result = OrderedDict()
        image = image.astype(np.float)
        help_dict = dict()
        result_scalar = UNIT_SCALE[result_units.value]
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
                val, unit = self.calculate_tree(tree, help_dict, kw)
                result[self.name_prefix + user_name] = val, str(unit).format(str(result_units))
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
    if np.all(cut_img == 0):
        return (0,) * len(voxel_size)
    orientation_matrix, _ = af.find_density_orientation(cut_img, voxel_size, 1)
    center_of_mass = af.density_mass_center(cut_img, voxel_size)
    positions = np.array(np.nonzero(cut_img), dtype=np.float64)
    for i, v in enumerate(reversed(voxel_size), start=1):
        positions[-i] *= v
        positions[-i] -= center_of_mass[i-1]
    centered = np.dot(orientation_matrix.T, positions)
    size = np.max(centered, axis=1) - np.min(centered, axis=1)
    return size


def get_main_axis_length(index: int, area_array: np.ndarray, image: np.ndarray, voxel_size,
                         result_scalar,
                         _cache=False, **kwargs):
    _cache = _cache and "_area" in kwargs and "_per_component" in kwargs
    if _cache:
        help_dict: Dict = kwargs["help_dict"]
        _area: AreaType = kwargs["_area"]
        _per_component: PerComponent = kwargs["_per_component"]
        hash_name = hash_fun_call_name(calculate_main_axis, {}, _area, _per_component)
        if hash_name not in help_dict:
            help_dict[hash_name] = calculate_main_axis(area_array, image, [x * result_scalar for x in voxel_size])
        return help_dict[hash_name][index]
    else:
        return calculate_main_axis(area_array, image, [x * result_scalar for x in voxel_size])[index]


def hash_fun_call_name(fun: Callable, arguments: Dict, area: AreaType, per_component: PerComponent):
    if hasattr(fun, "__module__"):
        fun_name = f"{fun.__module__}.{fun.__name__}"
    else:
        fun_name = fun.__name__
    return "{}: {} # {} & {}".format(fun_name, arguments, area, per_component)


class Volume(StatisticMethodBase):
    text_info = "Volume", "Calculate volume of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return np.count_nonzero(area_array) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** ndim


# From Malandain, G., & Boissonnat, J. (2002). Computing the diameter of a point set,
# 12(6), 489–509. https://doi.org/10.1142/S0218195902001006


def double_normal(point_index: int, point_positions: np.ndarray, points_array: np.ndarray):
    """
    :param point_index: index of starting points
    :param point_positions: points array of size (points_num, number of dimensions)
    :param points_array: bool matrix with information about which points are in set
    :return:
    """
    delta = 0
    dn = 0, 0
    while True:
        new_delta = delta
        points_array[point_index] = 0
        dist_array = np.sum(np.array((point_positions - point_positions[point_index]) ** 2), 1)
        dist_array[points_array == 0] = 0
        point2_index = np.argmax(dist_array)
        if dist_array[point2_index] > new_delta:
            delta = dist_array[point2_index]
            dn = point_index, point2_index
            point_index = point2_index
        if new_delta == delta:
            return dn, delta


def iterative_double_normal(points_positions: np.ndarray):
    """
    :param points_positions: points array of size (points_num, number of dimensions)
    :return: square power of diameter, 2-tuple of points index gave information which points ar chosen
    """
    delta = 0
    dn = 0, 0
    point_index = 0
    points_array = np.ones(points_positions.shape[0], dtype=np.bool)
    while True:
        dn_r, delta_r = double_normal(point_index, points_positions, points_array)
        if delta_r > delta:
            delta = delta_r
            dn = dn_r
            mid_point = (points_positions[dn[0]] + points_positions[dn[1]]) / 2
            dist_array = np.sum(np.array((points_positions - mid_point) ** 2), 1)
            dist_array[~points_array] = 0
            if np.any(dist_array >= delta / 4):
                point_index = np.argmax(dist_array)
            else:
                break

        else:
            break
    return delta, dn


class Diameter(StatisticMethodBase):
    text_info = "Diameter", "Diameter of area"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        pos = np.transpose(np.nonzero(get_border(area_array))).astype(np.float)
        if pos.size == 0:
            return 0
        for i, val in enumerate([x * result_scalar for x in reversed(voxel_size)], start=1):
            pos[:, -i] *= val
        diam_sq, cords = iterative_double_normal(pos)
        return np.sqrt(diam_sq)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class DiameterOld(StatisticMethodBase):
    text_info = "Diameter old", "Diameter of area (Very slow)"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return calc_diam(get_border(area_array), [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class PixelBrightnessSum(StatisticMethodBase):
    text_info = "Pixel Brightness Sum", "Sum of pixel brightness for current segmentation"

    @staticmethod
    def calculate_property(area_array: np.ndarray, image: np.ndarray, **_):
        """
        :param area_array: mask for area
        :param image: data. same shape like area_type
        :return: Pixels brightness sum on given area
        """
        if np.any(area_array):
            return np.sum(image[area_array > 0])
        return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class ComponentsNumber(StatisticMethodBase):
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


class MaximumPixelBrightness(StatisticMethodBase):
    text_info = "Maximum pixel brightness", "Calculate maximum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.max(image[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class MinimumPixelBrightness(StatisticMethodBase):
    text_info = "Minimum pixel brightness", "Calculate minimum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.min(image[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class MeanPixelBrightness(StatisticMethodBase):
    text_info = "Mean pixel brightness", "Calculate mean pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.mean(image[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class MedianPixelBrightness(StatisticMethodBase):
    text_info = "Median pixel brightness", "Calculate median pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.median(image[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class StandardDeviationOfPixelBrightness(StatisticMethodBase):
    text_info = "Standard deviation of pixel brightness", \
                "Calculate standard deviation of pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, image, **_):
        if np.any(area_array):
            return np.std(image[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")


class MomentOfInertia(StatisticMethodBase):
    text_info = "Moment of inertia", "Calculate moment of inertia for segmented structure"

    @staticmethod
    def calculate_property(area_array, image, voxel_size, **_):
        if image.ndim != 3:
            return None
        img = np.copy(image)
        img[area_array == 0] = 0
        if np.all(img == 0):
            return 0
        return af.calculate_density_momentum(img, voxel_size)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 2 * symbols("Pixel_brightness")


class LongestMainAxisLength(StatisticMethodBase):
    text_info = "Longest main axis length", "Length of first main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(0, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class MiddleMainAxisLength(StatisticMethodBase):
    text_info = "Middle main axis length", "Length of second main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(1, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class ShortestMainAxisLength(StatisticMethodBase):
    text_info = "Shortest main axis length", "Length of third main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(2, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class Compactness(StatisticMethodBase):
    text_info = "Compactness", "Calculate compactness off segmentation (Surface^1.5/volume)"

    @staticmethod
    def calculate_property(**kwargs):
        cache = kwargs["_cache"] if "_cache" in kwargs else False
        cache = cache and "help_dict" in kwargs
        cache = cache and "_area" in kwargs
        cache = cache and "_per_component" in kwargs
        if cache:
            help_dict = kwargs["help_dict"]
            border_hash_str = hash_fun_call_name(Surface, {}, kwargs["_area"], kwargs["_per_component"])
            if border_hash_str not in help_dict:
                border_surface = Surface.calculate_property(**kwargs)
                help_dict[border_hash_str] = border_surface
            else:
                border_surface = help_dict[border_hash_str]

            volume_hash_str = hash_fun_call_name(Volume, {}, kwargs["_area"], kwargs["_per_component"])

            if volume_hash_str not in help_dict:
                volume = Volume.calculate_property(**kwargs)
                help_dict[volume_hash_str] = volume
            else:
                volume = help_dict[volume_hash_str]
        else:
            border_surface = Surface.calculate_property(**kwargs)
            volume = Volume.calculate_property(**kwargs)
        return border_surface ** 1.5 / volume

    @classmethod
    def get_units(cls, ndim):
        return Surface.get_units(ndim) / Volume.get_units(ndim)


class Sphericity(StatisticMethodBase):
    text_info = "Sphericity", "volume/((4/3 * π * radius **3) for 3d data and volume/((π * radius **2) for 2d data"

    @staticmethod
    def calculate_property(**kwargs):
        if all(key in kwargs for key in ["help_dict", "_area", "_per_component"])\
                and ("_cache" not in kwargs or kwargs["_cache"]):
            help_dict = kwargs["help_dict"]
        else:
            help_dict = {}
            kwargs.update({"_area": AreaType.Segmentation, "_per_component": PerComponent.No})
        volume_hash_str = hash_fun_call_name(Volume, {}, kwargs["_area"], kwargs["_per_component"])
        if volume_hash_str not in help_dict:
            volume = Volume.calculate_property(**kwargs)
            help_dict[volume_hash_str] = volume
        else:
            volume = help_dict[volume_hash_str]

        diameter_hash_str = hash_fun_call_name(Diameter, {}, kwargs["_area"], kwargs["_per_component"])
        if diameter_hash_str not in help_dict:
            diameter_val = Diameter.calculate_property(**kwargs)
            help_dict[diameter_hash_str] = diameter_val
        else:
            diameter_val = help_dict[diameter_hash_str]
        radius = diameter_val / 2
        if kwargs["area_array"].shape[0] > 1:
            return volume / (4/3 * pi * (radius ** 3))
        else:
            return volume / (pi * (radius **2))

    @classmethod
    def get_units(cls, ndim):
        return Volume.get_units(ndim) / Diameter.get_units(ndim) ** 3


class Surface(StatisticMethodBase):
    text_info = "Surface", "Calculating surface of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):
        return calculate_volume_surface(area_array, [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 2


class RimVolume(StatisticMethodBase):
    text_info = "Rim Volume", "Calculate volumes for elements in radius (in physical units) from mask"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 10000), property_type=float),
                AlgorithmProperty("units", "Units", Units.nm, property_type=Units)]

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
        return symbols("{}") ** 3


class RimPixelBrightnessSum(StatisticMethodBase):
    text_info = "Rim Pixel Brightness Sum", \
                "Calculate mass for components located within rim (in physical units) from mask"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 10000), property_type=float),
                AlgorithmProperty("units", "Units", Units.nm, property_type=Units)]

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
    surf_im: np.ndarray = np.array(volume_mask).astype(np.uint8).squeeze()
    for ax in range(surf_im.ndim):
        border_surface += \
            np.count_nonzero(
                np.logical_xor(surf_im.take(np.arange(surf_im.shape[ax]-1), axis=ax),
                               surf_im.take(np.arange(surf_im.shape[ax]-1)+1, axis=ax))
            ) * reduce(lambda x, y: x*y, [voxel_size[x] for x in range(surf_im.ndim) if x != ax])
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
                          MinimumPixelBrightness, MeanPixelBrightness, MedianPixelBrightness,
                          StandardDeviationOfPixelBrightness, MomentOfInertia, LongestMainAxisLength,
                          MiddleMainAxisLength, ShortestMainAxisLength, Compactness, Sphericity, Surface, RimVolume,
                          RimPixelBrightnessSum)
