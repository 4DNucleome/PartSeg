import traceback
from collections import OrderedDict
from enum import Enum
from functools import reduce
from math import pi
from typing import NamedTuple, Optional, Dict, Callable, List, Any, Union, Tuple, MutableMapping, Iterator

import SimpleITK
import numpy as np
from scipy.spatial.distance import cdist
from sympy import symbols

from .. import autofit as af
from ..channel_class import Channel
from ..algorithm_describe_base import Register, AlgorithmProperty
from ..mask_partition_utils import BorderRim, SplitMaskOnPart
from ..class_generator import enum_register
from ..universal_const import UNIT_SCALE, Units
from ..utils import class_to_dict
from .measurement_base import Leaf, Node, MeasurementEntry, MeasurementMethodBase, PerComponent, AreaType


# TODO change image to channel in signature of measurement calculate_property


class ProhibitedDivision(Exception):
    pass


class SettingsValue(NamedTuple):
    function: Callable
    help_message: str
    arguments: Optional[dict]
    is_component: bool
    default_area: Optional[AreaType] = None


class ComponentsInfo(NamedTuple):
    segmentation_components: np.ndarray
    mask_components: np.ndarray
    components_translation: Dict[int, List[int]]


def empty_fun(_a0=None, _a1=None):
    pass


MeasurementValueType = Union[float, List[float], str]
MeasurementResultType = Tuple[MeasurementValueType, str]
MeasurementResultInputType = Tuple[MeasurementValueType, str, Tuple[PerComponent, AreaType]]


class MeasurementResult(MutableMapping[str, MeasurementResultType]):
    """
    Class for storage measurements info.
    """

    def __init__(self, components_info: ComponentsInfo):
        self.components_info = components_info
        self._data_dict = OrderedDict()
        self._units_dict: Dict[str, str] = dict()
        self._type_dict: Dict[str, Tuple[PerComponent, AreaType]] = dict()
        self._units_dict["Mask component"] = ""
        self._units_dict["Segmentation component"] = ""

    def __str__(self):
        text = ""
        for key, val in self._data_dict.items():
            text += f"{key}: {val}; type {self._type_dict[key]}, units {self._units_dict[key]}\n"
        return text

    def __setitem__(self, k: str, v: MeasurementResultInputType) -> None:

        self._data_dict[k] = v[0]
        self._units_dict[k] = v[1]
        self._type_dict[k] = v[2]
        if k == "File name":
            self._data_dict.move_to_end("File name", False)

    def __delitem__(self, v: str) -> None:
        del self._data_dict[v]
        del self._units_dict[v]
        del self._type_dict[v]

    def __getitem__(self, k: str) -> MeasurementResultType:
        return self._data_dict[k], self._units_dict[k]

    def __len__(self) -> int:
        return len(self._data_dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data_dict)

    def set_filename(self, path_fo_file: str):
        """
        Set name of file to be presented as first position.
        """
        self._data_dict["File name"] = path_fo_file
        self._type_dict["File name"] = PerComponent.No, AreaType.Segmentation
        self._units_dict["File name"] = ""
        self._data_dict.move_to_end("File name", False)

    def get_component_info(self) -> Tuple[bool, bool]:
        """
        Get information which type of components are in storage.

        :return: has_mask_components, has_segmentation_components
        """
        has_mask_components = any(
            [x == PerComponent.Yes and y != AreaType.Segmentation for x, y in self._type_dict.values()]
        )
        has_segmentation_components = any(
            [x == PerComponent.Yes and y == AreaType.Segmentation for x, y in self._type_dict.values()]
        )
        return has_mask_components, has_segmentation_components

    def get_labels(self) -> List[str]:
        """Get labels for measurement. Base are keys of this storage.
        If has mask components, or has segmentation_components then add this labels"""
        has_mask_components, has_segmentation_components = self.get_component_info()
        labels = list(self._data_dict.keys())
        index = 1 if "File name" in self._data_dict else 0
        if has_mask_components:
            labels.insert(index, "Mask component")
        if has_segmentation_components:
            labels.insert(index, "Segmentation component")
        return labels

    def get_units(self) -> List[str]:
        return [self._units_dict[x] for x in self.get_labels()]

    def get_global_names(self):
        """Get names for only parameters which are not 'PerComponent.Yes'"""
        labels = list(self._data_dict.keys())
        return [x for x in labels if self._type_dict[x] != PerComponent.Yes]

    def get_global_parameters(self):
        """Get only parameters which are not 'PerComponent.Yes'"""
        if "File name" in self._data_dict:
            name = self._data_dict["File name"]
            res = [name]
            iterator = iter(self._data_dict.keys())
            next(iterator)
        else:
            res = []
            iterator = iter(self._data_dict.keys())
        for el in iterator:
            per_comp = self._type_dict[el][0]
            val = self._data_dict[el]
            if per_comp != PerComponent.Yes:
                res.append(val)
        return res

    def get_separated(self) -> List[List[MeasurementValueType]]:
        """Get measurements separated for each component"""
        has_mask_components, has_segmentation_components = self.get_component_info()
        if not (has_mask_components or has_segmentation_components):
            return [list(self._data_dict.values())]
        if has_mask_components and has_segmentation_components:
            translation = self.components_info.components_translation
            component_info = [(x, y) for x in translation.keys() for y in translation[x]]
        elif has_mask_components:
            component_info = [(0, x) for x in self.components_info.mask_components]
        else:
            component_info = [(x, 0) for x in self.components_info.segmentation_components]
        counts = len(component_info)
        mask_to_pos = {val: i for i, val in enumerate(self.components_info.mask_components)}
        segmentation_to_pos = {val: i for i, val in enumerate(self.components_info.segmentation_components)}
        if "File name" in self._data_dict:
            name = self._data_dict["File name"]
            res = [[name] for _ in range(counts)]
            iterator = iter(self._data_dict.keys())
            next(iterator)
        else:
            res = [[] for _ in range(counts)]
            iterator = iter(self._data_dict.keys())
        if has_segmentation_components:
            for i, num in enumerate(component_info):
                res[i].append(num[0])
        if has_mask_components:
            for i, num in enumerate(component_info):
                res[i].append(num[1])
        for el in iterator:
            per_comp, area_type = self._type_dict[el]
            val = self._data_dict[el]
            if per_comp != PerComponent.Yes:
                for i in range(counts):
                    res[i].append(val)
            else:
                if area_type == AreaType.Segmentation:
                    for i, (seg, _mask) in enumerate(component_info):
                        res[i].append(val[segmentation_to_pos[seg]])
                else:
                    for i, (_seg, mask) in enumerate(component_info):
                        res[i].append(val[mask_to_pos[mask]])
        return res


class MeasurementProfile(object):
    PARAMETERS = ["name", "chosen_fields", "reversed_brightness", "use_gauss_image", "name_prefix"]

    def __init__(self, name, chosen_fields: List[MeasurementEntry], name_prefix=""):
        self.name = name
        self.chosen_fields: List[MeasurementEntry] = chosen_fields
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

    def _get_par_component_and_area_type(self, tree: Union[Node, Leaf]) -> Tuple[PerComponent, AreaType]:
        if isinstance(tree, Leaf):
            method = MEASUREMENT_DICT[tree.name]
            area_type = method.area_type(tree.area)
            if tree.per_component == PerComponent.Mean:
                return PerComponent.No, area_type
            return tree.per_component, area_type
        else:
            left_par, left_area = self._get_par_component_and_area_type(tree.left)
            right_par, right_area = self._get_par_component_and_area_type(tree.left)
            if PerComponent.Yes == left_par or PerComponent.Yes == right_par:
                res_par = PerComponent.Yes
            else:
                res_par = PerComponent.No
            area_set = {left_area, right_area}
            if len(area_set) == 1:
                res_area = area_set.pop()
            elif AreaType.Segmentation in area_set:
                res_area = AreaType.Segmentation
            else:
                res_area = AreaType.Mask_without_segmentation
            return res_par, res_area

    def get_channels_num(self):
        resp = set()
        for el in self.chosen_fields:
            resp.update(el.get_channel_num(MEASUREMENT_DICT))
        return resp

    def __str__(self):
        text = "Set name: {}\n".format(self.name)
        if self.name_prefix != "":
            text += "Name prefix: {}\n".format(self.name_prefix)
        text += "Measurements list:\n"
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
            res.append(
                (
                    (self.name_prefix + el.name, el.get_unit(unit, 3)),
                    self._is_component_measurement(el.calculation_tree),
                )
            )
        return res

    def get_parameters(self):
        return class_to_dict(self, *self.PARAMETERS)

    def is_any_mask_measurement(self):
        for el in self.chosen_fields:
            if self.need_mask(el.calculation_tree):
                return True
        return False

    def _is_component_measurement(self, node):
        if isinstance(node, Leaf):
            return node.per_component == PerComponent.Yes
        else:
            return self._is_component_measurement(node.left) or self._is_component_measurement(node.right)

    def calculate_tree(
        self, node: Union[Node, Leaf], segmentation_mask_map: ComponentsInfo, help_dict: dict, kwargs: dict
    ) -> Tuple[Union[float, np.ndarray], symbols, AreaType]:
        """
        Main function for calculation tree of measurements. It is executed recursively

        :param node: measurement to calculate
        :param segmentation_mask_map: map from mask segmentation components to mask components. Needed for division
        :param help_dict: dict to cache calculation result. It reduce recalculations of same measurements.
        :param kwargs: additional info needed by measurements
        :return: measurement value
        """
        if isinstance(node, Leaf):
            method = MEASUREMENT_DICT[node.name]
            kw = dict(kwargs)
            kw.update(node.dict)
            hash_str = hash_fun_call_name(method, node.dict, node.area, node.per_component, node.channel)
            area_type = method.area_type(node.area)
            if hash_str in help_dict:
                val = help_dict[hash_str]
            else:
                if node.channel is not None:
                    kw["channel"] = kw[f"chanel_{node.channel}"]
                    kw["channel_num"] = node.channel
                else:
                    kw["channel_num"] = -1
                kw["help_dict"] = help_dict
                kw["_area"] = node.area
                kw["_per_component"] = node.per_component
                kw["_cache"] = True
                if area_type == AreaType.Mask:
                    kw["area_array"] = kw["mask"]
                elif area_type == AreaType.Mask_without_segmentation:
                    kw["area_array"] = kw["mask_without_segmentation"]
                elif area_type == AreaType.Segmentation:
                    kw["area_array"] = kw["segmentation"]
                else:
                    raise ValueError(f"Unknown area type {node.area}")
                if node.per_component != PerComponent.No:
                    kw["_cache"] = False
                    val = []
                    area_array = kw["area_array"]
                    if area_type == AreaType.Segmentation:
                        components = segmentation_mask_map.segmentation_components
                    else:
                        components = segmentation_mask_map.mask_components
                    for i in components:
                        kw["area_array"] = area_array == i
                        val.append(method.calculate_property(**kw))
                    if node.per_component == PerComponent.Mean:
                        val = np.mean(val)
                    else:
                        val = np.array(val)
                else:
                    val = method.calculate_property(**kw)
                help_dict[hash_str] = val
            unit: symbols = method.get_units(3) if kw["channel"].shape[1] > 1 else method.get_units(2)
            if node.power != 1:
                return pow(val, node.power), pow(unit, node.power), area_type
            return val, unit, area_type
        elif isinstance(node, Node):
            left_res, left_unit, left_area = self.calculate_tree(node.left, segmentation_mask_map, help_dict, kwargs)
            right_res, right_unit, right_area = self.calculate_tree(
                node.right, segmentation_mask_map, help_dict, kwargs
            )
            if node.op == "/":
                if isinstance(left_res, np.ndarray) and isinstance(right_res, np.ndarray) and left_area != right_area:
                    area_set = {left_area, right_area}
                    if area_set == {AreaType.Segmentation, AreaType.Mask_without_segmentation}:
                        raise ProhibitedDivision("This division is prohibited")
                    if area_set == {AreaType.Segmentation, AreaType.Mask}:
                        res = []
                        # TODO Test this part of code
                        for val, num in zip(left_res, segmentation_mask_map.segmentation_components):
                            div_vals = segmentation_mask_map.components_translation[num]
                            if len(div_vals) != 1:
                                raise ProhibitedDivision("Cannot calculate when object do not belongs to one mask area")
                            if left_area == AreaType.Segmentation:
                                res.append(val / right_res[div_vals[0] - 1])
                            else:
                                res.append(right_res[div_vals[0] - 1] / val)
                        return np.array(res), left_unit / right_unit, AreaType.Segmentation
                    left_area = AreaType.Mask_without_segmentation

                return left_res / right_res, left_unit / right_unit, left_area
        raise ValueError("Wrong measurement: {}".format(node))

    @staticmethod
    def get_segmentation_to_mask_component(segmentation: np.ndarray, mask: Optional[np.ndarray]) -> ComponentsInfo:
        """
        Calculate map from segmentation component num to mask component num

        :param segmentation: numpy array with segmentation labeled as positive integers
        :param mask: numpy array with mask labeled as positive integer
        :return: map
        """
        components = np.unique(segmentation)
        if components[0] == 0 or components[0] is None:
            components = components[1:]
        mask_components = np.unique(mask)
        if mask_components[0] == 0 or mask_components[0] is None:
            mask_components = mask_components[1:]
        res = OrderedDict()
        if mask is None:
            res = {i: [] for i in components}
        elif np.max(mask) == 1:
            res = {i: [1] for i in components}
        else:
            for num in components:
                res[num] = list(np.unique(mask[segmentation == num]))
        return ComponentsInfo(components, mask_components, res)

    def get_component_and_area_info(self) -> List[Tuple[PerComponent, AreaType]]:
        """For each measurement check if is per component and in which types """
        res = []
        for el in self.chosen_fields:
            tree = el.calculation_tree
            res.append(self._get_par_component_and_area_type(tree))
        return res

    def calculate(
        self,
        channel: np.ndarray,
        segmentation: np.ndarray,
        full_mask: np.ndarray,
        mask: Optional[np.ndarray],
        voxel_size,
        result_units: Units,
        range_changed: Callable[[int, int], Any] = None,
        step_changed: Callable[[int], Any] = None,
        **kwargs,
    ) -> MeasurementResult:
        """
        Calculate measurements on given set of parameters

        :param channel: main channel on which measurements should be calculated
        :param segmentation: array with segmentation labeled as positive
        :param full_mask:
        :param mask:
        :param voxel_size:
        :param result_units:
        :param range_changed: callback function to set information about steps range
        :param step_changed: callback function fo set information about steps done
        :param kwargs: additional data required by measurements. Ex additional channels
        :return: measurements
        """
        if range_changed is None:
            range_changed = empty_fun
        if step_changed is None:
            step_changed = empty_fun
        if self._need_mask and mask is None:
            raise ValueError("measurement need mask")
        channel = channel.astype(np.float)
        help_dict = dict()
        segmentation_mask_map = self.get_segmentation_to_mask_component(segmentation, mask)
        result = MeasurementResult(segmentation_mask_map)
        result_scalar = UNIT_SCALE[result_units.value]
        kw = {
            "channel": channel,
            "segmentation": segmentation,
            "mask": mask,
            "full_segmentation": full_mask,
            "voxel_size": voxel_size,
            "result_scalar": result_scalar,
        }
        for el in kwargs.keys():
            if not el.startswith("channel_"):
                raise ValueError(f"unknown parameter {el} of calculate function")
        for num in self.get_channels_num():
            if f"channel_{num}" not in kwargs:
                raise ValueError(f"channel_{num} need to be passed as argument of calculate function")
        kw.update(kwargs)
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
            component_and_area = self._get_par_component_and_area_type(tree)
            try:
                val, unit, _area = self.calculate_tree(tree, segmentation_mask_map, help_dict, kw)
                if isinstance(val, np.ndarray):
                    val = list(val)
                result[self.name_prefix + user_name] = val, str(unit).format(str(result_units)), component_and_area
            except ZeroDivisionError:
                result[self.name_prefix + user_name] = "Div by zero", "", component_and_area
            except TypeError:
                traceback.print_exc()
                result[self.name_prefix + user_name] = "None div", "", component_and_area
            except AttributeError:
                result[self.name_prefix + user_name] = "No attribute", "", component_and_area
            except ProhibitedDivision as e:
                result[self.name_prefix + user_name] = e.args[0], "", component_and_area
        return result


def calculate_main_axis(area_array: np.ndarray, channel: np.ndarray, voxel_size):
    # TODO check if it produces good values
    if len(channel.shape) == 4:
        if channel.shape[0] != 1:
            raise ValueError("This measurements do not support time data")
        channel = channel[0]
    cut_img = np.copy(channel)
    cut_img[area_array == 0] = 0
    if np.all(cut_img == 0):
        return (0,) * len(voxel_size)
    orientation_matrix, _ = af.find_density_orientation(cut_img, voxel_size, 1)
    center_of_mass = af.density_mass_center(cut_img, voxel_size)
    positions = np.array(np.nonzero(cut_img), dtype=np.float64)
    for i, v in enumerate(reversed(voxel_size), start=1):
        positions[-i] *= v
        positions[-i] -= center_of_mass[i - 1]
    centered = np.dot(orientation_matrix.T, positions)
    size = np.max(centered, axis=1) - np.min(centered, axis=1)
    return size


def get_main_axis_length(
    index: int, area_array: np.ndarray, channel: np.ndarray, voxel_size, result_scalar, _cache=False, **kwargs
):
    _cache = _cache and "_area" in kwargs and "_per_component" in kwargs
    if _cache:
        help_dict: Dict = kwargs["help_dict"]
        _area: AreaType = kwargs["_area"]
        _per_component: PerComponent = kwargs["_per_component"]
        hash_name = hash_fun_call_name(calculate_main_axis, {}, _area, _per_component, kwargs["channel_num"])
        if hash_name not in help_dict:
            help_dict[hash_name] = calculate_main_axis(area_array, channel, [x * result_scalar for x in voxel_size])
        return help_dict[hash_name][index]
    else:
        return calculate_main_axis(area_array, channel, [x * result_scalar for x in voxel_size])[index]


def hash_fun_call_name(fun: Callable, arguments: Dict, area: AreaType, per_component: PerComponent, channel: Channel):
    if hasattr(fun, "__module__"):
        fun_name = f"{fun.__module__}.{fun.__name__}"
    else:
        fun_name = fun.__name__
    return "{}: {} # {} & {} * {}".format(fun_name, arguments, area, per_component, channel)


class Volume(MeasurementMethodBase):
    text_info = "Volume", "Calculate volume of current segmentation"

    @classmethod
    def calculate_property(cls, area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
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


class Diameter(MeasurementMethodBase):
    text_info = "Diameter", "Diameter of area"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
        pos = np.transpose(np.nonzero(get_border(area_array))).astype(np.float)
        if pos.size == 0:
            return 0
        for i, val in enumerate([x * result_scalar for x in reversed(voxel_size)], start=1):
            pos[:, -i] *= val
        diam_sq = iterative_double_normal(pos)[0]
        return np.sqrt(diam_sq)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class DiameterOld(MeasurementMethodBase):
    text_info = "Diameter old", "Diameter of area (Very slow)"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
        return calc_diam(get_border(area_array), [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class PixelBrightnessSum(MeasurementMethodBase):
    text_info = "Pixel Brightness Sum", "Sum of pixel brightness for current segmentation"

    @staticmethod
    def calculate_property(area_array: np.ndarray, channel: np.ndarray, **_):  # pylint: disable=W0221
        """
        :param area_array: mask for area
        :param channel: data. same shape like area_type
        :return: Pixels brightness sum on given area
        """
        if area_array.shape != channel.shape:
            if area_array.size == channel.size:
                channel = channel.reshape(area_array.shape)
            else:
                raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.sum(channel[area_array > 0])
        return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class ComponentsNumber(MeasurementMethodBase):
    text_info = "Components Number", "Calculate number of connected components on segmentation"

    @staticmethod
    def calculate_property(area_array, **_):  # pylint: disable=W0221
        return np.unique(area_array).size - 1

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(cls.text_info[0], per_component=PerComponent.No)

    @classmethod
    def get_units(cls, ndim):
        return symbols("count")


class MaximumPixelBrightness(MeasurementMethodBase):
    text_info = "Maximum pixel brightness", "Calculate maximum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):
        if area_array.shape != channel.shape:
            if area_array.size == channel.size:
                channel = channel.reshape(area_array.shape)
            else:
                raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.max(channel[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class MinimumPixelBrightness(MeasurementMethodBase):
    text_info = "Minimum pixel brightness", "Calculate minimum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):
        if area_array.shape != channel.shape:
            if area_array.size == channel.size:
                channel = channel.reshape(area_array.shape)
            else:
                raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.min(channel[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class MeanPixelBrightness(MeasurementMethodBase):
    text_info = "Mean pixel brightness", "Calculate mean pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):  # pylint: disable=W0221
        if area_array.shape != channel.shape:
            if area_array.size == channel.size:
                channel = channel.reshape(area_array.shape)
            else:
                raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.mean(channel[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class MedianPixelBrightness(MeasurementMethodBase):
    text_info = "Median pixel brightness", "Calculate median pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):  # pylint: disable=W0221
        if area_array.shape != channel.shape:
            if area_array.size == channel.size:
                channel = channel.reshape(area_array.shape)
            else:
                raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.median(channel[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class StandardDeviationOfPixelBrightness(MeasurementMethodBase):
    text_info = (
        "Standard deviation of pixel brightness",
        "Calculate standard deviation of pixel brightness for current area",
    )

    @staticmethod
    def calculate_property(area_array, channel, **_):  # pylint: disable=W0221
        if area_array.shape != channel.shape:
            if area_array.size == channel.size:
                channel = channel.reshape(area_array.shape)
            else:
                raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.std(channel[area_array > 0])
        else:
            return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class MomentOfInertia(MeasurementMethodBase):
    text_info = "Moment of inertia", "Calculate moment of inertia for segmented structure"

    @staticmethod
    def calculate_property(area_array, channel, voxel_size, **_):  # pylint: disable=W0221
        if len(channel.shape) == 4:
            if channel.shape[0] != 1:
                raise ValueError("This measurements do not support time data")
            channel = channel[0]
        if channel.ndim != 3:
            return None
        img = np.copy(channel)
        img[area_array == 0] = 0
        if np.all(img == 0):
            return 0
        return af.calculate_density_momentum(img, voxel_size)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 2 * symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class LongestMainAxisLength(MeasurementMethodBase):
    text_info = "Longest main axis length", "Length of first main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(0, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @classmethod
    def need_channel(cls):
        return True


class MiddleMainAxisLength(MeasurementMethodBase):
    text_info = "Middle main axis length", "Length of second main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(1, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @classmethod
    def need_channel(cls):
        return True


class ShortestMainAxisLength(MeasurementMethodBase):
    text_info = "Shortest main axis length", "Length of third main axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(2, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @classmethod
    def need_channel(cls):
        return True


class Compactness(MeasurementMethodBase):
    text_info = "Compactness", "Calculate compactness off segmentation (Surface^1.5/volume)"

    @staticmethod
    def calculate_property(**kwargs):
        cache = kwargs["_cache"] if "_cache" in kwargs else False
        cache = cache and "help_dict" in kwargs
        cache = cache and "_area" in kwargs
        cache = cache and "_per_component" in kwargs
        if cache:
            help_dict = kwargs["help_dict"]
            border_hash_str = hash_fun_call_name(Surface, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1))
            if border_hash_str not in help_dict:
                border_surface = Surface.calculate_property(**kwargs)
                help_dict[border_hash_str] = border_surface
            else:
                border_surface = help_dict[border_hash_str]

            volume_hash_str = hash_fun_call_name(Volume, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1))

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


class Sphericity(MeasurementMethodBase):
    text_info = "Sphericity", "volume/((4/3 * π * radius **3) for 3d data and volume/((π * radius **2) for 2d data"

    @staticmethod
    def calculate_property(**kwargs):
        if all(key in kwargs for key in ["help_dict", "_area", "_per_component"]) and (
            "_cache" not in kwargs or kwargs["_cache"]
        ):
            help_dict = kwargs["help_dict"]
        else:
            help_dict = {}
            kwargs.update({"_area": AreaType.Segmentation, "_per_component": PerComponent.No})
        volume_hash_str = hash_fun_call_name(Volume, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1))
        if volume_hash_str not in help_dict:
            volume = Volume.calculate_property(**kwargs)
            help_dict[volume_hash_str] = volume
        else:
            volume = help_dict[volume_hash_str]

        diameter_hash_str = hash_fun_call_name(Diameter, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1))
        if diameter_hash_str not in help_dict:
            diameter_val = Diameter.calculate_property(**kwargs)
            help_dict[diameter_hash_str] = diameter_val
        else:
            diameter_val = help_dict[diameter_hash_str]
        radius = diameter_val / 2
        if kwargs["area_array"].shape[0] > 1:
            return volume / (4 / 3 * pi * (radius ** 3))
        else:
            return volume / (pi * (radius ** 2))

    @classmethod
    def get_units(cls, ndim):
        return Volume.get_units(ndim) / Diameter.get_units(ndim) ** 3


class Surface(MeasurementMethodBase):
    text_info = "Surface", "Calculating surface of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
        return calculate_volume_surface(area_array, [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 2


class RimVolume(MeasurementMethodBase):
    text_info = "Rim Volume", "Calculate volumes for elements in radius (in physical units) from mask"

    @classmethod
    def get_fields(cls):
        return BorderRim.get_fields()

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask)

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **kwargs):  # pylint: disable=W0221
        border_mask_array = BorderRim.border_mask(voxel_size=voxel_size, result_scalar=result_scalar, **kwargs)
        if border_mask_array is None:
            return None
        final_mask = np.array((border_mask_array > 0) * (area_array > 0))
        return np.count_nonzero(final_mask) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 3

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.Segmentation


class RimPixelBrightnessSum(MeasurementMethodBase):
    text_info = (
        "Rim Pixel Brightness Sum",
        "Calculate mass for components located within rim (in physical units) from mask",
    )

    @classmethod
    def get_fields(cls):
        return BorderRim.get_fields()

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask)

    @staticmethod
    def calculate_property(channel, area_array, **kwargs):  # pylint: disable=W0221
        if len(channel.shape) == 4:
            if channel.shape[0] != 1:
                raise ValueError("This measurements do not support time data")
            channel = channel[0]
        border_mask_array = BorderRim.border_mask(**kwargs)
        if border_mask_array is None:
            return None
        final_mask = np.array((border_mask_array > 0) * (area_array > 0))
        if np.any(final_mask):
            return np.sum(channel[final_mask])
        return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.Segmentation


class DistancePoint(Enum):
    Border = 1
    Mass_center = 2
    Geometrical_center = 3

    def __str__(self):
        return self.name.replace("_", " ")


try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    reloading
except NameError:
    reloading = False
    enum_register.register_class(DistancePoint)


class DistanceMaskSegmentation(MeasurementMethodBase):
    text_info = "segmentation distance", "Calculate distance between segmentation and mask"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("distance_from_mask", "Distance from mask", DistancePoint.Border),
            AlgorithmProperty("distance_to_segmentation", "Distance to segmentation", DistancePoint.Border),
        ]

    @staticmethod
    def calculate_points(channel, area_array, voxel_size, result_scalar, point_type: DistancePoint) -> np.ndarray:
        if point_type == DistancePoint.Border:
            area_pos = np.transpose(np.nonzero(get_border(area_array))).astype(np.float)
            area_pos += 0.5
            for i, val in enumerate([x * result_scalar for x in reversed(voxel_size)], start=1):
                area_pos[:, -i] *= val
        elif point_type == DistancePoint.Mass_center:
            im = np.copy(channel)
            im[area_array == 0] = 0
            area_pos = np.array([af.density_mass_center(im, voxel_size) * result_scalar])
        else:
            area_pos = np.array([af.density_mass_center(area_array > 0, voxel_size) * result_scalar])
        return area_pos

    @classmethod
    def calculate_property(
        cls,
        channel,
        area_array,
        mask,
        voxel_size,
        result_scalar,
        distance_from_mask: DistancePoint,
        distance_to_segmentation: DistancePoint,
        *args,
        **kwargs,
    ):  # pylint: disable=W0221
        if len(channel.shape) == 4:
            if channel.shape[0] != 1:
                raise ValueError("This measurements do not support time data")
            channel = channel[0]
        if not (np.any(mask) and np.any(area_array)):
            return 0
        mask_pos = cls.calculate_points(channel, mask, voxel_size, result_scalar, distance_from_mask)
        seg_pos = cls.calculate_points(channel, area_array, voxel_size, result_scalar, distance_to_segmentation)
        if mask_pos.shape[0] == 1 or seg_pos.shape[0] == 1:
            return np.min(cdist(mask_pos, seg_pos))
        else:
            min_val = np.inf
            for i in range(seg_pos.shape[0]):
                min_val = min(min_val, np.min(cdist(mask_pos, np.array([seg_pos[i]]))))
            return min_val

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @classmethod
    def need_channel(cls):
        return True

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.Segmentation


class SplitOnPartVolume(MeasurementMethodBase):
    text_info = (
        "split on part volume",
        "Split mask on parts and then calculate volume of cross " "of segmentation and mask part",
    )

    @classmethod
    def get_fields(cls):
        return SplitMaskOnPart.get_fields() + [
            AlgorithmProperty("part_selection", "Which part  (from border)", 2, (1, 1024))
        ]

    @staticmethod
    def calculate_property(part_selection, area_array, voxel_size, result_scalar, **kwargs):  # pylint: disable=W0221
        masked = SplitMaskOnPart.split(voxel_size=voxel_size, **kwargs)
        mask = masked == part_selection
        return np.count_nonzero(mask * area_array) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 3

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask)

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.Segmentation


class SplitOnPartPixelBrightnessSum(MeasurementMethodBase):
    text_info = (
        "split on part pixel brightness sum",
        "Split mask on parts and then calculate pixel brightness sum" " of cross of segmentation and mask part",
    )

    @classmethod
    def get_fields(cls):
        return SplitMaskOnPart.get_fields() + [
            AlgorithmProperty("part_selection", "Which part (from border)", 2, (1, 1024))
        ]

    @staticmethod
    def calculate_property(part_selection, channel, area_array, **kwargs):  # pylint: disable=W0221
        masked = SplitMaskOnPart.split(**kwargs)
        mask = np.array(masked == part_selection)
        if channel.ndim - mask.ndim == 1:
            channel = channel[0]
        return np.sum(channel[mask * area_array > 0])

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask)

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.Segmentation


def pixel_volume(spacing, result_scalar):
    return reduce((lambda x, y: x * y), [x * result_scalar for x in spacing])


def calculate_volume_surface(volume_mask, voxel_size):
    border_surface = 0
    surf_im: np.ndarray = np.array(volume_mask).astype(np.uint8).squeeze()
    for ax in range(surf_im.ndim):
        border_surface += np.count_nonzero(
            np.logical_xor(
                surf_im.take(np.arange(surf_im.shape[ax] - 1), axis=ax),
                surf_im.take(np.arange(surf_im.shape[ax] - 1) + 1, axis=ax),
            )
        ) * reduce(lambda x, y: x * y, [voxel_size[x] for x in range(surf_im.ndim) if x != ax])
    return border_surface


def get_border(array):
    if array.dtype == np.bool:
        array = array.astype(np.uint8)
    return SimpleITK.GetArrayFromImage(SimpleITK.LabelContour(SimpleITK.GetImageFromArray(array)))


def calc_diam(array, voxel_size):
    pos = np.transpose(np.nonzero(array)).astype(np.float)
    for i, val in enumerate(voxel_size):
        pos[:, i] *= val
    diam = 0
    for i, p in enumerate(zip(pos[:-1])):
        tmp = np.array((pos[i + 1 :] - p) ** 2)
        diam = max(diam, np.max(np.sum(tmp, 1)))
    return np.sqrt(diam)


MEASUREMENT_DICT = Register(
    Volume,
    Diameter,
    PixelBrightnessSum,
    ComponentsNumber,
    MaximumPixelBrightness,
    MinimumPixelBrightness,
    MeanPixelBrightness,
    MedianPixelBrightness,
    StandardDeviationOfPixelBrightness,
    MomentOfInertia,
    LongestMainAxisLength,
    MiddleMainAxisLength,
    ShortestMainAxisLength,
    Compactness,
    Sphericity,
    Surface,
    RimVolume,
    RimPixelBrightnessSum,
    DistanceMaskSegmentation,
    SplitOnPartVolume,
    SplitOnPartPixelBrightnessSum,
    suggested_base_class=MeasurementMethodBase,
)
"""Register with all measurements algorithms"""
