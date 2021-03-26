from collections import OrderedDict
from enum import Enum
from functools import reduce
from math import pi
from typing import Any, Callable, Dict, Iterator, List, MutableMapping, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import SimpleITK
from mahotas.features import haralick
from scipy.spatial.distance import cdist
from sympy import symbols

from PartSegImage import Image

from .. import autofit as af
from ..algorithm_describe_base import AlgorithmProperty, Register
from ..channel_class import Channel
from ..class_generator import enum_register
from ..mask_partition_utils import BorderRim, MaskDistanceSplit
from ..roi_info import ROIInfo
from ..universal_const import UNIT_SCALE, Units
from .measurement_base import AreaType, Leaf, MeasurementEntry, MeasurementMethodBase, Node, PerComponent

# TODO change image to channel in signature of measurement calculate_property

NO_COMPONENT = -1


class ProhibitedDivision(Exception):
    pass


class SettingsValue(NamedTuple):
    function: Callable
    help_message: str
    arguments: Optional[dict]
    is_component: bool
    default_area: Optional[AreaType] = None


class ComponentsInfo(NamedTuple):
    """
    Class for storage information about relation between roi components and mask components

    :ivar numpy.ndarray roi_components: list of roi components
    :ivar numpy.ndarray mask_components: list of mask components
    :ivar Dict[int, List[int]] components_translation: mapping
        from roi components to mask components base on intersections
    """

    roi_components: np.ndarray
    mask_components: np.ndarray
    components_translation: Dict[int, List[int]]


def empty_fun(_a0=None, _a1=None):
    """This function  is be used as dummy reporting function."""


MeasurementValueType = Union[float, List[float], str]
MeasurementResultType = Tuple[MeasurementValueType, str]
MeasurementResultInputType = Tuple[MeasurementValueType, str, Tuple[PerComponent, AreaType]]


FILE_NAME_STR = "File name"


class MeasurementResult(MutableMapping[str, MeasurementResultType]):
    """
    Class for storage measurements info.
    """

    def __init__(self, components_info: ComponentsInfo):
        self.components_info = components_info
        self._data_dict = OrderedDict()
        self._units_dict: Dict[str, str] = {}
        self._type_dict: Dict[str, Tuple[PerComponent, AreaType]] = {}
        self._units_dict["Mask component"] = ""
        self._units_dict["Segmentation component"] = ""

    def __str__(self):  # pragma: no cover
        return "".join(
            f"{key}: {val}; type {self._type_dict[key]}, units {self._units_dict[key]}\n"
            for key, val in self._data_dict.items()
        )

    def __setitem__(self, k: str, v: MeasurementResultInputType) -> None:

        self._data_dict[k] = v[0]
        self._units_dict[k] = v[1]
        self._type_dict[k] = v[2]

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
        self._data_dict[FILE_NAME_STR] = path_fo_file
        self._type_dict[FILE_NAME_STR] = PerComponent.No, AreaType.ROI
        self._units_dict[FILE_NAME_STR] = ""
        self._data_dict.move_to_end(FILE_NAME_STR, False)

    def get_component_info(self) -> Tuple[bool, bool]:
        """
        Get information which type of components are in storage.

        :return: has_mask_components, has_segmentation_components
        """
        has_mask_components = any((x == PerComponent.Yes and y != AreaType.ROI for x, y in self._type_dict.values()))
        has_segmentation_components = any(
            (x == PerComponent.Yes and y == AreaType.ROI for x, y in self._type_dict.values())
        )
        return has_mask_components, has_segmentation_components

    def get_labels(self, expand=True) -> List[str]:
        """
        If expand is false return list of keys of this storage.
        Otherwise return  labels for measurement. Base are keys of this storage.
        If has mask components, or has segmentation_components then add this labels
        """

        if not expand:
            return list(self.keys())
        has_mask_components, has_segmentation_components = self.get_component_info()
        labels = list(self._data_dict.keys())
        index = 1 if FILE_NAME_STR in self._data_dict else 0
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
        return [x for x in labels if self._type_dict[x][0] != PerComponent.Yes]

    def get_global_parameters(self):
        """Get only parameters which are not 'PerComponent.Yes'"""
        if FILE_NAME_STR in self._data_dict:
            name = self._data_dict[FILE_NAME_STR]
            res = [name]
            iterator = iter(self._data_dict.keys())
            try:
                next(iterator)
            except StopIteration:
                pass
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
            component_info = [(x, 0) for x in self.components_info.roi_components]
        counts = len(component_info)
        mask_to_pos = {val: i for i, val in enumerate(self.components_info.mask_components)}
        segmentation_to_pos = {val: i for i, val in enumerate(self.components_info.roi_components)}
        if FILE_NAME_STR in self._data_dict:
            name = self._data_dict[FILE_NAME_STR]
            res = [[name] for _ in range(counts)]
            iterator = iter(self._data_dict.keys())
            try:
                next(iterator)
            except StopIteration:
                pass
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
                if area_type == AreaType.ROI:
                    for i, (seg, _mask) in enumerate(component_info):
                        res[i].append(val[segmentation_to_pos[seg]])
                else:
                    for i, (_seg, mask) in enumerate(component_info):
                        res[i].append(val[mask_to_pos[mask]])
        return res


class MeasurementProfile:
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
            return tree.area in [AreaType.Mask, AreaType.Mask_without_ROI]
        return self.need_mask(tree.left) or self.need_mask(tree.right)

    def _need_mask_without_segmentation(self, tree):
        if isinstance(tree, Leaf):
            return tree.area == AreaType.Mask_without_ROI
        return self._need_mask_without_segmentation(tree.left) or self._need_mask_without_segmentation(tree.right)

    def _get_par_component_and_area_type(self, tree: Union[Node, Leaf]) -> Tuple[PerComponent, AreaType]:
        if isinstance(tree, Leaf):
            method = MEASUREMENT_DICT[tree.name]
            area_type = method.area_type(tree.area)
            if tree.per_component == PerComponent.Mean:
                return PerComponent.No, area_type
            return tree.per_component, area_type

        left_par, left_area = self._get_par_component_and_area_type(tree.left)
        right_par, right_area = self._get_par_component_and_area_type(tree.left)
        if PerComponent.Yes in [left_par, right_par]:
            res_par = PerComponent.Yes
        else:
            res_par = PerComponent.No
        area_set = {left_area, right_area}
        if len(area_set) == 1:
            res_area = area_set.pop()
        elif AreaType.ROI in area_set:
            res_area = AreaType.ROI
        else:
            res_area = AreaType.Mask_without_ROI
        return res_par, res_area

    def get_channels_num(self) -> Set[Channel]:
        resp = set()
        for el in self.chosen_fields:
            resp.update(el.get_channel_num(MEASUREMENT_DICT))
        return resp

    def __str__(self):
        text = f"Set name: {self.name}\n"
        if self.name_prefix != "":
            text += f"Name prefix: {self.name_prefix}\n"
        text += "Measurements list:\n"
        for el in self.chosen_fields:
            text += f"{el.name}\n"
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

    def is_any_mask_measurement(self):
        return any(self.need_mask(el.calculation_tree) for el in self.chosen_fields)

    def _is_component_measurement(self, node):
        if isinstance(node, Leaf):
            return node.per_component == PerComponent.Yes
        return self._is_component_measurement(node.left) or self._is_component_measurement(node.right)

    @staticmethod
    def _calculate_leaf_value(
        node: Union[Node, Leaf], segmentation_mask_map: ComponentsInfo, help_dict: dict, kwargs: dict
    ) -> Union[float, np.ndarray]:
        method: MeasurementMethodBase = MEASUREMENT_DICT[node.name]
        area_type = method.area_type(node.area)
        area_type_dict = {
            AreaType.Mask: "mask",
            AreaType.Mask_without_ROI: "mask_without_segmentation",
            AreaType.ROI: "segmentation",
        }
        kw = dict(kwargs)
        kw.update(node.dict)

        if node.channel is not None:
            kw["channel"] = kw[f"channel_{node.channel}"]
            kw["channel_num"] = node.channel
        else:
            kw["channel_num"] = -1
        kw["help_dict"] = help_dict
        kw["_area"] = node.area
        kw["_per_component"] = node.per_component
        kw["_cache"] = True  # TODO remove cache argument
        kw["area_array"] = kw[area_type_dict[area_type]]
        kw["_component_num"] = NO_COMPONENT
        if node.per_component == PerComponent.No:
            val = method.calculate_property(**kw)
        else:
            # TODO use cache for per component calculate
            # kw["_cache"] = False
            val = []
            area_array = kw["area_array"]
            if area_type == AreaType.ROI:
                components = segmentation_mask_map.roi_components
            else:
                components = segmentation_mask_map.mask_components
            for i in components:
                kw["area_array"] = area_array == i
                kw["_component_num"] = i
                val.append(method.calculate_property(**kw))
            val = np.array(val)
            if node.per_component == PerComponent.Mean:
                val = np.mean(val) if val.size else 0
        return val

    def _calculate_leaf(
        self, node: Leaf, segmentation_mask_map: ComponentsInfo, help_dict: dict, kwargs: dict
    ) -> Tuple[Union[float, np.ndarray], symbols, AreaType]:
        method: MeasurementMethodBase = MEASUREMENT_DICT[node.name]

        hash_str = hash_fun_call_name(method, node.dict, node.area, node.per_component, node.channel, NO_COMPONENT)
        area_type = method.area_type(node.area)
        if hash_str in help_dict:
            val = help_dict[hash_str]
        else:
            val = self._calculate_leaf_value(node, segmentation_mask_map, help_dict, kwargs)
            help_dict[hash_str] = val
        unit: symbols = method.get_units(3) if kwargs["image"].is_stack else method.get_units(2)
        if node.power != 1:
            return pow(val, node.power), pow(unit, node.power), area_type
        return val, unit, area_type

    def _calculate_node(
        self, node: Node, segmentation_mask_map: ComponentsInfo, help_dict: dict, kwargs: dict
    ) -> Tuple[Union[float, np.ndarray], symbols, AreaType]:
        left_res, left_unit, left_area = self.calculate_tree(node.left, segmentation_mask_map, help_dict, kwargs)
        right_res, right_unit, right_area = self.calculate_tree(node.right, segmentation_mask_map, help_dict, kwargs)
        if node.op != "/":
            raise ValueError(f"Wrong measurement: {node}")
        if isinstance(left_res, np.ndarray) and isinstance(right_res, np.ndarray) and left_area != right_area:
            area_set = {left_area, right_area}
            if area_set == {AreaType.ROI, AreaType.Mask_without_ROI}:  # pragma: no cover
                raise ProhibitedDivision("This division is prohibited")
            if area_set == {AreaType.ROI, AreaType.Mask}:
                res = []
                reverse = False
                if left_area == AreaType.Mask:
                    left_res, right_res = right_res, left_res
                    reverse = True
                for val, num in zip(left_res, segmentation_mask_map.roi_components):
                    div_vals = segmentation_mask_map.components_translation[num]
                    if len(div_vals) != 1:  # pragma: no cover
                        raise ProhibitedDivision("Cannot calculate when object do not belongs to one mask area")
                    if left_area == AreaType.ROI:
                        res.append(val / right_res[div_vals[0] - 1])
                    else:
                        res.append(right_res[div_vals[0] - 1] / val)
                res = np.array(res)
                if reverse:
                    res = 1 / res
                    # TODO check this area type
                return res, left_unit / right_unit, AreaType.ROI
            # TODO check this
            left_area = AreaType.Mask_without_ROI

        return left_res / right_res, left_unit / right_unit, left_area

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
            return self._calculate_leaf(node, segmentation_mask_map, help_dict, kwargs)
        if isinstance(node, Node):
            return self._calculate_node(node, segmentation_mask_map, help_dict, kwargs)
        raise ValueError(f"Node {node} need to be instance of Leaf or Node")

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
                if res[num][0] == 0:
                    res[num] = res[num][1:]
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
        image: Image,
        channel_num: int,
        roi: Union[np.ndarray, ROIInfo],
        result_units: Units,
        range_changed: Callable[[int, int], Any] = empty_fun,
        step_changed: Callable[[int], Any] = empty_fun,
        time: int = 0,
    ) -> MeasurementResult:
        """
        Calculate measurements on given set of parameters

        :param image: image on which measurements should be calculated
        :param roi: array with segmentation labeled as positive integers
        :param result_units: units which should be used to present results.
        :param range_changed: callback function to set information about steps range
        :param step_changed: callback function fo set information about steps done
        :param time: which data point should be measured
        :return: measurements
        """

        def get_time(array: np.ndarray):
            if array is not None and array.ndim == 4:
                return array.take(time, axis=image.time_pos)
            return array

        if self._need_mask and image.mask is None:
            raise ValueError("measurement need mask")
        channel = image.get_channel(channel_num).astype(float)
        cache_dict = {}
        result_scalar = UNIT_SCALE[result_units.value]
        roi_alternative = {}
        if isinstance(roi, ROIInfo):
            for name, array in roi.alternative.items():
                roi_alternative[name] = get_time(array)
        kw = {
            "image": image,
            "channel": get_time(channel),
            "segmentation": get_time(roi if isinstance(roi, np.ndarray) else roi.roi),
            "mask": get_time(image.mask),
            "voxel_size": image.spacing,
            "result_scalar": result_scalar,
            "roi_alternative": roi_alternative,
            "roi_annotation": roi.annotations if isinstance(roi, ROIInfo) else {},
        }
        segmentation_mask_map = self.get_segmentation_to_mask_component(kw["segmentation"], kw["mask"])
        result = MeasurementResult(segmentation_mask_map)
        for num in self.get_channels_num():
            kw["channel_{num}"] = get_time(image.get_channel(num))
        if any(self._need_mask_without_segmentation(el.calculation_tree) for el in self.chosen_fields):
            mm = kw["mask"].copy()
            mm[kw["segmentation"] > 0] = 0
            kw["mask_without_segmentation"] = mm

        range_changed(0, len(self.chosen_fields))
        for i, entry in enumerate(self.chosen_fields):
            step_changed(i)
            result[self.name_prefix + entry.name] = self._calc_single_field(
                entry, segmentation_mask_map, cache_dict, kw, result_units
            )

        return result

    def _calc_single_field(
        self,
        entry: MeasurementEntry,
        segmentation_mask_map: ComponentsInfo,
        cache_dict: dict,
        additional_args: dict,
        result_units,
    ):
        tree = entry.calculation_tree
        component_and_area = self._get_par_component_and_area_type(tree)
        try:
            val, unit, _area = self.calculate_tree(tree, segmentation_mask_map, cache_dict, additional_args)
            if isinstance(val, np.ndarray):
                val = list(val)
            return val, str(unit).format(str(result_units)), component_and_area
        except ZeroDivisionError:  # pragma: no cover
            return "Div by zero", "", component_and_area
        except TypeError as e:  # pragma: no cover
            if e.args[0].startswith("unsupported operand type(s) for /:"):
                return "None div", "", component_and_area
            raise e
        except AttributeError:  # pragma: no cover
            return "No attribute", "", component_and_area
        except ProhibitedDivision as e:  # pragma: no cover
            return e.args[0], "", component_and_area


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
    return np.max(centered, axis=1) - np.min(centered, axis=1)


def get_main_axis_length(
    index: int, area_array: np.ndarray, channel: np.ndarray, voxel_size, result_scalar, _cache=False, **kwargs
):
    _cache = _cache and "_area" in kwargs and "_per_component" in kwargs
    if _cache:
        help_dict: Dict = kwargs["help_dict"]
        _area: AreaType = kwargs["_area"]
        _per_component: PerComponent = kwargs["_per_component"]
        hash_name = hash_fun_call_name(
            calculate_main_axis, {}, _area, _per_component, kwargs["channel_num"], kwargs["_component_num"]
        )
        if hash_name not in help_dict:
            help_dict[hash_name] = calculate_main_axis(area_array, channel, [x * result_scalar for x in voxel_size])
        return help_dict[hash_name][index]

    return calculate_main_axis(area_array, channel, [x * result_scalar for x in voxel_size])[index]


def hash_fun_call_name(
    fun: Union[Callable, MeasurementMethodBase],
    arguments: Dict,
    area: AreaType,
    per_component: PerComponent,
    channel: Channel,
    components_num: int,
) -> str:
    """
    Calculate string for properly cache measurements result.

    :param fun: method for which hash string should be calculated
    :param arguments: its additional arguments
    :param area: type of rea
    :param per_component: If it is per component
    :param channel: channel number on which calculation is performed
    :return: unique string for such set of arguments
    """
    if hasattr(fun, "__module__"):
        fun_name = f"{fun.__module__}.{fun.__name__}"
    else:
        fun_name = fun.__name__
    return f"{fun_name}: {arguments} # {area} & {per_component} * {channel} ^ {components_num}"


class Volume(MeasurementMethodBase):
    text_info = "Volume", "Calculate volume of current segmentation"

    @classmethod
    def calculate_property(cls, area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
        return np.count_nonzero(area_array) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** ndim


class Voxels(MeasurementMethodBase):
    text_info = "Voxels", "Calculate number of voxels of current segmentation"

    @classmethod
    def calculate_property(cls, area_array, **_):  # pylint: disable=W0221
        return np.count_nonzero(area_array)

    @classmethod
    def get_units(cls, ndim):
        return symbols("1")


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
    points_array = np.ones(points_positions.shape[0], dtype=bool)
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
    """
    Class for calculate diameter of ROI in fast way.
    From Malandain, G., & Boissonnat, J. (2002). Computing the diameter of a point set,
    12(6), 489–509. https://doi.org/10.1142/S0218195902001006

    """

    text_info = "Diameter", "Diameter of area"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
        pos = np.transpose(np.nonzero(get_border(area_array))).astype(float)
        if pos.size == 0:
            return 0
        for i, val in enumerate((x * result_scalar for x in reversed(voxel_size)), start=1):
            pos[:, -i] *= val
        diam_sq = iterative_double_normal(pos)[0]
        return np.sqrt(diam_sq)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class DiameterOld(MeasurementMethodBase):  # pragma: no cover
    """
    n**2 calculate diameter of ROI
    """

    text_info = "Diameter old", "Diameter of area (Very slow)"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
        return calc_diam(get_border(area_array), [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class PixelBrightnessSum(MeasurementMethodBase):
    text_info = "Pixel brightness sum", "Sum of pixel brightness for current segmentation"

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
            else:  # pragma: no cover
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
    text_info = "Components number", "Calculate number of connected components on segmentation"

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
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.max(channel[area_array > 0])
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
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.min(channel[area_array > 0])
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
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.mean(channel[area_array > 0])
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
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.median(channel[area_array > 0])
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
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        if np.any(area_array):
            return np.std(channel[area_array > 0])
        return 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class Moment(MeasurementMethodBase):
    text_info = "Moment", "Calculate moment of segmented structure"

    @staticmethod
    def calculate_property(area_array, channel, voxel_size, **_):  # pylint: disable=W0221
        if len(channel.shape) == 4:
            if channel.shape[0] != 1:  # pragma: no cover
                raise ValueError("This measurements do not support time data")
            channel = channel[0]
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


class FirstPrincipalAxisLength(MeasurementMethodBase):
    text_info = "First principal axis length", "Length of first principal axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(0, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @classmethod
    def need_channel(cls):
        return True


class SecondPrincipalAxisLength(MeasurementMethodBase):
    text_info = "Second principal axis length", "Length of second principal axis"

    @staticmethod
    def calculate_property(**kwargs):
        return get_main_axis_length(1, **kwargs)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @classmethod
    def need_channel(cls):
        return True


class ThirdPrincipalAxisLength(MeasurementMethodBase):
    text_info = "Third principal axis length", "Length of third principal axis"

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
            border_hash_str = hash_fun_call_name(
                Surface, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1), kwargs["_component_num"]
            )
            if border_hash_str not in help_dict:
                border_surface = Surface.calculate_property(**kwargs)
                help_dict[border_hash_str] = border_surface
            else:
                border_surface = help_dict[border_hash_str]

            volume_hash_str = hash_fun_call_name(
                Volume, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1), kwargs["_component_num"]
            )

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
    text_info = "Sphericity", "volume/(4/3 * π * radius **3) for 3d data and volume/(π * radius **2) for 2d data"

    @staticmethod
    def calculate_property(**kwargs):
        if all(key in kwargs for key in ["help_dict", "_area", "_per_component"]) and (
            "_cache" not in kwargs or kwargs["_cache"]
        ):
            help_dict = kwargs["help_dict"]
        else:
            help_dict = {}
            kwargs.update({"_area": AreaType.ROI, "_per_component": PerComponent.No, "_component_num": NO_COMPONENT})
        volume_hash_str = hash_fun_call_name(
            Volume, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1), kwargs["_component_num"]
        )
        if volume_hash_str not in help_dict:
            volume = Volume.calculate_property(**kwargs)
            help_dict[volume_hash_str] = volume
        else:
            volume = help_dict[volume_hash_str]

        diameter_hash_str = hash_fun_call_name(
            Diameter, {}, kwargs["_area"], kwargs["_per_component"], Channel(-1), kwargs["_component_num"]
        )
        if diameter_hash_str not in help_dict:
            diameter_val = Diameter.calculate_property(**kwargs)
            help_dict[diameter_hash_str] = diameter_val
        else:
            diameter_val = help_dict[diameter_hash_str]
        radius = diameter_val / 2
        if kwargs["area_array"].shape[0] > 1:
            return volume / (4 / 3 * pi * (radius ** 3))
        return volume / (pi * (radius ** 2))

    @classmethod
    def get_units(cls, ndim):
        return Volume.get_units(ndim) / Diameter.get_units(ndim) ** ndim


class Surface(MeasurementMethodBase):
    text_info = "Surface", "Calculating surface of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=W0221
        return calculate_volume_surface(area_array, [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 2


class RimVolume(MeasurementMethodBase):
    text_info = "rim volume", "Calculate volumes for elements in radius (in physical units) from mask"

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
        return symbols("{}") ** ndim

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.ROI


class RimPixelBrightnessSum(MeasurementMethodBase):
    text_info = (
        "rim pixel brightness sum",
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
            if channel.shape[0] != 1:  # pragma: no cover
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
        return AreaType.ROI


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
            area_pos = np.transpose(np.nonzero(get_border(area_array))).astype(float)
            area_pos += 0.5
            for i, val in enumerate((x * result_scalar for x in reversed(voxel_size)), start=1):
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
        if 1 in {mask_pos.shape[0], seg_pos.shape[0]}:
            return np.min(cdist(mask_pos, seg_pos))

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
        return AreaType.ROI


class SplitOnPartVolume(MeasurementMethodBase):
    text_info = (
        "distance splitting volume",
        "Split mask on parts and then calculate volume of cross of segmentation and mask part",
    )

    @classmethod
    def get_fields(cls):
        return MaskDistanceSplit.get_fields() + [
            AlgorithmProperty("part_selection", "Which part  (from border)", 2, (1, 1024))
        ]

    @staticmethod
    def calculate_property(part_selection, area_array, voxel_size, result_scalar, **kwargs):  # pylint: disable=W0221
        masked = MaskDistanceSplit.split(voxel_size=voxel_size, **kwargs)
        mask = masked == part_selection
        return np.count_nonzero(mask * area_array) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** ndim

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.Mask)

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.ROI


class SplitOnPartPixelBrightnessSum(MeasurementMethodBase):
    text_info = (
        "distance splitting pixel brightness sum",
        "Split mask on parts and then calculate pixel brightness sum of cross of segmentation and mask part",
    )

    @classmethod
    def get_fields(cls):
        return MaskDistanceSplit.get_fields() + [
            AlgorithmProperty("part_selection", "Which part (from border)", 2, (1, 1024))
        ]

    @staticmethod
    def calculate_property(part_selection, channel, area_array, **kwargs):  # pylint: disable=W0221
        masked = MaskDistanceSplit.split(**kwargs)
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
        return AreaType.ROI

    @classmethod
    def need_channel(cls):
        return True


HARALIC_FEATURES = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()


def _rescale_image(data: np.ndarray):
    if data.dtype == np.uint8:
        return data
    min_val = data.min()
    max_val = data.max()
    return ((data - min_val) / ((max_val - min_val) / 255)).astype(np.uint8)


class Haralick(MeasurementMethodBase):
    @classmethod
    def get_units(cls, ndim) -> symbols:
        return "1"

    text_info = "Haralick", "Calculate Haralick features"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty("feature", "Feature", HARALIC_FEATURES[0], possible_values=HARALIC_FEATURES),
            AlgorithmProperty("distance", "Distance", 1, options_range=(1, 10)),
        ]

    @classmethod
    def need_channel(cls):
        return True

    @classmethod
    def calculate_property(
        cls, area_array, channel, distance, feature, _cache=False, **kwargs
    ):  # pylint: disable=W0221
        _cache = _cache and "_area" in kwargs and "_per_component" in kwargs
        if _cache:
            help_dict: Dict = kwargs["help_dict"]
            _area: AreaType = kwargs["_area"]
            _per_component: PerComponent = kwargs["_per_component"]
            hash_name = hash_fun_call_name(
                Haralick, {"distance": distance}, _area, _per_component, kwargs["channel_num"], kwargs["_component_num"]
            )
            if hash_name not in help_dict:
                help_dict[hash_name] = cls.calculate_haralick(channel, area_array, distance)
            return help_dict[hash_name][HARALIC_FEATURES.index(feature)]

        res = cls.calculate_haralick(channel, area_array, distance)
        return res[HARALIC_FEATURES.index(feature)]

    @staticmethod
    def calculate_haralick(channel, area_array, distance):
        data = channel.copy()
        data[area_array == 0] = 0
        data = _rescale_image(data.squeeze())
        return haralick(data, distance=distance, ignore_zeros=True, return_mean=True)


def pixel_volume(spacing, result_scalar):
    return reduce((lambda x, y: x * y), [x * result_scalar for x in spacing])


def calculate_volume_surface(volume_mask, voxel_size):
    surf_im: np.ndarray = np.array(volume_mask).astype(np.uint8).squeeze()
    return sum(
        (
            np.count_nonzero(
                np.logical_xor(
                    surf_im.take(np.arange(surf_im.shape[ax] - 1), axis=ax),
                    surf_im.take(np.arange(surf_im.shape[ax] - 1) + 1, axis=ax),
                )
            )
            * reduce(
                lambda x, y: x * y,
                [voxel_size[x] for x in range(surf_im.ndim) if x != ax],
            )
        )
        for ax in range(surf_im.ndim)
    )


def get_border(array):
    if array.dtype == bool:
        array = array.astype(np.uint8)
    return SimpleITK.GetArrayFromImage(SimpleITK.LabelContour(SimpleITK.GetImageFromArray(array)))


def calc_diam(array, voxel_size):  # pragma: no cover
    pos = np.transpose(np.nonzero(array)).astype(float)
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
    Moment,
    FirstPrincipalAxisLength,
    SecondPrincipalAxisLength,
    ThirdPrincipalAxisLength,
    Compactness,
    Sphericity,
    Surface,
    RimVolume,
    RimPixelBrightnessSum,
    DistanceMaskSegmentation,
    SplitOnPartVolume,
    SplitOnPartPixelBrightnessSum,
    Voxels,
    Haralick,
    suggested_base_class=MeasurementMethodBase,
)
"""Register with all measurements algorithms"""
