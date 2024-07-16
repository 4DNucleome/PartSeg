import warnings
from collections import OrderedDict
from contextlib import suppress
from enum import Enum
from functools import reduce
from math import pi
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import SimpleITK
from local_migrator import register_class, rename_key
from mahotas.features import haralick
from pydantic import Field
from scipy.spatial.distance import cdist
from sympy import Rational, symbols

from PartSegCore import autofit as af
from PartSegCore.algorithm_describe_base import Register, ROIExtractionProfile
from PartSegCore.analysis.calculate_pipeline import calculate_segmentation_step
from PartSegCore.analysis.measurement_base import (
    AreaType,
    Leaf,
    MeasurementEntry,
    MeasurementMethodBase,
    Node,
    PerComponent,
    has_mask_components,
    has_roi_components,
)
from PartSegCore.mask_partition_utils import BorderRim, MaskDistanceSplit
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.restartable_segmentation_algorithms import LowerThresholdAlgorithm
from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegCore.utils import BaseModel
from PartSegImage import Channel, Image

# TODO change image to channel in signature of measurement calculate_property

NO_COMPONENT = -1


class CorrelationEnum(str, Enum):
    pearson = "Pearson correlation coefficient"
    manders = "Mander's overlap coefficient"
    intensity = "Intensity correlation quotient"
    spearman = "Spearman rank correlation"

    def __str__(self):
        return self.value


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

    def has_components(self):
        return all(len(x) for x in self.components_translation.values())


def empty_fun(_a0=None, _a1=None):
    """This function is being used as dummy reporting function."""


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

    def to_dataframe(self, all_components=False) -> pd.DataFrame:
        data = self.get_separated(all_components)
        columns = [
            f"{label} ({units})" if units else label
            for label, units in zip(self.get_labels(all_components=all_components), self.get_units(all_components))
        ]
        df = pd.DataFrame(data, columns=columns, index=self.components_info.roi_components)
        if "Segmentation component" in df.columns:
            df = df.astype({"Segmentation component": int}).set_index("Segmentation component")
        return df

    def set_filename(self, path_fo_file: str):
        """
        Set name of file to be presented as first position.
        """
        self._data_dict[FILE_NAME_STR] = path_fo_file
        self._type_dict[FILE_NAME_STR] = PerComponent.No, AreaType.ROI
        self._units_dict[FILE_NAME_STR] = ""
        self._data_dict.move_to_end(FILE_NAME_STR, False)

    def get_component_info(self, all_components: bool = False) -> Tuple[bool, bool]:
        """
        Get information which type of components are in storage.

        :return: has_mask_components, has_segmentation_components
        """
        if all_components and self.components_info.has_components():
            return True, True

        return has_mask_components(self._type_dict.values()), has_roi_components(self._type_dict.values())

    def get_labels(self, expand=True, all_components=False) -> List[str]:
        """
        If expand is false return list of keys of this storage.
        Otherwise return  labels for measurement. Base are keys of this storage.
        If has mask components, or has segmentation_components then add this labels
        """

        if not expand:
            return list(self.keys())
        mask_components, roi_components = self.get_component_info(all_components)
        labels = list(self._data_dict.keys())
        index = 1 if FILE_NAME_STR in self._data_dict else 0
        if mask_components:
            labels.insert(index, "Mask component")
        if roi_components:
            labels.insert(index, "Segmentation component")
        return labels

    def get_units(self, all_components=False) -> List[str]:
        return [self._units_dict[x] for x in self.get_labels(all_components=all_components)]

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
            with suppress(StopIteration):
                next(iterator)  # skipcq: PTC-W0063`
        else:
            res = []
            iterator = iter(self._data_dict.keys())
        for el in iterator:
            per_comp = self._type_dict[el][0]
            val = self._data_dict[el]
            if per_comp not in {PerComponent.Yes, PerComponent.Per_Mask_component}:
                res.append(val)
        return res

    def _get_component_info(self, mask_components, roi_components):
        if mask_components:
            if roi_components:
                translation = self.components_info.components_translation
                return [(x, y) for x in translation for y in translation[x]]
            return [(0, x) for x in self.components_info.mask_components]
        return [(x, 0) for x in self.components_info.roi_components]

    def _prepare_res_iterator(self, counts):
        if FILE_NAME_STR in self._data_dict:
            name = self._data_dict[FILE_NAME_STR]
            res = [[name] for _ in range(counts)]
            iterator = iter(self._data_dict.keys())
            with suppress(StopIteration):
                next(iterator)  # skipcq: PTC-W0063`
        else:
            res = [[] for _ in range(counts)]
            iterator = iter(self._data_dict.keys())
        return res, iterator

    def get_separated(self, all_components=False) -> List[List[MeasurementValueType]]:
        """Get measurements separated for each component"""
        mask_components, roi_components = self.get_component_info(all_components)
        if not mask_components and not roi_components:
            return [list(self._data_dict.values())]
        component_info = self._get_component_info(mask_components, roi_components)
        res, iterator = self._prepare_res_iterator(len(component_info))

        for i, num in enumerate(component_info):
            if roi_components:
                res[i].append(num[0])
            if mask_components:
                res[i].append(num[1])

        mask_to_pos = {val: i for i, val in enumerate(self.components_info.mask_components)}
        segmentation_to_pos = {val: i for i, val in enumerate(self.components_info.roi_components)}

        for el in iterator:
            per_comp, area_type = self._type_dict[el]
            val = self._data_dict[el]
            for i, (seg, mask) in enumerate(component_info):
                if per_comp not in {PerComponent.Yes, PerComponent.Per_Mask_component}:
                    res[i].append(val)
                elif area_type == AreaType.ROI:
                    res[i].append(val[segmentation_to_pos[seg]])
                else:
                    res[i].append(val[mask_to_pos[mask]])
        return res


class MeasurementProfile(BaseModel):
    name: str
    chosen_fields: List[MeasurementEntry]
    name_prefix: str = ""

    @property
    def _need_mask(self):
        return any(cf_val.calculation_tree.need_mask() for cf_val in self.chosen_fields)

    def to_dict(self):  # pragma: no cover
        warnings.warn(
            f"{self.__class__.__name__}.to_dict is deprecated. Use as_dict instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return dict(self)

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
            if tree.per_component == PerComponent.Per_Mask_component:
                return tree.per_component, AreaType.Mask
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
        if self.name_prefix:
            text += f"Name prefix: {self.name_prefix}\n"
        text += "Measurements list:\n"
        for el in self.chosen_fields:
            text += f"{el.name}\n"
        return text

    def get_component_info(self, unit: Units):
        """
        :return: list[((str, str), bool)]
        """
        # Fixme remove binding to 3 dimensions
        return [
            (
                (self.name_prefix + el.name, el.get_unit(unit, 3)),
                self._is_component_measurement(el.calculation_tree),
            )
            for el in self.chosen_fields
        ]

    def is_any_mask_measurement(self):
        return any(el.calculation_tree.need_mask() for el in self.chosen_fields)

    def _is_component_measurement(self, node):
        if isinstance(node, Leaf):
            return node.per_component in {PerComponent.Yes, PerComponent.Per_Mask_component}
        return self._is_component_measurement(node.left) or self._is_component_measurement(node.right)

    @staticmethod
    def _prepare_leaf_kw(node, kwargs, method, area_type):
        area_type_dict = {
            AreaType.Mask: "mask",
            AreaType.Mask_without_ROI: "mask_without_segmentation",
            AreaType.ROI: "segmentation",
        }
        kw = dict(kwargs)
        kw.update(dict(node.parameters))

        if node.channel is not None:
            kw["channel"] = kw[f"channel_{node.channel}"]
            kw["channel_num"] = node.channel
        else:
            kw["channel_num"] = -1

        kw["_area"] = node.area
        kw["_per_component"] = node.per_component
        kw["_cache"] = True  # TODO remove cache argument
        kw["area_array"] = kw[area_type_dict[area_type]]
        kw["_component_num"] = NO_COMPONENT
        return kw

    def _clip_arrays(self, kw, node: Leaf, method: MeasurementMethodBase, component_index: int):
        if node.area != AreaType.ROI or method.need_full_data():
            bounds = tuple(slice(None, None) for _ in kw["area_array"].shape)
        elif node.per_component == PerComponent.Per_Mask_component:
            bounds = tuple(kw["mask_bound_info"][component_index].get_slices(margin=1))
        else:
            bounds = tuple(kw["bounds_info"][component_index].get_slices(margin=1))
        kw2 = kw.copy()

        component_mark_area = (
            kw2["area_array"] if node.per_component != PerComponent.Per_Mask_component else kw2["mask"]
        )

        kw2["_component_num"] = component_index

        area_array = kw2["area_array"][bounds].copy()
        area_array[component_mark_area[bounds] != component_index] = 0

        kw2["area_array"] = area_array
        im_bounds = list(bounds)
        image: Image = kw["image"]
        im_bounds.insert(image.time_pos, slice(None))
        kw2["image"] = image.cut_image(tuple(im_bounds))
        for name in ["channel", "segmentation", "roi", "mask"] + [f"channel_{num}" for num in self.get_channels_num()]:
            if kw[name] is not None:
                kw2[name] = kw[name][bounds]
        kw2["roi_alternative"] = kw2["roi_alternative"].copy()
        for name, array in kw2["roi_alternative"].items():
            kw2["roi_alternative"][name] = array[bounds]
        return kw2

    def _calculate_leaf_value(
        self, node: Union[Node, Leaf], segmentation_mask_map: ComponentsInfo, kwargs: dict
    ) -> Union[float, np.ndarray]:
        method: MeasurementMethodBase = MEASUREMENT_DICT[node.name]
        kw = self._prepare_leaf_kw(node, kwargs, method, method.area_type(node.area))

        if node.per_component == PerComponent.No:
            return method.calculate_property(**kw)
        # TODO use cache for per component calculate
        val = []
        if method.area_type(node.area) == AreaType.ROI and node.per_component != PerComponent.Per_Mask_component:
            components = segmentation_mask_map.roi_components
        else:
            components = segmentation_mask_map.mask_components
        for i in components:
            kw2 = self._clip_arrays(kw, node, method, i)
            val.append(method.calculate_property(**kw2))
        val = np.array(val)
        if node.per_component == PerComponent.Mean:
            val = np.mean(val) if val.size else 0
        return val

    def _calculate_leaf(
        self, node: Leaf, segmentation_mask_map: ComponentsInfo, help_dict: dict, kwargs: dict
    ) -> Tuple[Union[float, np.ndarray], symbols, AreaType]:
        method: MeasurementMethodBase = MEASUREMENT_DICT[node.name]

        hash_str = hash_fun_call_name(
            method, node.parameters, node.area, node.per_component, node.channel, NO_COMPONENT
        )
        area_type = method.area_type(node.area)
        if node.per_component == PerComponent.Per_Mask_component:
            area_type = AreaType.Mask
        if hash_str in help_dict:
            val = help_dict[hash_str]
        else:
            kwargs["help_dict"] = help_dict
            val = self._calculate_leaf_value(node, segmentation_mask_map, kwargs)
            help_dict[hash_str] = val
        unit: symbols = method.get_units(3) if kwargs["image"].is_stack else method.get_units(2)
        if node.power != 1:
            return pow(val, node.power), pow(unit, Rational(node.power)), area_type
        return val, unit, area_type

    def _calculate_node(
        self, node: Node, segmentation_mask_map: ComponentsInfo, help_dict: dict, kwargs: dict
    ) -> Tuple[Union[float, np.ndarray], symbols, AreaType]:
        if node.op != "/":
            raise ValueError(f"Wrong measurement: {node}")
        left_res, left_unit, left_area = self.calculate_tree(node.left, segmentation_mask_map, help_dict, kwargs)
        right_res, right_unit, right_area = self.calculate_tree(node.right, segmentation_mask_map, help_dict, kwargs)
        if not (isinstance(left_res, np.ndarray) and isinstance(right_res, np.ndarray) and left_area != right_area):
            return left_res / right_res, left_unit / right_unit, left_area
        area_set = {left_area, right_area}
        if area_set == {AreaType.ROI, AreaType.Mask_without_ROI}:  # pragma: no cover
            raise ProhibitedDivision("This division is prohibited")
        if area_set == {AreaType.Mask, AreaType.Mask_without_ROI}:
            return left_res / right_res, left_unit / right_unit, AreaType.Mask_without_ROI

        # if area_set == {AreaType.ROI, AreaType.Mask}:
        res = []
        if left_area == AreaType.Mask:
            roi_res, mask_res = right_res, left_res
        else:
            roi_res, mask_res = left_res, right_res
        for val, num in zip(roi_res, segmentation_mask_map.roi_components):
            mask_components = segmentation_mask_map.components_translation[num]
            if len(mask_components) != 1:  # pragma: no cover
                raise ProhibitedDivision("Cannot calculate when object do not belongs to one mask area")
            if left_area == AreaType.ROI:
                res.append(val / mask_res[mask_components[0] - 1])
            else:
                res.append(mask_res[mask_components[0] - 1] / val)
        res = np.array(res)
        return res, left_unit / right_unit, AreaType.ROI
        # TODO check this

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
        """For each measurement check if is per component and in which types"""
        res = []
        for el in self.chosen_fields:
            tree = el.calculation_tree
            res.append(self._get_par_component_and_area_type(tree))
        return res

    def get_segmentation_mask_map(self, image: Image, roi: Union[np.ndarray, ROIInfo], time: int = 0) -> ComponentsInfo:
        def get_time(array: np.ndarray):
            if array is not None and array.ndim == 4:
                return array.take(time, axis=image.time_pos)
            return array

        return self.get_segmentation_to_mask_component(
            get_time(roi if isinstance(roi, np.ndarray) else roi.roi), get_time(image.mask)
        )

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
        :param channel_num: channel number on which measurements should be calculated
        :param roi: array with segmentation labeled as positive integers
        :param result_units: units which should be used to present results.
        :param range_changed: callback function to set information about steps range
        :param step_changed: callback function for set information about steps done
        :param time: which data point should be measured
        :return: measurements
        """

        segmentation_mask_map = self.get_segmentation_mask_map(image, roi, time)
        result = MeasurementResult(segmentation_mask_map)
        range_changed(0, len(self.chosen_fields))
        for i, (name, data) in enumerate(
            self.calculate_yield(
                image=image,
                channel_num=channel_num,
                roi=roi,
                result_units=result_units,
                segmentation_mask_map=segmentation_mask_map,
                time=time,
            ),
            start=1,
        ):
            result[name] = data
            step_changed(i)

        return result

    def calculate_yield(
        self,
        image: Image,
        channel_num: int,
        roi: Union[np.ndarray, ROIInfo],
        result_units: Units,
        segmentation_mask_map: ComponentsInfo,
        time: int = 0,
    ) -> Generator[MeasurementResultInputType, None, None]:
        """
        Calculate measurements on given set of parameters

        :param image: image on which measurements should be calculated
        :param roi: array with segmentation labeled as positive integers
        :param result_units: units which should be used to present results.
        :param segmentation_mask_map: information which component of roi belongs to which mask component.
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
        if isinstance(roi, np.ndarray):
            roi = ROIInfo(roi).fit_to_image(image)
        mask_bound_info = None
        if isinstance(image.mask, np.ndarray):
            mask_bound_info = {
                k: v.del_dim(image.time_pos) if len(v.lower) == 4 else v
                for k, v in ROIInfo(image.mask).fit_to_image(image).bound_info.items()
            }
        roi_alternative = {}
        for name, array in roi.alternative.items():
            roi_alternative[name] = get_time(array)
        kw = {
            "image": image,
            "channel": get_time(channel),
            "segmentation": get_time(roi.roi),
            "roi": get_time(roi.roi),
            "bounds_info": {
                k: v.del_dim(image.time_pos) if len(v.lower) == 4 else v for k, v in roi.bound_info.items()
            },
            "mask_bound_info": mask_bound_info,
            "mask": get_time(image.mask),
            "voxel_size": image.spacing,
            "result_scalar": result_scalar,
            "roi_alternative": roi_alternative,
            "roi_annotation": roi.annotations,
        }
        for num in self.get_channels_num():
            kw[f"channel_{num}"] = get_time(image.get_channel(num))
        if any(self._need_mask_without_segmentation(el.calculation_tree) for el in self.chosen_fields):
            mm = kw["mask"].copy()
            mm[kw["segmentation"] > 0] = 0
            kw["mask_without_segmentation"] = mm

        for entry in self.chosen_fields:
            name = self.name_prefix + entry.name
            yield name, self._calc_single_field(entry, segmentation_mask_map, cache_dict, kw, result_units)

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
    if _cache and "_area" in kwargs and "_per_component" in kwargs and "channel_num" in kwargs:
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
    def calculate_property(cls, area_array, voxel_size, result_scalar, **_):  # pylint: disable=arguments-differ
        return np.count_nonzero(area_array) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** ndim


class Voxels(MeasurementMethodBase):
    text_info = "Voxels", "Calculate number of voxels of current segmentation"

    @classmethod
    def calculate_property(cls, area_array, **_):  # pylint: disable=arguments-differ
        return np.count_nonzero(area_array)

    @classmethod
    def get_units(cls, ndim):
        return symbols("1")


# From Malandain, G., & Boissonnat, J. (2002). Computing the diameter of a point set,
# 12(6), 489-509. https://doi.org/10.1142/S0218195902001006


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
    12(6), 489-509. https://doi.org/10.1142/S0218195902001006

    """

    text_info = "Diameter", "Diameter of area"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=arguments-differ
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
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=arguments-differ
        return calc_diam(get_border(area_array), [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")


class PixelBrightnessSum(MeasurementMethodBase):
    text_info = "Pixel brightness sum", "Sum of pixel brightness for current segmentation"

    @staticmethod
    def calculate_property(area_array: np.ndarray, channel: np.ndarray, **_):  # pylint: disable=arguments-differ
        """
        :param area_array: mask for area
        :param channel: data. same shape like area_type
        :return: Pixels brightness sum on given area
        """
        if area_array.shape != channel.shape:
            if area_array.size == channel.size:
                channel = channel.reshape(area_array.shape)
            else:  # pragma: no cover
                raise ValueError(f"channel ({channel.shape}) and mask ({area_array.shape}) do not fit each other")
        return np.sum(channel[area_array > 0]) if np.any(area_array) else 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class ComponentsNumber(MeasurementMethodBase):
    text_info = "Components number", "Calculate number of connected components on segmentation"

    @staticmethod
    def calculate_property(area_array, **_):  # pylint: disable=arguments-differ
        return np.unique(area_array).size - 1

    @classmethod
    def get_units(cls, ndim):
        return symbols("count")


class MaximumPixelBrightness(MeasurementMethodBase):
    text_info = "Maximum pixel brightness", "Calculate maximum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):  # pylint: disable=arguments-differ
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError(f"channel ({channel.shape}) and mask ({area_array.shape}) do not fit each other")
        return np.max(channel[area_array > 0]) if np.any(area_array) else 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class MinimumPixelBrightness(MeasurementMethodBase):
    text_info = "Minimum pixel brightness", "Calculate minimum pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):  # pylint: disable=arguments-differ
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        return np.min(channel[area_array > 0]) if np.any(area_array) else 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class MeanPixelBrightness(MeasurementMethodBase):
    text_info = "Mean pixel brightness", "Calculate mean pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):  # pylint: disable=arguments-differ
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        return np.mean(channel[area_array > 0]) if np.any(area_array) else 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class MedianPixelBrightness(MeasurementMethodBase):
    text_info = "Median pixel brightness", "Calculate median pixel brightness for current area"

    @staticmethod
    def calculate_property(area_array, channel, **_):  # pylint: disable=arguments-differ
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        return np.median(channel[area_array > 0]) if np.any(area_array) else 0

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
    def calculate_property(area_array, channel, **_):  # pylint: disable=arguments-differ
        if area_array.shape != channel.shape:  # pragma: no cover
            raise ValueError("channel and mask do not fit each other")
        return np.std(channel[area_array > 0]) if np.any(area_array) else 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True


class Moment(MeasurementMethodBase):
    text_info = "Moment", "Calculate moment of segmented structure"

    @staticmethod
    def calculate_property(area_array, channel, voxel_size, **_):  # pylint: disable=arguments-differ
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
    def calculate_property(**kwargs):  # pylint: disable=arguments-differ
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
    def calculate_property(**kwargs):  # pylint: disable=arguments-differ
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
    def calculate_property(**kwargs):  # pylint: disable=arguments-differ
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
    def calculate_property(**kwargs):  # pylint: disable=arguments-differ
        if kwargs.get("_cache", False) and "help_dict" in kwargs and "_area" in kwargs and "_per_component" in kwargs:
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
        return border_surface**1.5 / volume

    @classmethod
    def get_units(cls, ndim):
        return Surface.get_units(ndim) / Volume.get_units(ndim)


class Sphericity(MeasurementMethodBase):
    text_info = "Sphericity", "volume/(4/3 * π * radius **3) for 3d data and volume/(π * radius **2) for 2d data"

    @staticmethod
    def calculate_property(**kwargs):  # pylint: disable=arguments-differ
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
            return volume / (4 / 3 * pi * (radius**3))
        return volume / (pi * (radius**2))

    @classmethod
    def get_units(cls, ndim):
        return Volume.get_units(ndim) / Diameter.get_units(ndim) ** ndim


class Surface(MeasurementMethodBase):
    text_info = "Surface", "Calculating surface of current segmentation"

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **_):  # pylint: disable=arguments-differ
        return calculate_volume_surface(area_array, [x * result_scalar for x in voxel_size])

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** 2


class RimVolume(MeasurementMethodBase):
    text_info = "rim volume", "Calculate volumes for elements in radius (in physical units) from mask"
    __argument_class__ = BorderRim.__argument_class__

    @classmethod
    def get_starting_leaf(cls):
        return super().get_starting_leaf().replace_(area=AreaType.Mask)

    @staticmethod
    def calculate_property(area_array, voxel_size, result_scalar, **kwargs):  # pylint: disable=arguments-differ
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
    __argument_class__ = BorderRim.__argument_class__

    @classmethod
    def get_starting_leaf(cls):
        return super().get_starting_leaf().replace_(area=AreaType.Mask)

    @staticmethod
    def calculate_property(channel, area_array, **kwargs):  # pylint: disable=arguments-differ
        if len(channel.shape) == 4:
            if channel.shape[0] != 1:  # pragma: no cover
                raise ValueError("This measurements do not support time data")
            channel = channel[0]
        border_mask_array = BorderRim.border_mask(**kwargs)
        if border_mask_array is None:
            return None
        final_mask = np.array((border_mask_array > 0) * (area_array > 0))
        return np.sum(channel[final_mask]) if np.any(final_mask) else 0

    @classmethod
    def get_units(cls, ndim):
        return symbols("Pixel_brightness")

    @classmethod
    def need_channel(cls):
        return True

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.ROI


@register_class(old_paths=["PartSeg.utils.analysis.statistics_calculation.DistancePoint"])
class DistancePoint(Enum):
    Border = 1
    Mass_center = 2
    Geometrical_center = 3

    def __str__(self):
        return self.name.replace("_", " ")


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("distance_to_segmentation", "distance_to_roi"))])
class DistanceMaskROIParameters(BaseModel):
    distance_from_mask: DistancePoint = DistancePoint.Border
    distance_to_roi: DistancePoint = Field(DistancePoint.Border, title="Distance to ROI")


class DistanceMaskROI(MeasurementMethodBase):
    text_info = "ROI distance", "Calculate distance between ROI and mask"
    __argument_class__ = DistanceMaskROIParameters

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
    def calculate_property(  # pylint: disable=arguments-differ
        cls,
        channel,
        area_array,
        mask,
        voxel_size,
        result_scalar,
        distance_from_mask: DistancePoint,
        distance_to_roi: DistancePoint,
        *args,
        **kwargs,
    ):  # pylint: disable=arguments-differ
        if len(channel.shape) == 4:
            if channel.shape[0] != 1:
                raise ValueError("This measurements do not support time data")
            channel = channel[0]
        if not (np.any(mask) and np.any(area_array)):
            return 0
        mask_pos = cls.calculate_points(channel, mask, voxel_size, result_scalar, distance_from_mask)
        seg_pos = cls.calculate_points(channel, area_array, voxel_size, result_scalar, distance_to_roi)
        if 1 in {mask_pos.shape[0], seg_pos.shape[0]}:
            return np.min(cdist(mask_pos, seg_pos))

        min_val = np.inf
        for i in range(seg_pos.shape[0]):
            min_val = min(min_val, np.min(cdist(mask_pos, np.array([seg_pos[i]]))))
        return min_val

    @classmethod
    def get_starting_leaf(cls):
        return super().get_starting_leaf().replace_(area=AreaType.Mask)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}")

    @classmethod
    def need_channel(cls):
        return True

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.ROI


class DistanceROIROIParameters(BaseModel):
    profile: ROIExtractionProfile = Field(
        ROIExtractionProfile(
            name="default",
            algorithm=LowerThresholdAlgorithm.get_name(),
            values=LowerThresholdAlgorithm.get_default_values(),
        ),
        title="ROI extraction profile",
    )
    distance_from_new_roi: DistancePoint = Field(DistancePoint.Border, title="Distance new ROI")
    distance_to_roi: DistancePoint = Field(DistancePoint.Border, title="Distance to ROI")


class DistanceROIROI(DistanceMaskROI):
    text_info = "to new ROI distance", "Calculate distance between ROI and new ROI"
    __argument_class__ = DistanceROIROIParameters

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI)

    # noinspection PyMethodOverriding
    @classmethod
    def calculate_property(
        cls,
        channel: np.ndarray,
        image: Image,
        area_array: np.ndarray,
        profile: ROIExtractionProfile,
        mask: Optional[np.ndarray],
        voxel_size: Sequence[float],
        result_scalar: float,
        distance_from_new_roi: DistancePoint,
        distance_to_roi: DistancePoint,
        **kwargs,
    ):  # pylint: disable=arguments-differ
        if len(channel.shape) == 4:
            if channel.shape[0] != 1:
                raise ValueError("This measurements do not support time data")
            channel = channel[0]
        try:
            hash_name = hash_fun_call_name(
                calculate_segmentation_step,
                profile,
                kwargs["_area"],
                kwargs["_per_component"],
                Channel(-1),
                kwargs["_component_num"],
            )
            if hash_name in kwargs["help_dict"]:
                result = kwargs["help_dict"][hash_name]
            else:
                result, _ = calculate_segmentation_step(profile, image, mask)
                kwargs["help_dict"][hash_name] = result
        except KeyError:
            result, _ = calculate_segmentation_step(profile, image, mask)

        if np.any(result.roi[area_array > 0]):
            return 0

        return super().calculate_property(
            channel,
            area_array,
            result.roi,
            tuple(voxel_size),
            result_scalar,
            distance_from_mask=distance_from_new_roi,
            distance_to_roi=distance_to_roi,
        )

    @staticmethod
    def need_full_data():
        return True


class ROINeighbourhoodROIParameters(BaseModel):
    profile: ROIExtractionProfile = Field(
        ROIExtractionProfile(
            name="default",
            algorithm=LowerThresholdAlgorithm.get_name(),
            values=LowerThresholdAlgorithm.get_default_values(),
        ),
        title="ROI extraction profile",
    )
    distance: float = Field(500, ge=0, le=10000, title="Distance")
    units: Units = Units.nm


class ROINeighbourhoodROI(DistanceMaskROI):
    text_info = "Neighbourhood new ROI presence", "Count how many of new roi are present in neighbourhood of new ROI"
    __argument_class__ = ROINeighbourhoodROIParameters

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI)

    # noinspection PyMethodOverriding
    @classmethod
    def calculate_property(
        cls,
        image: Image,
        area_array: np.ndarray,
        profile: ROIExtractionProfile,
        mask: Optional[np.ndarray],
        voxel_size,
        distance: float,
        units: Units,
        **kwargs,
    ):  # pylint: disable=arguments-differ
        try:
            hash_name = hash_fun_call_name(
                calculate_segmentation_step,
                profile,
                kwargs["_area"],
                kwargs["_per_component"],
                Channel(-1),
                kwargs["_component_num"],
            )
            if hash_name in kwargs["help_dict"]:
                result = kwargs["help_dict"][hash_name]
            else:
                result, _ = calculate_segmentation_step(profile, image, mask)
                kwargs["help_dict"][hash_name] = result
        except KeyError:
            result, _ = calculate_segmentation_step(profile, image, mask)
        area_array = image.fit_array_to_image(area_array)
        units_scalar = UNIT_SCALE[units.value]
        final_radius = [int((distance / units_scalar) / x) for x in reversed(voxel_size)]

        dilated = SimpleITK.GetArrayFromImage(
            SimpleITK.BinaryDilate(
                SimpleITK.GetImageFromArray((area_array > 0).astype(np.uint8).squeeze()), final_radius
            )
        )
        dilated = dilated.reshape(area_array.shape)
        roi = image.fit_array_to_image(result.roi)

        components = set(np.unique(roi[dilated > 0]))
        if 0 in components:
            components.remove(0)

        return len(components)

    @staticmethod
    def need_full_data():
        return True


class SplitOnPartParameters(MaskDistanceSplit.__argument_class__):
    part_selection: int = Field(2, title="Which part (from border)", ge=1, le=1024)


class SplitOnPartVolume(MeasurementMethodBase):
    text_info = (
        "distance splitting volume",
        "Split mask on parts and then calculate volume of cross of segmentation and mask part",
    )
    __argument_class__ = SplitOnPartParameters

    @staticmethod
    def calculate_property(
        part_selection, area_array, voxel_size, result_scalar, **kwargs
    ):  # pylint: disable=arguments-differ
        masked = MaskDistanceSplit.split(voxel_size=voxel_size, **kwargs)
        mask = masked == part_selection
        return np.count_nonzero(mask * area_array) * pixel_volume(voxel_size, result_scalar)

    @classmethod
    def get_units(cls, ndim):
        return symbols("{}") ** ndim

    @classmethod
    def get_starting_leaf(cls):
        return super().get_starting_leaf().replace_(area=AreaType.Mask)

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.ROI


class SplitOnPartPixelBrightnessSum(MeasurementMethodBase):
    text_info = (
        "distance splitting pixel brightness sum",
        "Split mask on parts and then calculate pixel brightness sum of cross of segmentation and mask part",
    )
    __argument_class__ = SplitOnPartParameters

    @staticmethod
    def calculate_property(part_selection, channel, area_array, **kwargs):  # pylint: disable=arguments-differ
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
        return super().get_starting_leaf().replace_(area=AreaType.Mask)

    @staticmethod
    def area_type(area: AreaType):
        return AreaType.ROI

    @classmethod
    def need_channel(cls):
        return True


HARALIC_FEATURES = """AngularSecondMoment Contrast Correlation Variance
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()


class HaralickEnum(Enum):
    AngularSecondMoment = "AngularSecondMoment"
    Contrast = "Contrast"
    Correlation = "Correlation"
    Variance = "Variance"
    InverseDifferenceMoment = "InverseDifferenceMoment"
    SumAverage = "SumAverage"
    SumVariance = "SumVariance"
    SumEntropy = "SumEntropy"
    Entropy = "Entropy"
    DifferenceVariance = "DifferenceVariance"
    DifferenceEntropy = "DifferenceEntropy"
    InfoMeas1 = "InfoMeas1"
    InfoMeas2 = "InfoMeas2"

    def index(self) -> int:
        return list(self.__class__).index(self)


def _rescale_image(data: np.ndarray):
    if data.dtype == np.uint8:
        return data
    if np.issubdtype(data.dtype, np.integer) and data.min() >= 0 and data.max() < 255:
        return data.astype(np.uint8)
    min_val = data.min()
    max_val = data.max()
    return ((data - min_val) / ((max_val - min_val) / 254)).astype(np.uint8)


class HaralickParameters(BaseModel):
    feature: HaralickEnum = HaralickEnum.AngularSecondMoment
    distance: int = Field(1, ge=1, le=10)


class Haralick(MeasurementMethodBase):
    __argument_class__ = HaralickParameters

    @classmethod
    def get_units(cls, ndim) -> symbols:
        return "1"

    text_info = "Haralick", "Calculate Haralick features"

    @classmethod
    def need_channel(cls):
        return True

    @classmethod
    def calculate_property(  # pylint: disable=arguments-differ
        cls, area_array, channel, distance, feature, _cache=False, **kwargs
    ):  # pylint: disable=arguments-differ
        if isinstance(feature, str):
            feature = HaralickEnum(feature)
        if _cache := _cache and "_area" in kwargs and "_per_component" in kwargs:
            help_dict: Dict = kwargs["help_dict"]
            _area: AreaType = kwargs["_area"]
            _per_component: PerComponent = kwargs["_per_component"]
            hash_name = hash_fun_call_name(
                Haralick, {"distance": distance}, _area, _per_component, kwargs["channel_num"], kwargs["_component_num"]
            )
            if hash_name not in help_dict:
                help_dict[hash_name] = cls.calculate_haralick(channel, area_array, distance)
            return help_dict[hash_name][feature.index()]

        res = cls.calculate_haralick(channel, area_array, distance)
        return res[feature.index()]

    @staticmethod
    def calculate_haralick(channel, area_array, distance):
        data = channel.copy()
        data[area_array == 0] = 0
        data = _rescale_image(data.squeeze())
        return haralick(data, distance=distance, ignore_zeros=True, return_mean=True)


class ComponentBoundingBox(MeasurementMethodBase):
    text_info = "Component Bounding Box", "bounding box as string"

    @classmethod
    def get_units(cls, ndim):
        return "str"

    @staticmethod
    def calculate_property(bounds_info, _component_num, **kwargs):  # pylint: disable=arguments-differ
        return str(bounds_info[_component_num])

    @classmethod
    def get_starting_leaf(cls):
        return super().get_starting_leaf().replace_(area=AreaType.ROI, per_component=PerComponent.Yes)


class GetROIAnnotationTypeParameters(BaseModel):
    name: str = ""


class GetROIAnnotationType(MeasurementMethodBase):
    text_info = "annotation by name", "Get roi annotation by name"
    __argument_class__ = GetROIAnnotationTypeParameters

    @classmethod
    def get_starting_leaf(cls):
        return Leaf(name=cls.text_info[0], area=AreaType.ROI, per_component=PerComponent.Yes)

    @staticmethod
    def calculate_property(roi_annotation, name, _component_num, **kwargs):  # pylint: disable=arguments-differ
        return str(roi_annotation.get(_component_num, {}).get(name, ""))

    @classmethod
    def get_units(cls, ndim):
        return "str"


class ColocalizationMeasurementParameters(BaseModel):
    channel_fst: Channel = Field(0, title="Channel 1")
    channel_scd: Channel = Field(1, title="Channel 2")
    colocalization: CorrelationEnum = CorrelationEnum.pearson
    randomize: bool = Field(
        False, description="If randomize orders of pixels in one channel", title="Randomize channel"
    )
    randomize_repeat: int = Field(10, description="Number of repetitions for mean_calculate", title="Randomize num")


class ColocalizationMeasurement(MeasurementMethodBase):
    text_info = "Colocalization", "Measurement of colocalization of two channels."
    __argument_class__ = ColocalizationMeasurementParameters

    @staticmethod
    def _calculate_masked(data_1, data_2, colocalization):
        if colocalization == CorrelationEnum.spearman:
            data_1 = data_1.argsort().argsort().astype(float)
            data_2 = data_2.argsort().argsort().astype(float)
            colocalization = CorrelationEnum.pearson
        if colocalization == CorrelationEnum.pearson:
            data_1_mean = np.mean(data_1)
            data_2_mean = np.mean(data_2)
            nominator = np.sum((data_1 - data_1_mean) * (data_2 - data_2_mean))
            numerator = np.sqrt(np.sum((data_1 - data_1_mean) ** 2) * np.sum((data_2 - data_2_mean) ** 2))
            return nominator / numerator
        if colocalization == CorrelationEnum.manders:
            nominator = np.sum(data_1 * data_2)
            numerator = np.sqrt(np.sum(data_1**2) * np.sum(data_2**2))
            return nominator / numerator
        if colocalization == CorrelationEnum.intensity:
            data_1_mean = np.mean(data_1)
            data_2_mean = np.mean(data_2)
            return np.sum((data_1 > data_1_mean) == (data_2 > data_2_mean)) / data_1.size - 0.5

        raise RuntimeError(f"Not supported colocalization method {colocalization}")  # pragma: no cover

    @classmethod
    def calculate_property(  # pylint: disable=arguments-differ
        cls, area_array, colocalization, randomize=False, randomize_repeat=10, channel_fst=0, channel_scd=1, **kwargs
    ):  # pylint: disable=arguments-differ
        mask_binary = area_array > 0
        data_1 = kwargs[f"channel_{channel_fst}"][mask_binary].astype(float)
        data_2 = kwargs[f"channel_{channel_scd}"][mask_binary].astype(float)
        if not randomize:
            return cls._calculate_masked(data_1, data_2, colocalization)
        res_list = []
        for _ in range(randomize_repeat):
            rand_data2 = np.random.default_rng().permutation(data_2)
            res_list.append(cls._calculate_masked(data_1, rand_data2, colocalization))
        return np.mean(res_list)

    @classmethod
    def get_units(cls, ndim) -> symbols:
        return 1


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


class VoxelSize(MeasurementMethodBase):
    text_info = "Voxel size", "Voxel size"

    @classmethod
    def calculate_property(cls, voxel_size, result_scalar, **kwargs):  # pylint: disable=arguments-differ
        return " x ".join([str(x * result_scalar) for x in voxel_size])

    @classmethod
    def get_units(cls, ndim) -> symbols:
        return symbols("str {}")


MEASUREMENT_DICT = Register(suggested_base_class=MeasurementMethodBase)
"""Register with all measurements algorithms"""

MEASUREMENT_DICT.register(Volume)
MEASUREMENT_DICT.register(Diameter)
MEASUREMENT_DICT.register(PixelBrightnessSum, old_names=["Pixel Brightness Sum"])
MEASUREMENT_DICT.register(ComponentBoundingBox)
MEASUREMENT_DICT.register(GetROIAnnotationType)
MEASUREMENT_DICT.register(ComponentsNumber, old_names=["Components Number"])
MEASUREMENT_DICT.register(MaximumPixelBrightness)
MEASUREMENT_DICT.register(MinimumPixelBrightness)
MEASUREMENT_DICT.register(MeanPixelBrightness)
MEASUREMENT_DICT.register(MedianPixelBrightness)
MEASUREMENT_DICT.register(StandardDeviationOfPixelBrightness)
MEASUREMENT_DICT.register(ColocalizationMeasurement)
MEASUREMENT_DICT.register(Moment, old_names=["Moment of inertia"])
MEASUREMENT_DICT.register(FirstPrincipalAxisLength, old_names=["Longest main axis length"])
MEASUREMENT_DICT.register(SecondPrincipalAxisLength, old_names=["Middle main axis length"])
MEASUREMENT_DICT.register(ThirdPrincipalAxisLength, old_names=["Shortest main axis length"])
MEASUREMENT_DICT.register(Compactness)
MEASUREMENT_DICT.register(Sphericity)
MEASUREMENT_DICT.register(Surface)
MEASUREMENT_DICT.register(RimVolume, old_names=["Rim Volume"])
MEASUREMENT_DICT.register(RimPixelBrightnessSum, old_names=["Rim Pixel Brightness Sum"])
MEASUREMENT_DICT.register(ROINeighbourhoodROI)
MEASUREMENT_DICT.register(DistanceMaskROI, old_names=["segmentation distance"])
MEASUREMENT_DICT.register(DistanceROIROI, old_names=["to ROI distance"])
MEASUREMENT_DICT.register(SplitOnPartVolume, old_names=["split on part volume"])
MEASUREMENT_DICT.register(SplitOnPartPixelBrightnessSum, old_names=["split on part pixel brightness sum"])
MEASUREMENT_DICT.register(Voxels)
MEASUREMENT_DICT.register(Haralick)
MEASUREMENT_DICT.register(VoxelSize)
