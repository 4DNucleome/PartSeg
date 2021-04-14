import operator
from abc import ABC
from copy import deepcopy
from typing import List, Type, Union

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.analysis.analysis_utils import SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.analysis.calculate_pipeline import calculate_pipeline
from PartSegCore.convex_fill import _convex_fill, convex_fill
from PartSegCore.image_operations import RadiusType
from PartSegCore.mask_create import MaskProperty, calculate_mask
from PartSegCore.roi_info import BoundInfo, ROIInfo
from PartSegCore.segmentation import SegmentationAlgorithm, algorithm_base
from PartSegCore.segmentation import restartable_segmentation_algorithms as sa
from PartSegCore.segmentation.noise_filtering import noise_filtering_dict
from PartSegCore.segmentation.watershed import sprawl_dict
from PartSegImage import Image


def get_two_parts_array():
    data = np.zeros((1, 50, 100, 100, 1), dtype=np.uint16)
    data[0, 10:40, 10:40, 10:90] = 50
    data[0, 10:40, 50:90, 10:90] = 50
    data[0, 15:35, 15:35, 15:85] = 70
    data[0, 15:35, 55:85, 15:85] = 70
    data[0, 10:40, 40:50, 10:90] = 40
    return data


def get_two_parts():
    return Image(get_two_parts_array(), (100, 50, 50), "")


def get_two_parts_reversed():
    data = get_two_parts_array()
    data = 100 - data
    return Image(data, (100, 50, 50), "")


def get_multiple_part_array(part_num):
    data = np.zeros((1, 20, 40, 40 * part_num, 1), dtype=np.uint8)
    data[0, 4:16, 8:32, 8 : 40 * part_num - 8] = 40
    for i in range(part_num):
        data[0, 5:15, 10:30, 40 * i + 10 : 40 * i + 30] = 50
        data[0, 7:13, 15:25, 40 * i + 15 : 40 * i + 25] = 70
    return data


def get_multiple_part(part_num):
    return Image(get_multiple_part_array(part_num), (100, 50, 50), "")


def get_multiple_part_reversed(part_num):
    data = 100 - get_multiple_part_array(part_num)
    return Image(data, (100, 50, 50), "")


def get_two_parts_side():
    data = get_two_parts_array()
    data[0, 25, 40:45, 50] = 49
    data[0, 25, 45:50, 51] = 49
    return Image(data, (100, 50, 50), "")


def get_two_parts_side_reversed():
    data = get_two_parts_array()
    data[0, 25, 40:45, 50] = 49
    data[0, 25, 45:50, 51] = 49
    data = 100 - data
    return Image(data, (100, 50, 50), "")


def empty(_s: str, _i: int):
    """mock function for callback"""


@pytest.mark.parametrize("algorithm_name", analysis_algorithm_dict.keys())
def test_base_parameters(algorithm_name):
    algorithm_class = analysis_algorithm_dict[algorithm_name]
    assert algorithm_class.get_name() == algorithm_name
    algorithm_class: Type[SegmentationAlgorithm]
    obj = algorithm_class()
    values = algorithm_class.get_default_values()
    obj.set_parameters(**values)
    parameters = obj.get_segmentation_profile()
    assert parameters.algorithm == algorithm_name
    assert parameters.values == values


class BaseThreshold:
    def check_result(self, result, sizes, op, parameters):
        assert result.roi.max() == len(sizes)
        assert np.all(op(np.bincount(result.roi.flat)[1:], np.array(sizes)))
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == self.get_algorithm_class().get_name()

    def get_parameters(self) -> dict:
        if hasattr(self, "parameters") and isinstance(self.parameters, dict):
            return deepcopy(self.parameters)
        raise NotImplementedError

    def get_shift(self):
        if hasattr(self, "shift"):
            return deepcopy(self.shift)
        raise NotImplementedError

    @staticmethod
    def get_base_object():
        raise NotImplementedError

    @staticmethod
    def get_side_object():
        raise NotImplementedError

    def get_algorithm_class(self) -> Type[SegmentationAlgorithm]:
        raise NotImplementedError()


class BaseOneThreshold(BaseThreshold, ABC):  # pylint: disable=W0223
    def test_simple(self):
        image = self.get_base_object()
        alg: SegmentationAlgorithm = self.get_algorithm_class()()
        parameters = self.get_parameters()
        alg.set_image(image)
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        self.check_result(result, [96000, 72000], operator.eq, parameters)

        parameters["threshold"]["values"]["threshold"] += self.get_shift()
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        self.check_result(result, [192000], operator.eq, parameters)

    def test_side_connection(self):
        image = self.get_side_object()
        alg: SegmentationAlgorithm = self.get_algorithm_class()()
        parameters = self.get_parameters()
        parameters["side_connection"] = True
        alg.set_image(image)
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        self.check_result(result, [96000 + 5, 72000 + 5], operator.eq, parameters)

        parameters["side_connection"] = False
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        self.check_result(result, [96000 + 5 + 72000 + 5], operator.eq, parameters)


class TestLowerThreshold(BaseOneThreshold):
    parameters = {
        "channel": 0,
        "minimum_size": 30000,
        "threshold": {"name": "Manual", "values": {"threshold": 45}},
        "noise_filtering": {"name": "None", "values": {}},
        "side_connection": False,
    }
    shift = -6

    @staticmethod
    def get_base_object():
        return get_two_parts()

    @staticmethod
    def get_side_object():
        return get_two_parts_side()

    def get_algorithm_class(self) -> Type[SegmentationAlgorithm]:
        return sa.LowerThresholdAlgorithm


class TestUpperThreshold(BaseOneThreshold):
    parameters = {
        "channel": 0,
        "minimum_size": 30000,
        "threshold": {"name": "Manual", "values": {"threshold": 55}},
        "noise_filtering": {"name": "None", "values": {}},
        "side_connection": False,
    }
    shift = 6

    @staticmethod
    def get_base_object():
        return get_two_parts_reversed()

    @staticmethod
    def get_side_object():
        return get_two_parts_side_reversed()

    def get_algorithm_class(self) -> Type[SegmentationAlgorithm]:
        return sa.UpperThresholdAlgorithm


class TestRangeThresholdAlgorithm:
    def test_simple(self):
        image = get_two_parts()
        alg = sa.RangeThresholdAlgorithm()
        parameters = {
            "lower_threshold": 45,
            "upper_threshold": 60,
            "channel": 0,
            "minimum_size": 8000,
            "noise_filtering": {"name": "None", "values": {}},
            "side_connection": False,
        }
        alg.set_parameters(**parameters)
        alg.set_image(image)
        result = alg.calculation_run(empty)
        assert np.max(result.roi) == 2
        assert np.all(
            np.bincount(result.roi.flat)[1:] == np.array([30 * 40 * 80 - 20 * 30 * 70, 30 * 30 * 80 - 20 * 20 * 70])
        )
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

        parameters["lower_threshold"] -= 6
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert np.max(result.roi) == 1
        assert np.bincount(result.roi.flat)[1] == 30 * 80 * 80 - 20 * 50 * 70
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

    def test_side_connection(self):
        image = get_two_parts_side()
        alg = sa.RangeThresholdAlgorithm()
        parameters = {
            "lower_threshold": 45,
            "upper_threshold": 60,
            "channel": 0,
            "minimum_size": 8000,
            "noise_filtering": {"name": "None", "values": {}},
            "side_connection": True,
        }
        alg.set_parameters(**parameters)
        alg.set_image(image)
        result = alg.calculation_run(empty)
        assert np.max(result.roi) == 2
        assert np.all(
            np.bincount(result.roi.flat)[1:]
            == np.array([30 * 40 * 80 - 20 * 30 * 70 + 5, 30 * 30 * 80 - 20 * 20 * 70 + 5])
        )
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

        parameters["side_connection"] = False
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert np.max(result.roi) == 1
        assert np.bincount(result.roi.flat)[1] == 30 * 70 * 80 - 20 * 50 * 70 + 10
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()


class BaseFlowThreshold(BaseThreshold, ABC):  # pylint: disable=W0223
    @pytest.mark.parametrize("sprawl_algorithm_name", sprawl_dict.keys())
    @pytest.mark.parametrize("compare_op", [operator.eq, operator.ge])
    @pytest.mark.parametrize("components", [2] + list(range(3, 15, 2)))
    def test_multiple(self, sprawl_algorithm_name, compare_op, components):
        alg = self.get_algorithm_class()()
        parameters = self.get_parameters()
        image = self.get_multiple_part(components)
        alg.set_image(image)
        sprawl_algorithm = sprawl_dict[sprawl_algorithm_name]
        parameters["sprawl_type"] = {"name": sprawl_algorithm_name, "values": sprawl_algorithm.get_default_values()}
        if compare_op(1, 0):
            parameters["threshold"]["values"]["base_threshold"]["values"]["threshold"] += self.get_shift()
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        self.check_result(result, [4000] * components, compare_op, parameters)

    @pytest.mark.parametrize("algorithm_name", sprawl_dict.keys())
    def test_side_connection(self, algorithm_name):
        image = self.get_side_object()
        alg = self.get_algorithm_class()()
        parameters = self.get_parameters()
        parameters["side_connection"] = True
        alg.set_image(image)
        val = sprawl_dict[algorithm_name]
        parameters["sprawl_type"] = {"name": algorithm_name, "values": val.get_default_values()}
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        self.check_result(result, [96000 + 5, 72000 + 5], operator.eq, parameters)

    def get_multiple_part(self, parts_num):
        raise NotImplementedError


class TestLowerThresholdFlow(BaseFlowThreshold):
    parameters = {
        "channel": 0,
        "minimum_size": 30,
        "threshold": {
            "name": "Base/Core",
            "values": {
                "core_threshold": {"name": "Manual", "values": {"threshold": 55}},
                "base_threshold": {"name": "Manual", "values": {"threshold": 45}},
            },
        },
        "noise_filtering": {"name": "None", "values": {}},
        "side_connection": False,
        "sprawl_type": {"name": "Euclidean sprawl", "values": {}},
    }
    shift = -6
    get_base_object = staticmethod(get_two_parts)
    get_side_object = staticmethod(get_two_parts_side)
    get_multiple_part = staticmethod(get_multiple_part)

    def get_algorithm_class(self) -> Type[SegmentationAlgorithm]:
        return sa.LowerThresholdFlowAlgorithm


class TestUpperThresholdFlow(BaseFlowThreshold):
    parameters = {
        "channel": 0,
        "minimum_size": 30,
        "threshold": {
            "name": "Base/Core",
            "values": {
                "core_threshold": {"name": "Manual", "values": {"threshold": 45}},
                "base_threshold": {"name": "Manual", "values": {"threshold": 55}},
            },
        },
        "noise_filtering": {"name": "None", "values": {}},
        "side_connection": False,
        "sprawl_type": {"name": "Euclidean sprawl", "values": {}},
    }
    shift = 6
    get_base_object = staticmethod(get_two_parts_reversed)
    get_side_object = staticmethod(get_two_parts_side_reversed)
    get_multiple_part = staticmethod(get_multiple_part_reversed)

    def get_algorithm_class(self) -> Type[SegmentationAlgorithm]:
        return sa.UpperThresholdFlowAlgorithm


class TestMaskCreate:
    def test_simple_mask(self):
        mask_array = np.zeros((10, 20, 20), dtype=np.uint8)
        mask_array[3:7, 6:14, 6:14] = 1
        prop = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.NO,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
        )
        new_mask = calculate_mask(prop, mask_array, None, (1, 1, 1))
        assert np.all(new_mask == mask_array)
        mask_array2 = np.copy(mask_array)
        mask_array2[4:6, 8:12, 8:12] = 2
        new_mask = calculate_mask(prop, mask_array2, None, (1, 1, 1))
        assert np.all(new_mask == mask_array)
        prop2 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.NO,
            max_holes_size=0,
            save_components=True,
            clip_to_mask=False,
        )
        new_mask = calculate_mask(prop2, mask_array2, None, (1, 1, 1))
        assert np.all(new_mask == mask_array2)

    def test_fill_holes(self):
        mask_base_array = np.zeros((1, 20, 30, 30), dtype=np.uint8)
        mask_base_array[:, 4:16, 8:22, 8:22] = 1
        mask1_array = np.copy(mask_base_array)
        mask1_array[:, 4:16, 10:15, 10:15] = 0
        prop = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.R2D,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
        )
        new_mask = calculate_mask(prop, mask1_array, None, (1, 1, 1))
        assert np.all(mask_base_array == new_mask)

        prop = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.R3D,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
        )
        new_mask = calculate_mask(prop, mask1_array, None, (1, 1, 1))
        assert np.all(mask1_array == new_mask)

        mask2_array = np.copy(mask1_array)
        mask2_array[:, 5:15, 10:15, 17:20] = 0
        new_mask = calculate_mask(prop, mask2_array, None, (1, 1, 1))
        assert np.all(mask1_array == new_mask)

    def test_fill_holes_components(self):
        mask_base_array = np.zeros((20, 30, 30), dtype=np.uint8)
        mask_base_array[4:16, 6:15, 6:24] = 1
        mask_base_array[4:16, 15:24, 6:24] = 2
        res_mask1 = (mask_base_array > 0).astype(np.uint8)
        res_mask2 = np.copy(mask_base_array)
        mask_base_array[6:14, 8:12, 8:22] = 0
        mask_base_array[6:14, 18:22, 8:22] = 0
        prop1 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.R3D,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
        )
        prop2 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.R3D,
            max_holes_size=0,
            save_components=True,
            clip_to_mask=False,
        )
        new_mask = calculate_mask(prop1, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_mask1)
        new_mask = calculate_mask(prop2, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_mask2)

        mask_base_array[6:14, 14:16, 8:22] = 0
        res_mask2[6:14, 14:16, 8:22] = 0
        new_mask = calculate_mask(prop1, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_mask1)
        new_mask = calculate_mask(prop2, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_mask2)

    def test_fill_holes_size(self):
        mask_base_array = np.zeros((1, 20, 20, 40), dtype=np.uint8)
        mask_base_array[0, 2:18, 2:18, 4:36] = 1
        mask_base_array[0, 4:16, 4:16, 6:18] = 0
        mask1_array = np.copy(mask_base_array)
        mask1_array[0, 6:14, 6:14, 24:32] = 0

        prop1 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.R2D,
            max_holes_size=70,
            save_components=False,
            clip_to_mask=False,
        )
        prop2 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.R3D,
            max_holes_size=530,
            save_components=True,
            clip_to_mask=False,
        )

        new_mask = calculate_mask(prop1, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == mask_base_array)
        new_mask = calculate_mask(prop2, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == mask_base_array)

    @pytest.mark.parametrize("radius_type", [RadiusType.R2D, RadiusType.R3D])
    @pytest.mark.parametrize("radius", [1, -1])
    @pytest.mark.parametrize("time", [1, 2, 5])
    def test_dilate(self, radius_type, radius, time):
        mask_base_array = np.zeros((time, 30, 30, 30), dtype=np.uint8)
        mask_base_array[:, 10:20, 10:20, 10:20] = 1
        prop1 = MaskProperty(
            dilate=radius_type,
            dilate_radius=radius,
            fill_holes=RadiusType.NO,
            max_holes_size=70,
            save_components=False,
            clip_to_mask=False,
        )
        res_array1 = np.zeros((1, 30, 30, 30), dtype=np.uint8)
        slices: List[Union[int, slice]] = [slice(None)] * 4
        for i in range(1, 4):
            slices[i] = slice(10 - radius, 20 + radius)
        if radius_type == RadiusType.R2D:
            slices[1] = slice(10, 20)
        res_array1[tuple(slices)] = 1
        if radius_type == RadiusType.R3D:
            res_array1[:, (9, 9, 9, 9), (9, 9, 20, 20), (9, 20, 20, 9)] = 0
            res_array1[:, (20, 20, 20, 20), (9, 9, 20, 20), (9, 20, 20, 9)] = 0
        new_mask = calculate_mask(prop1, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_array1)

    @pytest.mark.parametrize("radius", [-1, -2, -3])
    def test_dilate_spacing_negative(self, radius):
        mask_base_array = np.zeros((1, 30, 30, 30), dtype=np.uint8)
        mask_base_array[:, 10:20, 5:25, 5:25] = 1
        res_array1 = np.zeros((1, 30, 30, 30), dtype=np.uint8)
        s = slice(10, 20) if radius == -1 else slice(11, 19)
        res_array1[:, s, 5 - radius : 25 + radius, 5 - radius : 25 + radius] = 1

        prop1 = MaskProperty(
            dilate=RadiusType.R3D,
            dilate_radius=radius,
            fill_holes=RadiusType.NO,
            max_holes_size=70,
            save_components=False,
            clip_to_mask=False,
        )
        new_mask = calculate_mask(prop1, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask == res_array1)

    def test_dilate_spacing_positive(self):
        mask_base_array = np.zeros((1, 30, 30, 30), dtype=np.uint8)
        mask_base_array[:, 10:20, 10:20, 10:20] = 1
        prop1 = MaskProperty(
            dilate=RadiusType.R3D,
            dilate_radius=1,
            fill_holes=RadiusType.NO,
            max_holes_size=70,
            save_components=False,
            clip_to_mask=False,
        )
        prop2 = MaskProperty(
            dilate=RadiusType.R3D,
            dilate_radius=2,
            fill_holes=RadiusType.NO,
            max_holes_size=70,
            save_components=False,
            clip_to_mask=False,
        )
        prop3 = MaskProperty(
            dilate=RadiusType.R3D,
            dilate_radius=3,
            fill_holes=RadiusType.NO,
            max_holes_size=70,
            save_components=False,
            clip_to_mask=False,
        )
        res_array1 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array1[10:20, 9:21, 9:21] = 1
        new_mask = calculate_mask(prop1, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask[0] == res_array1)
        res_array2 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array2[10:20, 8:22, 8:22] = 1
        res_array2[(9, 20), 9:21, 9:21] = 1
        res_array2[:, (8, 21, 8, 21), (8, 8, 21, 21)] = 0
        new_mask = calculate_mask(prop2, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask[0] == res_array2)
        res_array3 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array3[(9, 20), 8:22, 9:21] = 1
        res_array3[(9, 9, 20, 20), 9:21, (8, 21, 8, 21)] = 1

        res_array3[10:20, 7:23, 9:21] = 1
        res_array3[10:20, 8:22, (8, 21)] = 1
        res_array3[10:20, 9:21, (7, 22)] = 1
        new_mask = calculate_mask(prop3, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask == res_array3)

    def test_clip_mask(self):
        mask_base_array = np.zeros((30, 30, 30), dtype=np.uint8)
        mask_base_array[10:20, 10:20, 10:20] = 1
        mask2_array = np.copy(mask_base_array)
        mask2_array[13:17, 13:17, 13:17] = 0
        prop1 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=-0,
            fill_holes=RadiusType.NO,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
        )
        prop2 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=-0,
            fill_holes=RadiusType.NO,
            max_holes_size=70,
            save_components=False,
            clip_to_mask=True,
        )
        new_mask1 = calculate_mask(prop1, mask_base_array, mask2_array, (1, 1, 1))
        new_mask2 = calculate_mask(prop2, mask_base_array, mask2_array, (1, 1, 1))
        assert np.all(new_mask1 == mask_base_array)
        assert np.all(new_mask2 == mask2_array)

    def test_reversed_mask(self):
        mask_base_array = np.zeros((30, 30, 30), dtype=np.uint8)
        mask_base_array[10:20, 10:20, 10:20] = 1
        mask2_array = np.copy(mask_base_array)
        mask2_array[13:17, 13:17, 13:17] = 0
        prop1 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=-0,
            fill_holes=RadiusType.NO,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
            reversed_mask=False,
        )
        prop2 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=-0,
            fill_holes=RadiusType.NO,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
            reversed_mask=True,
        )
        new_mask1 = calculate_mask(prop1, mask_base_array, mask2_array, (1, 1, 1))
        new_mask2 = calculate_mask(prop2, mask_base_array, mask2_array, (1, 1, 1))
        assert np.all(new_mask1 == mask_base_array)
        assert np.all(new_mask2 == (mask_base_array == 0))


# TODO add Border rim and multiple otsu tests


class TestPipeline:
    @staticmethod
    def get_image():
        data = np.zeros((1, 50, 100, 100, 2), dtype=np.uint16)
        data[0, 10:40, 20:80, 20:60, 0] = 10
        data[0, 10:40, 20:80, 40:80, 1] = 10
        return Image(data, (100, 50, 50), "")

    @pytest.mark.parametrize("use_mask", [True, False])
    def test_pipeline_simple(self, use_mask):
        image = self.get_image()
        prop1 = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=-0,
            fill_holes=RadiusType.NO,
            max_holes_size=0,
            save_components=False,
            clip_to_mask=False,
        )
        parameters1 = {
            "channel": 0,
            "minimum_size": 30,
            "threshold": {"name": "Manual", "values": {"threshold": 5}},
            "noise_filtering": {"name": "None", "values": {}},
            "side_connection": False,
        }
        parameters2 = {
            "channel": 1,
            "minimum_size": 30,
            "threshold": {"name": "Manual", "values": {"threshold": 5}},
            "noise_filtering": {"name": "None", "values": {}},
            "side_connection": False,
        }
        seg_profile1 = ROIExtractionProfile(name="Unknown", algorithm="Lower threshold", values=parameters1)
        pipeline_element = SegmentationPipelineElement(mask_property=prop1, segmentation=seg_profile1)
        seg_profile2 = ROIExtractionProfile(name="Unknown", algorithm="Lower threshold", values=parameters2)

        pipeline = SegmentationPipeline(name="test", segmentation=seg_profile2, mask_history=[pipeline_element])
        mask = np.ones(image.get_channel(0).shape, dtype=np.uint8) if use_mask else None
        result = calculate_pipeline(image=image, mask=mask, pipeline=pipeline, report_fun=empty)
        result_segmentation = np.zeros((50, 100, 100), dtype=np.uint8)
        result_segmentation[10:40, 20:80, 40:60] = 1
        assert np.all(result.roi_info.roi == result_segmentation)


class TestNoiseFiltering:
    @pytest.mark.parametrize("algorithm_name", noise_filtering_dict.keys())
    def test_base(self, algorithm_name):
        noise_remove_algorithm = noise_filtering_dict[algorithm_name]
        data = get_two_parts_array()[0, ..., 0]
        noise_remove_algorithm.noise_filter(data, (1, 1, 1), noise_remove_algorithm.get_default_values())


class TestConvexFill:
    def test_simple(self):
        arr = np.zeros((30, 30, 30), dtype=np.uint8)
        arr[10:-10, 10:-10, 10:-10] = 1
        res = convex_fill(arr)
        assert np.all(res == arr)
        arr = np.zeros((30, 60, 30), dtype=np.uint8)
        arr[10:-10, 10:-40, 10:-10] = 1
        arr[10:-10, -30:-10, 10:-10] = 2
        res = convex_fill(arr)
        assert np.all(res == arr)

    def test_missing_value(self):
        arr = np.zeros((30, 30, 30), dtype=np.uint8)
        arr[10:-10, 10:-10, 10:-10] = 2
        res = convex_fill(arr)
        assert np.all(res == arr)
        arr = np.zeros((30, 60, 30), dtype=np.uint8)
        arr[10:-10, 10:-40, 10:-10] = 1
        arr[10:-10, -30:-10, 10:-10] = 3
        res = convex_fill(arr)
        assert np.all(res == arr)

    def test_fill(self):
        arr = np.zeros((30, 30, 30), dtype=np.uint8)
        arr[10:-10, 10:-10, 10:-10] = 1
        arr2 = np.copy(arr)
        arr2[15:-15, 15:-15, 15:-15] = 0
        res = convex_fill(arr2)
        assert np.all(res == arr)
        arr2[15:-15, 15:-10, 15:-15] = 0
        res = convex_fill(arr2)
        assert np.all(res == arr)
        arr = np.zeros((30, 60, 30), dtype=np.uint8)
        arr[10:-10, 10:-40, 10:-10] = 1
        arr[10:-10, -30:-10, 10:-10] = 2
        arr2 = np.copy(arr)
        arr2[15:-15, 15:-15, 20:-10] = 0
        arr2[15:-15, -30:-10, 20:-10] = 0
        res = convex_fill(arr2)
        assert np.all(res == arr)

    def test_removed_object(self):
        arr = np.zeros((30, 30, 30), dtype=np.uint8)
        arr[10:-10, 10:-10, 10:-10] = 1
        arr2 = np.copy(arr)
        arr2[15:-15, 15:-15, 15:-15] = 2
        res = convex_fill(arr2)
        assert np.all(res == arr)
        arr2[15:-15, 15:-10, 15:-15] = 2
        res = convex_fill(arr2)
        assert np.all(res == arr)

    def test__convex_fill(self):
        arr = np.zeros((20, 20), dtype=bool)
        assert _convex_fill(arr) is None


class TestSegmentationInfo:
    def test_none(self):
        si = ROIInfo(None)
        assert si.roi is None
        assert len(si.bound_info) == 0
        assert len(si.sizes) == 0

    def test_empty(self):
        si = ROIInfo(np.zeros((10, 10), dtype=np.uint8))
        assert np.all(si.roi == 0)
        assert len(si.bound_info) == 0
        assert len(si.sizes) == 1

    @pytest.mark.parametrize("num", [1, 5])
    def test_simple(self, num):
        data = np.zeros((10, 10), dtype=np.uint8)
        data[2:8, 2:8] = num
        si = ROIInfo(data)
        assert len(si.bound_info) == 1
        assert num in si.bound_info
        assert isinstance(si.bound_info[num], BoundInfo)
        assert np.all(si.bound_info[num].lower == [2, 2])
        assert np.all(si.bound_info[num].upper == [7, 7])
        assert len(si.sizes) == num + 1
        assert np.all(si.sizes[1:num] == 0)
        assert si.sizes[num] == 36

    @pytest.mark.parametrize("dims", [3, 5, 6])
    def test_more_dims(self, dims):
        si = ROIInfo(np.ones((10,) * dims, dtype=np.uint8))
        assert len(si.bound_info[1].lower) == dims
        assert len(si.bound_info[1].upper) == dims
        assert np.all(si.bound_info[1].lower == 0)
        assert np.all(si.bound_info[1].upper == 9)
        assert len(si.sizes) == 2
        assert np.all(si.sizes == [0, 10 ** dims])

    @pytest.mark.parametrize("comp_num", [2, 4, 8])
    def test_multiple_components(self, comp_num):
        data = np.zeros((10 * comp_num, 10), dtype=np.uint8)
        for i in range(comp_num):
            data[i * 10 + 2 : i * 10 + 8, 2:8] = i + 1
        si = ROIInfo(data)
        assert len(si.bound_info) == comp_num
        assert set(si.bound_info.keys()) == set(range(1, comp_num + 1))
        for i in range(comp_num):
            assert np.all(si.bound_info[i + 1].lower == [i * 10 + 2, 2])
            assert np.all(si.bound_info[i + 1].upper == [i * 10 + 7, 7])
        assert len(si.sizes) == comp_num + 1
        assert np.all(si.sizes[1:] == 36)

        data[-1, 8] = 1
        si = ROIInfo(data)
        assert np.all(si.bound_info[1].lower == 2)
        assert np.all(si.bound_info[1].upper == [10 * comp_num - 1, 8])


def test_bound_info():
    bi = BoundInfo(lower=np.array([1, 1, 1]), upper=np.array([5, 5, 5]))
    assert np.all(bi.box_size() == 5)
    assert len(bi.box_size()) == 3
    assert len(bi.get_slices()) == 3
    assert np.all([x == slice(1, 6) for x in bi.get_slices()])


def test_dict_repr(monkeypatch):
    assert algorithm_base.dict_repr({}) == repr({})
    assert algorithm_base.dict_repr({1: 1}) == repr({1: 1})
    assert algorithm_base.dict_repr({1: 1, 2: {2: 2}}) == repr({1: 1, 2: {2: 2}})

    count = [0]

    def _repr(x):
        count[0] += 1
        return ""

    monkeypatch.setattr(algorithm_base, "numpy_repr", _repr)
    algorithm_base.dict_repr({1: np.zeros(5)})
    assert count[0] == 1
    count = [0]
    algorithm_base.dict_repr({1: np.zeros(5), 2: {1: np.zeros(5)}})
    assert count[0] == 2
