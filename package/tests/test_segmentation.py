from abc import ABC

import numpy as np
from copy import deepcopy
from typing import Type

from PartSeg.utils.convex_fill import convex_fill, _convex_fill
from PartSegImage import Image
from PartSeg.utils.algorithm_describe_base import SegmentationProfile
from PartSeg.utils.analysis.algorithm_description import analysis_algorithm_dict
from PartSeg.utils.analysis.analysis_utils import SegmentationPipelineElement, SegmentationPipeline
from PartSeg.utils.calculate_pipeline import calculate_pipeline
from PartSeg.utils.image_operations import RadiusType
from PartSeg.utils.mask_create import calculate_mask, MaskProperty
from PartSeg.utils.segmentation import restartable_segmentation_algorithms as sa, SegmentationAlgorithm
from PartSeg.utils.segmentation.noise_filtering import noise_filtering_dict
from PartSeg.utils.segmentation.sprawl import sprawl_dict


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
    pass


def test_base_parameters():
    for key, val in analysis_algorithm_dict.items():
        assert val.get_name() == key
        val: Type[SegmentationAlgorithm]
        obj = val()
        values = val.get_default_values()
        obj.set_parameters(**values)
        parameters = obj.get_segmentation_profile()
        assert parameters.algorithm == key
        assert parameters.values == values


class BaseThreshold(object):
    def get_parameters(self):
        if hasattr(self, "parameters"):
            return deepcopy(self.parameters)
        raise NotImplementedError

    def get_shift(self):
        if hasattr(self, "shift"):
            return deepcopy(self.shift)
        raise NotImplementedError

    def get_base_object(self):
        raise NotImplementedError

    def get_side_object(self):
        raise NotImplementedError

    algorithm_class = None


class BaseOneThreshold(BaseThreshold, ABC):
    def test_simple(self):
        image = self.get_base_object()
        alg: SegmentationAlgorithm = self.algorithm_class()
        parameters = self.get_parameters()
        alg.set_image(image)
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000, 72000]))  # 30*40*80, 30*30*80
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

        parameters['threshold']["values"]["threshold"] += self.get_shift()
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 192000  # 30*80*80
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

    def test_side_connection(self):
        image = self.get_side_object()
        alg: SegmentationAlgorithm = self.algorithm_class()
        parameters = self.get_parameters()
        parameters['side_connection'] = True
        alg.set_image(image)
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000 + 5, 72000 + 5]))
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

        parameters['side_connection'] = False
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 96000 + 5 + 72000 + 5
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()


class TestLowerThreshold(BaseOneThreshold):
    parameters = {"channel": 0, "minimum_size": 30000, 'threshold': {'name': 'Manual', 'values': {'threshold': 45}},
                  'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': False}
    shift = -6
    get_base_object = staticmethod(get_two_parts)
    get_side_object = staticmethod(get_two_parts_side)
    algorithm_class = sa.LowerThresholdAlgorithm


class TestUpperThreshold(BaseOneThreshold):
    parameters = {"channel": 0, "minimum_size": 30000, 'threshold': {'name': 'Manual', 'values': {'threshold': 55}},
                  'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': False}
    shift = 6
    get_base_object = staticmethod(get_two_parts_reversed)
    get_side_object = staticmethod(get_two_parts_side_reversed)
    algorithm_class = sa.UpperThresholdAlgorithm


class TestRangeThresholdAlgorithm(object):
    def test_simple(self):
        image = get_two_parts()
        alg = sa.RangeThresholdAlgorithm()
        parameters = {'lower_threshold': 45, 'upper_threshold': 60, 'channel': 0, 'minimum_size': 8000,
                      'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': False}
        alg.set_parameters(**parameters)
        alg.set_image(image)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array(
            [30 * 40 * 80 - 20 * 30 * 70, 30 * 30 * 80 - 20 * 20 * 70]))
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

        parameters['lower_threshold'] -= 6
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 30*80*80 - 20 * 50 * 70
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

    def test_side_connection(self):
        image = get_two_parts_side()
        alg = sa.RangeThresholdAlgorithm()
        parameters = {'lower_threshold': 45, 'upper_threshold': 60, 'channel': 0, 'minimum_size': 8000,
                      'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': True}
        alg.set_parameters(**parameters)
        alg.set_image(image)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array(
            [30 * 40 * 80 - 20 * 30 * 70 + 5, 30 * 30 * 80 - 20 * 20 * 70 + 5]))
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()

        parameters['side_connection'] = False
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 30 * 70 * 80 - 20 * 50 * 70 + 10
        assert result.parameters.values == parameters
        assert result.parameters.algorithm == alg.get_name()


class BaseFlowThreshold(BaseThreshold, ABC):
    def test_simple(self):
        image = self.get_base_object()
        alg = self.algorithm_class()
        parameters = self.get_parameters()
        alg.set_image(image)
        for key, val in sprawl_dict.items():
            parameters["sprawl_type"] = {'name': key, 'values': val.get_default_values()}
            alg.set_parameters(**parameters)
            result = alg.calculation_run(empty)
            assert result.segmentation.max() == 2
            assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000, 72000]))  # 30*40*80, 30*30*80
            assert result.parameters.values == parameters
            assert result.parameters.algorithm == alg.get_name()
        parameters["threshold"]["values"]["base_threshold"]['values']["threshold"] += self.get_shift()
        for key, val in sprawl_dict.items():
            parameters["sprawl_type"] = {'name': key, 'values': val.get_default_values()}
            alg.set_parameters(**parameters)
            result = alg.calculation_run(empty)
            assert result.segmentation.max() == 2
            assert np.all(np.bincount(result.segmentation.flat)[1:] >= np.array([96000, 72000]))  # 30*40*80, 30*30*80
            assert result.parameters.values == parameters
            assert result.parameters.algorithm == alg.get_name()

    def test_side_connection(self):
        image = self.get_side_object()
        alg = self.algorithm_class()
        parameters = self.get_parameters()
        parameters['side_connection'] = True
        alg.set_image(image)
        for key, val in sprawl_dict.items():
            parameters["sprawl_type"] = {'name': key, 'values': val.get_default_values()}
            alg.set_parameters(**parameters)
            result = alg.calculation_run(empty)
            assert result.segmentation.max() == 2
            assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000 + 5, 72000 + 5]))
            assert result.parameters.values == parameters
            assert result.parameters.algorithm == alg.get_name()


class TestLowerThresholdFlow(BaseFlowThreshold):
    parameters = {"channel": 0, "minimum_size": 30,
                  'threshold': {'name': 'Base/Core',
                                'values': {
                                    'core_threshold': {'name': 'Manual', 'values': {'threshold': 55}},
                                    'base_threshold': {'name': 'Manual', 'values': {'threshold': 45}}}},
                  'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': False,
                  'sprawl_type': {'name': 'Euclidean sprawl', 'values': {}}}
    shift = -6
    get_base_object = staticmethod(get_two_parts)
    get_side_object = staticmethod(get_two_parts_side)
    algorithm_class = sa.LowerThresholdFlowAlgorithm


class TestUpperThresholdFlow(BaseFlowThreshold):
    parameters = {"channel": 0, "minimum_size": 30,
                  'threshold': {'name': 'Base/Core',
                                'values': {
                                    'core_threshold': {'name': 'Manual', 'values': {'threshold': 45}},
                                    'base_threshold': {'name': 'Manual', 'values': {'threshold': 55}}}},
                  'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': False,
                  'sprawl_type': {'name': 'Euclidean sprawl', 'values': {}}}
    shift = 6
    get_base_object = staticmethod(get_two_parts_reversed)
    get_side_object = staticmethod(get_two_parts_side_reversed)
    algorithm_class = sa.UpperThresholdFlowAlgorithm


class TestMaskCreate:
    def test_simple_mask(self):
        mask_array = np.zeros((10, 20, 20), dtype=np.uint8)
        mask_array[3:7, 6:14, 6:14] = 1
        prop = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.NO, max_holes_size=0,
                            save_components=False, clip_to_mask=False)
        new_mask = calculate_mask(prop, mask_array, None, (1, 1, 1))
        assert np.all(new_mask == mask_array)
        mask_array2 = np.copy(mask_array)
        mask_array2[4:6, 8:12, 8:12] = 2
        new_mask = calculate_mask(prop, mask_array2, None, (1, 1, 1))
        assert np.all(new_mask == mask_array)
        prop2 = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.NO, max_holes_size=0,
                             save_components=True, clip_to_mask=False)
        new_mask = calculate_mask(prop2, mask_array2, None, (1, 1, 1))
        assert np.all(new_mask == mask_array2)

    def test_fill_holes(self):
        mask_base_array = np.zeros((20, 30, 30), dtype=np.uint8)
        mask_base_array[4:16, 8:22, 8:22] = 1
        mask1_array = np.copy(mask_base_array)
        mask1_array[4:16, 10:15, 10:15] = 0
        prop = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.R2D, max_holes_size=0,
                            save_components=False, clip_to_mask=False)
        new_mask = calculate_mask(prop, mask1_array, None, (1, 1, 1))
        assert np.all(mask_base_array == new_mask)

        prop = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.R3D, max_holes_size=0,
                            save_components=False, clip_to_mask=False)
        new_mask = calculate_mask(prop, mask1_array, None, (1, 1, 1))
        assert np.all(mask1_array == new_mask)

        mask2_array = np.copy(mask1_array)
        mask2_array[5:15, 10:15, 17:20] = 0
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
        prop1 = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.R3D, max_holes_size=0,
                             save_components=False, clip_to_mask=False)
        prop2 = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.R3D, max_holes_size=0,
                             save_components=True, clip_to_mask=False)
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
        mask_base_array = np.zeros((20, 20, 40), dtype=np.uint8)
        mask_base_array[2:18, 2:18, 4:36] = 1
        mask_base_array[4:16, 4:16, 6:18] = 0
        mask1_array = np.copy(mask_base_array)
        mask1_array[6:14, 6:14, 24:32] = 0

        prop1 = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.R2D, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        prop2 = MaskProperty(dilate=RadiusType.NO, dilate_radius=0, fill_holes=RadiusType.R3D, max_holes_size=530,
                             save_components=True, clip_to_mask=False)

        new_mask = calculate_mask(prop1, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == mask_base_array)
        new_mask = calculate_mask(prop2, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == mask_base_array)

    def test_dilate(self):
        mask_base_array = np.zeros((30, 30, 30), dtype=np.uint8)
        mask_base_array[10:20, 10:20, 10:20] = 1
        prop1 = MaskProperty(dilate=RadiusType.R2D, dilate_radius=-1, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        prop2 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=-1, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        res_array1 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array1[10:20, 11:19, 11:19] = 1
        new_mask = calculate_mask(prop1, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_array1)
        res_array2 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array2[11:19, 11:19, 11:19] = 1
        new_mask = calculate_mask(prop2, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_array2)

        prop1 = MaskProperty(dilate=RadiusType.R2D, dilate_radius=1, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        prop2 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=1, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        res_array1 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array1[10:20, 9:21, 9:21] = 1
        new_mask = calculate_mask(prop1, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_array1)
        res_array2 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array2[9:21, 9:21, 9:21] = 1
        res_array2[(9, 9, 9, 9), (9, 9, 20, 20), (9, 20, 20, 9)] = 0
        res_array2[(20, 20, 20, 20), (9, 9, 20, 20), (9, 20, 20, 9)] = 0
        new_mask = calculate_mask(prop2, mask_base_array, None, (1, 1, 1))
        assert np.all(new_mask == res_array2)

    def test_dilate_spacing_negative(self):
        mask_base_array = np.zeros((30, 30, 30), dtype=np.uint8)
        mask_base_array[10:20, 5:25, 5:25] = 1
        prop1 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=-1, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        prop2 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=-2, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        prop3 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=-3, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        res_array1 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array1[10:20, 6:24, 6:24] = 1
        new_mask = calculate_mask(prop1, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask == res_array1)
        res_array2 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array2[11:19, 7:23, 7:23] = 1
        new_mask = calculate_mask(prop2, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask == res_array2)
        res_array3 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array3[11:19, 8:22, 8:22] = 1
        new_mask = calculate_mask(prop3, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask == res_array3)

    def test_dilate_spacing_positive(self):
        mask_base_array = np.zeros((30, 30, 30), dtype=np.uint8)
        mask_base_array[10:20, 10:20, 10:20] = 1
        prop1 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=1, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        prop2 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=2, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        prop3 = MaskProperty(dilate=RadiusType.R3D, dilate_radius=3, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=False)
        res_array1 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array1[10:20, 9:21, 9:21] = 1
        new_mask = calculate_mask(prop1, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask == res_array1)
        res_array2 = np.zeros((30, 30, 30), dtype=np.uint8)
        res_array2[10:20, 8:22, 8:22] = 1
        res_array2[(9, 20), 9:21, 9:21] = 1
        res_array2[:, (8, 21, 8, 21), (8, 8, 21, 21)] = 0
        new_mask = calculate_mask(prop2, mask_base_array, None, (3, 1, 1))
        assert np.all(new_mask == res_array2)
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
        prop1 = MaskProperty(dilate=RadiusType.NO, dilate_radius=-0, fill_holes=RadiusType.NO, max_holes_size=0,
                             save_components=False, clip_to_mask=False)
        prop2 = MaskProperty(dilate=RadiusType.NO, dilate_radius=-0, fill_holes=RadiusType.NO, max_holes_size=70,
                             save_components=False, clip_to_mask=True)
        new_mask1 = calculate_mask(prop1, mask_base_array, mask2_array, (1, 1, 1))
        new_mask2 = calculate_mask(prop2, mask_base_array, mask2_array, (1, 1, 1))
        assert np.all(new_mask1 == mask_base_array)
        assert np.all(new_mask2 == mask2_array)


# TODO add Border rim and multiple otsu tests


class TestPipeline:
    @staticmethod
    def get_image():
        data = np.zeros((1, 50, 100, 100, 2), dtype=np.uint16)
        data[0, 10:40, 20:80, 20:60, 0] = 10
        data[0, 10:40, 20:80, 40:80, 1] = 10
        return Image(data, (100, 50, 50), "")

    def test_pipeline_simple(self):
        image = self.get_image()
        prop1 = MaskProperty(dilate=RadiusType.NO, dilate_radius=-0, fill_holes=RadiusType.NO, max_holes_size=0,
                             save_components=False, clip_to_mask=False)
        parameters1 = {"channel": 0, "minimum_size": 30, 'threshold': {'name': 'Manual', 'values': {'threshold': 5}},
                       'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': False}
        parameters2 = {"channel": 1, "minimum_size": 30, 'threshold': {'name': 'Manual', 'values': {'threshold': 5}},
                       'noise_filtering': {'name': 'None', 'values': {}}, 'side_connection': False}
        seg_profile1 = SegmentationProfile(name="Unknown", algorithm="Lower threshold", values=parameters1)
        pipeline_element = SegmentationPipelineElement(mask_property=prop1, segmentation=seg_profile1)
        seg_profile2 = SegmentationProfile(name="Unknown", algorithm="Lower threshold", values=parameters2)

        pipeline = SegmentationPipeline(name="test", segmentation=seg_profile2, mask_history=[pipeline_element])
        result = calculate_pipeline(image=image, mask=None, pipeline=pipeline, report_fun=empty)
        result_segmentation = np.zeros((50, 100, 100), dtype=np.uint8)
        result_segmentation[10:40, 20:80, 40:60] = 1
        assert np.all(result.segmentation == result_segmentation)


class TestNoiseFiltering:
    def test_base(self):
        for el in noise_filtering_dict.values():
            data = get_two_parts_array()[0, ..., 0]
            el.noise_filter(data, (1, 1, 1), el.get_default_values())


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
        arr = np.zeros((20, 20), dtype=np.bool)
        assert _convex_fill(arr) is None

