import numpy as np
from copy import deepcopy
from PartSeg.tiff_image import Image
from PartSeg.utils.segmentation import restartable_segmentation_algorithms as sa
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
    data[0, 25, 40:45, 50] = 50
    data[0, 25, 45:50, 51] = 50
    return Image(data, (100, 50, 50), "")

def get_two_parts_side_reversed():
    data = get_two_parts_array()
    data[0, 25, 40:45, 50] = 50
    data[0, 25, 45:50, 51] = 50
    data = 100 - data
    return Image(data, (100, 50, 50), "")


def empty(*_):
    pass


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


class BaseOneThreshold(BaseThreshold):
    def test_simple(self):
        image = self.get_base_object()
        alg = self.algorithm_class()
        parameters = self.get_parameters()
        alg.set_image(image)
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000, 72000]))  # 30*40*80, 30*30*80

        parameters['threshold']["values"]["threshold"] += self.get_shift()
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 192000  # 30*80*80

    def test_side_connection(self):
        image = self.get_side_object()
        alg = self.algorithm_class()
        parameters = self.get_parameters()
        parameters['side_connection'] = True
        alg.set_image(image)
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000 + 5, 72000 + 5]))

        parameters['side_connection'] = False
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 96000 + 5 + 72000 + 5


class TestLowerThreshold(BaseOneThreshold):
    parameters = {"channel": 0, "minimum_size": 30000, 'threshold': {'name': 'Manual', 'values': {'threshold': 45}},
                  'noise_removal': {'name': 'None', 'values': {}}, 'side_connection': False}
    shift = -6
    get_base_object = staticmethod(get_two_parts)
    get_side_object = staticmethod(get_two_parts_side)
    algorithm_class = sa.LowerThresholdAlgorithm


class TestUpperThreshold(BaseOneThreshold):
    parameters = {"channel": 0, "minimum_size": 30000, 'threshold': {'name': 'Manual', 'values': {'threshold': 55}},
                  'noise_removal': {'name': 'None', 'values': {}}, 'side_connection': False}
    shift = 6
    get_base_object = staticmethod(get_two_parts_reversed)
    get_side_object = staticmethod(get_two_parts_side_reversed)
    algorithm_class = sa.UpperThresholdAlgorithm


class TestRangeThresholdAlgorithm(object):
    def test_simple(self):
        image = get_two_parts()
        alg = sa.RangeThresholdAlgorithm()
        parameters = {'lower_threshold': 45, 'upper_threshold': 60, 'channel': 0, 'minimum_size': 8000,
                      'noise_removal': {'name': 'None', 'values': {}}, 'side_connection': False}
        alg.set_parameters(**parameters)
        alg.set_image(image)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array(
            [30 * 40 * 80 - 20 * 30 * 70, 30 * 30 * 80 - 20 * 20 * 70]))

        parameters['lower_threshold'] -= 6
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 30*80*80 - 20 * 50 * 70

    def test_side_connection(self):
        image = get_two_parts_side()
        alg = sa.RangeThresholdAlgorithm()
        parameters = {'lower_threshold': 45, 'upper_threshold': 60, 'channel': 0, 'minimum_size': 8000,
                      'noise_removal': {'name': 'None', 'values': {}}, 'side_connection': True}
        alg.set_parameters(**parameters)
        alg.set_image(image)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array(
            [30 * 40 * 80 - 20 * 30 * 70 + 5, 30 * 30 * 80 - 20 * 20 * 70 + 5]))
        parameters['side_connection'] = False
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 1
        assert np.bincount(result.segmentation.flat)[1] == 30 * 70 * 80 - 20 * 50 * 70 + 10


class BaseFlowThreshold(BaseThreshold):
    def test_simple(self):
        image = self.get_base_object()
        alg = self.algorithm_class()
        parameters = self.get_parameters()
        alg.set_image(image)
        for key in  sprawl_dict.keys():
            parameters["sprawl_type"] = {'name': key, 'values': {}}
            alg.set_parameters(**parameters)
            result = alg.calculation_run(empty)
            assert result.segmentation.max() == 2
            if not np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000, 72000])):
                print("aaa", key, np.bincount(result.segmentation.flat)[1:])
            assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000, 72000]))  # 30*40*80, 30*30*80


    def test_side_connection(self):
        image = self.get_side_object()
        alg = self.algorithm_class()
        parameters = self.get_parameters()
        parameters['side_connection'] = True
        alg.set_image(image)
        alg.set_parameters(**parameters)
        result = alg.calculation_run(empty)
        assert result.segmentation.max() == 2
        assert np.all(np.bincount(result.segmentation.flat)[1:] == np.array([96000 + 5, 72000 + 5]))


class TestLowerThresholdFlow(BaseFlowThreshold):
    parameters = {"channel": 0, "minimum_size": 30,
                  'threshold': {'name': 'Double Choose',
                                'values': {
                                    'core_threshold': {'name': 'Manual', 'values': {'threshold': 55}},
                                    'base_threshold': {'name': 'Manual', 'values': {'threshold': 45}}}},
                  'noise_removal': {'name': 'None', 'values': {}}, 'side_connection': False,
                  'sprawl_type': {'name': 'Euclidean sprawl', 'values': {}}}
    shift = 6
    get_base_object = staticmethod(get_two_parts)
    get_side_object = staticmethod(get_two_parts_side)
    algorithm_class = sa.LowerThresholdFlowAlgorithm

class TestUpperThresholdFlow(BaseFlowThreshold):
    parameters = {"channel": 0, "minimum_size": 30,
                  'threshold': {'name': 'Double Choose',
                                'values': {
                                    'core_threshold': {'name': 'Manual', 'values': {'threshold': 45}},
                                    'base_threshold': {'name': 'Manual', 'values': {'threshold': 55}}}},
                  'noise_removal': {'name': 'None', 'values': {}}, 'side_connection': False,
                  'sprawl_type': {'name': 'Euclidean sprawl', 'values': {}}}
    shift = 6
    get_base_object = staticmethod(get_two_parts_reversed)
    get_side_object = staticmethod(get_two_parts_side_reversed)
    algorithm_class = sa.UpperThresholdFlowAlgorithm