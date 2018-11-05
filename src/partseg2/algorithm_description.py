from partseg2.segment_algorithms import LowerThresholdAlgorithm, UpperThresholdAlgorithm, RangeThresholdAlgorithm, \
    LowerThresholdDistanceFlowAlgorithm, UpperThresholdDistanceFlowAlgorithm, LowerThresholdPathFlowAlgorithm, \
    UpperThresholdPathFlowAlgorithm, UpperThresholdPathDistanceFlowAlgorithm, LowerThresholdPathDistanceFlowAlgorithm
from project_utils.algorithms_description import AlgorithmProperty
from copy import deepcopy

from project_utils.image_operations import RadiusType


lower_threshold_algorithm = [AlgorithmProperty("threshold", "Threshold", 10000, (0, 10 ** 6), 100),
                             AlgorithmProperty("minimum_size", "Minimum size (pix)", 8000, (0, 10 ** 6), 1000),
                             AlgorithmProperty("use_gauss", "Use gauss", RadiusType.NO, None),
                             AlgorithmProperty("gauss_radius", "Gauss radius", 1.0, (0, 10), 0.1),
                             AlgorithmProperty("side_connection", "Connect only sides", False, (True, False))]

upper_threshold_algorithm = deepcopy(lower_threshold_algorithm)

range_threshold_algorithm = [AlgorithmProperty("lower_threshold", "Lower threshold", 10000, (0, 10 ** 6), 100),
                             AlgorithmProperty("upper_threshold", "Upper threshold", 10000, (0, 10 ** 6), 100),
                             AlgorithmProperty("minimum_size", "Minimum size (pix)", 8000, (0, 10 ** 6), 1000),
                             AlgorithmProperty("use_gauss", "Use gauss", RadiusType.NO, None),
                             AlgorithmProperty("gauss_radius", "Gauss radius", 1.0, (0, 10), 0.1),
                             AlgorithmProperty("side_connection", "Connect only sides", False, (True, False))]

base_flow_threshold_algorithm = deepcopy(lower_threshold_algorithm)
base_flow_threshold_algorithm.insert(1, AlgorithmProperty("base_threshold", "Base threshold", 10000, (0, 10 ** 6), 100))

# noinspection PyTypeChecker
upper_euclidean_flow_threshold_algorithm = ["May not work proper when voxel is not cube"] + deepcopy(
    base_flow_threshold_algorithm)

# noinspection PyTypeChecker
lower_euclidean_flow_threshold_algorithm = ["May not work proper when voxel is not cube"] + deepcopy(
    base_flow_threshold_algorithm)

lower_path_flow_threshold_algorithm = deepcopy(base_flow_threshold_algorithm)

upper_path_flow_threshold_algorithm = deepcopy(base_flow_threshold_algorithm)

lower_path_euclidean_flow_threshold_algorithm = deepcopy(lower_euclidean_flow_threshold_algorithm)

upper_path_euclidean_flow_threshold_algorithm = deepcopy(upper_euclidean_flow_threshold_algorithm)

part_algorithm_dict = {
    "Lower threshold": (lower_threshold_algorithm, LowerThresholdAlgorithm),
    "Upper threshold": (upper_threshold_algorithm, UpperThresholdAlgorithm),
    "Range threshold": (range_threshold_algorithm, RangeThresholdAlgorithm),
    "Lower threshold euclidean": (lower_euclidean_flow_threshold_algorithm, LowerThresholdDistanceFlowAlgorithm),
    "Upper threshold euclidean": (upper_euclidean_flow_threshold_algorithm, UpperThresholdDistanceFlowAlgorithm),
    "Lower threshold path": (lower_path_flow_threshold_algorithm, LowerThresholdPathFlowAlgorithm),
    "Upper threshold path": (upper_path_flow_threshold_algorithm, UpperThresholdPathFlowAlgorithm),
    "Lower threshold path euclidean": (
        lower_path_euclidean_flow_threshold_algorithm, LowerThresholdPathDistanceFlowAlgorithm),
    "Upper threshold path euclidean": (
        upper_path_euclidean_flow_threshold_algorithm, UpperThresholdPathDistanceFlowAlgorithm)
}


class SegmentationProfile(object):
    def __init__(self, name, algorithm, values):
        self.name = name
        self.algorithm = algorithm
        self.values = values

    def __str__(self):
        return "Segmentation profile name: " + self.name + "\nAlgorithm: " + self.algorithm + "\n" + "\n".join(
            [f"{k.replace('_', ' ')}: {v}" for k, v in self.values.items()])

    def __repr__(self):
        return f"SegmentationProfile(name={self.name}, algorithm={self.algorithm}, values={self.values})"
