from project_utils.segmentation.restartable_segmentation_algorithms import LowerThresholdAlgorithm, UpperThresholdAlgorithm, RangeThresholdAlgorithm, \
    LowerThresholdDistanceFlowAlgorithm, UpperThresholdDistanceFlowAlgorithm, LowerThresholdPathFlowAlgorithm, \
    UpperThresholdPathFlowAlgorithm, UpperThresholdPathDistanceFlowAlgorithm, LowerThresholdPathDistanceFlowAlgorithm

part_algorithm_dict = {
    "Lower threshold": LowerThresholdAlgorithm,
    "Upper threshold": UpperThresholdAlgorithm,
    "Range threshold": RangeThresholdAlgorithm,
    "Lower threshold euclidean": LowerThresholdDistanceFlowAlgorithm,
    "Upper threshold euclidean": UpperThresholdDistanceFlowAlgorithm,
    "Lower threshold path": LowerThresholdPathFlowAlgorithm,
    "Upper threshold path": UpperThresholdPathFlowAlgorithm,
    "Lower threshold path euclidean": LowerThresholdPathDistanceFlowAlgorithm,
    "Upper threshold path euclidean": UpperThresholdPathDistanceFlowAlgorithm
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
