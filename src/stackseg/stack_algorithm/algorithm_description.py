from project_utils.segmentation.segmentation_algorithm import ThresholdAlgorithm, ThresholdPreview, \
    AutoThresholdAlgorithm

stack_algorithm_dict = {
    "Threshold": ThresholdAlgorithm,
    "Only Threshold": ThresholdPreview,
    "Auto Threshold": AutoThresholdAlgorithm
}
