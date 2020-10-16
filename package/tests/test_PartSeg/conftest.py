import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.segmentation.restartable_segmentation_algorithms import BorderRim, LowerThresholdAlgorithm

try:
    # to ignore problem with test in docker container

    from PartSeg._roi_analysis.partseg_settings import PartSettings
    from PartSeg._roi_mask.main_window import ChosenComponents
    from PartSeg._roi_mask.stack_settings import StackSettings

    @pytest.fixture
    def part_settings(image, tmp_path, measurement_profiles):
        settings = PartSettings(tmp_path)
        settings.image = image
        for el in measurement_profiles:
            settings.measurement_profiles[el.name] = el
        return settings

    @pytest.fixture
    def stack_settings(qtbot, tmp_path):
        settings = StackSettings(tmp_path)
        chose = ChosenComponents()
        qtbot.addWidget(chose)
        settings.chosen_components_widget = chose
        return settings

    @pytest.fixture
    def part_settings_with_project(image, analysis_segmentation2, tmp_path):
        settings = PartSettings(tmp_path)
        settings.image = image
        settings.set_project_info(analysis_segmentation2)
        return settings

    @pytest.fixture
    def border_rim_profile():
        return ROIExtractionProfile("border_profile", BorderRim.get_name(), BorderRim.get_default_values())

    @pytest.fixture
    def lower_threshold_profile():
        return ROIExtractionProfile(
            "lower_profile", LowerThresholdAlgorithm.get_name(), LowerThresholdAlgorithm.get_default_values()
        )

    @pytest.fixture
    def sample_pipeline(border_rim_profile, lower_threshold_profile, mask_property):
        return SegmentationPipeline(
            "sample_pipeline", border_rim_profile, [SegmentationPipelineElement(lower_threshold_profile, mask_property)]
        )


except (RuntimeError, ImportError):
    pass
