import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.segmentation.restartable_segmentation_algorithms import BorderRim, LowerThresholdAlgorithm

try:
    # to ignore problem with test in docker container

    from PartSeg._roi_analysis.partseg_settings import PartSettings
    from PartSeg._roi_mask.main_window import ChosenComponents
    from PartSeg._roi_mask.stack_settings import StackSettings
    from PartSeg.common_gui import napari_image_view

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

    @pytest.fixture(autouse=True)
    def disable_threads_viewer(monkeypatch):
        def _prepare_layers(self, image, parameters, replace):
            self._add_image(napari_image_view._prepare_layers(image, parameters, replace))

        monkeypatch.setattr(napari_image_view.ImageView, "_prepare_layers", _prepare_layers)

        def _add_layer_util(self, index, layer, filters):
            self.viewer.add_layer(layer)

        monkeypatch.setattr(napari_image_view.ImageView, "_add_layer_util", _add_layer_util)

    @pytest.fixture(autouse=True)
    def check_opened_windows(qapp):
        yield
        widgets = qapp.topLevelWidgets()
        for widget in widgets:
            assert not widget.isVisible()


except (RuntimeError, ImportError):
    pass
