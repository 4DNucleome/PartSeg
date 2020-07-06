import pytest

try:
    # to ignore problem with test in docker container

    from PartSeg.segmentation_analysis.partseg_settings import PartSettings
    from PartSeg.segmentation_mask.main_window import ChosenComponents
    from PartSeg.segmentation_mask.stack_settings import StackSettings

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


except (RuntimeError, ImportError):
    pass
