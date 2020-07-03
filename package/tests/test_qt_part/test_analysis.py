import pytest

from PartSeg.segmentation_analysis.measurement_widget import MeasurementWidget, QMessageBox
from PartSeg.segmentation_analysis.partseg_settings import PartSettings


@pytest.fixture
def part_settings(measurement_profiles, image, tmp_path):
    settings = PartSettings(tmp_path)
    settings.image = image
    for el in measurement_profiles:
        settings.measurement_profiles[el.name] = el
    return settings


class TestMeasurementWidget:
    def test_missed_mask(self, qtbot, analysis_segmentation, part_settings, monkeypatch):
        def simple(*args, **kwargs):
            pass

        monkeypatch.setattr(QMessageBox, "information", simple)
        widget = MeasurementWidget(part_settings)
        qtbot.addWidget(widget)

        assert widget.measurement_type.count() == 3
        part_settings.set_project_info(analysis_segmentation)

        widget.measurement_type.setCurrentIndex(2)
        assert widget.measurement_type.currentIndex() == 0
        assert not widget.recalculate_button.isEnabled()

    def test_base(self, qtbot, analysis_segmentation, part_settings):
        widget = MeasurementWidget(part_settings)
        qtbot.addWidget(widget)

        assert widget.measurement_type.count() == 3
        part_settings.set_project_info(analysis_segmentation)
        widget.measurement_type.setCurrentIndex(1)
        assert widget.recalculate_button.isEnabled()
        widget.recalculate_button.click()
        assert widget.info_field.columnCount() == 2
        assert widget.info_field.rowCount() == 2
        assert widget.info_field.item(1, 1).text() == "4"

    def test_base2(self, qtbot, analysis_segmentation2, part_settings):
        widget = MeasurementWidget(part_settings)
        qtbot.addWidget(widget)

        assert widget.measurement_type.count() == 3
        part_settings.set_project_info(analysis_segmentation2)
        widget.measurement_type.setCurrentIndex(2)
        assert widget.recalculate_button.isEnabled()
        widget.recalculate_button.click()
        assert widget.info_field.columnCount() == 2
        assert widget.info_field.rowCount() == 3
        assert widget.info_field.item(1, 1).text() == "4"
