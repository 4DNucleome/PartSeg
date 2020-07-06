from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QCheckBox

from PartSeg.segmentation_analysis.measurement_widget import MeasurementWidget, QMessageBox
from PartSeg.segmentation_mask.simple_measurements import SimpleMeasurements


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


class TestSimpleMeasurementsWidget:
    def test_base(self, stack_settings, stack_segmentation1, qtbot):
        widget = SimpleMeasurements(stack_settings)
        qtbot.addWidget(widget)
        stack_settings.set_project_info(stack_segmentation1)
        widget.show()
        event = QEvent(QEvent.WindowActivate)
        QApplication.sendEvent(widget, event)
        assert widget.measurement_layout.count() > 2
        for i in range(2, widget.measurement_layout.count()):
            chk = widget.measurement_layout.itemAt(i).widget()
            assert isinstance(chk, QCheckBox)
            chk.setChecked(True)
        widget.calculate()
        assert widget.result_view.rowCount() == widget.measurement_layout.count() - 1
        assert widget.result_view.columnCount() == len(stack_settings.segmentation_info.bound_info) + 1
