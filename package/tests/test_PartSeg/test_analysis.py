import numpy as np
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QApplication, QCheckBox

from PartSeg._roi_analysis.measurement_widget import MeasurementsStorage, MeasurementWidget, QMessageBox
from PartSeg._roi_mask.simple_measurements import SimpleMeasurements
from PartSegCore.analysis.measurement_base import AreaType, PerComponent
from PartSegCore.analysis.measurement_calculation import ComponentsInfo, MeasurementResult


class TestMeasurementWidget:
    def test_missed_mask(self, qtbot, analysis_segmentation, part_settings, monkeypatch):
        def simple(*args, **kwargs):
            pass

        monkeypatch.setattr(QMessageBox, "information", simple)
        widget = MeasurementWidget(part_settings)
        qtbot.addWidget(widget)

        assert widget.measurement_type.count() == 3
        part_settings.set_project_info(analysis_segmentation)

        with qtbot.waitSignal(widget.measurement_type.currentIndexChanged):
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
        widget.horizontal_measurement_present.setChecked(True)
        assert widget.info_field.columnCount() == 2
        assert widget.info_field.rowCount() == 2

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
        widget.horizontal_measurement_present.setChecked(True)
        assert widget.info_field.columnCount() == 3
        assert widget.info_field.rowCount() == 2


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
        assert widget.result_view.columnCount() == len(stack_settings.roi_info.bound_info) + 1


class TestMeasurementsStorage:
    def test_empty(self):
        obj = MeasurementsStorage()
        assert obj.get_size(True) == (0, 0)
        assert obj.get_size(False) == (0, 0)

    def test_base(self):
        info = ComponentsInfo(np.arange(1, 4), np.array([]), {})
        storage = MeasurementResult(info)
        storage["aa"] = 1, "", (PerComponent.No, AreaType.ROI)
        storage["bb"] = [4, 5, 6], "np", (PerComponent.Yes, AreaType.ROI)
        obj = MeasurementsStorage()
        obj.add_measurements(storage)
        assert obj.get_size(True) == (2, 2)
        obj.set_expand(True)
        assert obj.get_size(True) == (3, 4)
        assert obj.get_size(False) == (4, 3)
        obj.set_expand(False)
        assert obj.get_val_as_str(0, 0, True) == "aa"
        assert obj.get_val_as_str(0, 0, False) == "aa"
        assert obj.get_val_as_str(0, 1, True) == "bb"
        assert obj.get_val_as_str(1, 0, False) == "bb"
        assert obj.get_header(True) == ["0", "1"]
        assert obj.get_header(False) == ["Name", "Value"]
        assert obj.get_rows(False) == ["0", "1"]
        assert obj.get_rows(True) == ["Name", "Value"]
        obj.set_expand(True)
        assert obj.get_header(False) == ["Name", "Value", "Value", "Value"]
