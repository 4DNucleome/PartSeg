from qtpy.QtCore import QByteArray, QEvent
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PartSeg._roi_mask.stack_settings import StackSettings
from PartSeg.common_gui.universal_gui_part import ChannelComboBox, EnumComboBox
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore import Units
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, PerComponent
from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT, MeasurementProfile, MeasurementResult


class SimpleMeasurements(QWidget):
    def __init__(self, settings: StackSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self.calculate)
        self.result_view = QTableWidget()
        self.channel_select = ChannelComboBox()
        self.units_select = EnumComboBox(Units)
        self.units_select.set_value(self.settings.get("simple_measurements.units", Units.nm))
        self.units_select.currentIndexChanged.connect(self.change_units)

        layout = QHBoxLayout()
        self.measurement_layout = QVBoxLayout()
        l1 = QHBoxLayout()
        l1.addWidget(QLabel("Units"))
        l1.addWidget(self.units_select)
        self.measurement_layout.addLayout(l1)
        l2 = QHBoxLayout()
        l2.addWidget(QLabel("Channel"))
        l2.addWidget(self.channel_select)
        self.measurement_layout.addLayout(l2)
        layout.addLayout(self.measurement_layout)
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_view)
        result_layout.addWidget(self.calculate_btn)
        layout.addLayout(result_layout)
        self.setLayout(layout)
        self.setWindowTitle("Measurement")
        if self.window() == self:
            try:
                geometry = self.settings.get_from_profile("simple_measurement_window_geometry")
                self.restoreGeometry(QByteArray.fromHex(bytes(geometry, "ascii")))
            except KeyError:
                pass

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Save geometry if widget is used as standalone window.
        """
        if self.window() == self:
            self.settings.set_in_profile(
                "simple_measurement_window_geometry", self.saveGeometry().toHex().data().decode("ascii")
            )
        super().closeEvent(event)

    def calculate(self):
        if self.settings.roi is None:
            QMessageBox.warning(self, "No segmentation", "need segmentation to work")
            return
        to_calculate = []
        for i in range(2, self.measurement_layout.count()):
            # noinspection PyTypeChecker
            chk: QCheckBox = self.measurement_layout.itemAt(i).widget()
            if chk.isChecked():
                leaf: Leaf = MEASUREMENT_DICT[chk.text()].get_starting_leaf()
                to_calculate.append(leaf.replace_(per_component=PerComponent.Yes, area=AreaType.ROI))
        if not to_calculate:
            QMessageBox.warning(self, "No measurement", "Select at least one measurement")
            return

        profile = MeasurementProfile("", [MeasurementEntry(x.name, x) for x in to_calculate])

        dial = ExecuteFunctionDialog(
            profile.calculate,
            kwargs={
                "image": self.settings.image,
                "channel_num": self.channel_select.get_value(),
                "roi": self.settings.roi_info,
                "result_units": self.units_select.get_value(),
            },
        )
        dial.exec()
        result: MeasurementResult = dial.get_result()
        values = result.get_separated()
        labels = result.get_labels()
        self.result_view.clear()
        self.result_view.setColumnCount(len(values) + 1)
        self.result_view.setRowCount(len(labels))
        for i, val in enumerate(labels):
            self.result_view.setItem(i, 0, QTableWidgetItem(val))
        for j, values_list in enumerate(values):
            for i, val in enumerate(values_list):
                self.result_view.setItem(i, j + 1, QTableWidgetItem(str(val)))

    def _clean_measurements(self):
        selected = set()
        for _ in range(self.measurement_layout.count() - 2):
            # noinspection PyTypeChecker
            chk: QCheckBox = self.measurement_layout.takeAt(2).widget()
            if chk.isChecked():
                selected.add(chk.text())
            chk.deleteLater()
        return selected

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.WindowActivate:
            self.channel_select.change_channels_num(self.settings.image.channels)
            selected = self._clean_measurements()
            for val in MEASUREMENT_DICT.values():
                area = val.get_starting_leaf().area
                pc = val.get_starting_leaf().per_component
                if (
                    val.get_fields()
                    or (area is not None and area != AreaType.ROI)
                    or (pc is not None and pc != PerComponent.Yes)
                ):
                    continue
                text = val.get_name()
                chk = QCheckBox(text)
                if text in selected:
                    chk.setChecked(True)
                self.measurement_layout.addWidget(chk)
        return super().event(event)

    def change_units(self):
        self.settings.set("simple_measurements.units", self.units_select.get_value())
