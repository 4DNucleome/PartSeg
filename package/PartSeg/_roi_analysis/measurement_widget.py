import locale
import os
from enum import Enum
from typing import List, Tuple

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent, QResizeEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QBoxLayout,
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

from PartSeg.common_backend.progress_thread import ExecuteFunctionThread
from PartSegCore.analysis.measurement_calculation import MeasurementProfile, MeasurementResult
from PartSegCore.universal_const import Units

from ..common_gui.searchable_combo_box import SearchCombBox
from ..common_gui.universal_gui_part import ChannelComboBox, EnumComboBox
from ..common_gui.waiting_dialog import WaitingDialog
from .partseg_settings import PartSettings


class FileNamesEnum(Enum):
    No = 1
    Short = 2
    Full = 3

    def __str__(self):
        return self.name


class MeasurementsStorage:
    """class for storage measurements result"""

    def __init__(self):
        self.header = []
        self.max_rows = 0
        self.content = []
        self.measurements = []
        self.expand = False
        self.show_units = False

    def clear(self):
        """clear storage"""
        self.header = []
        self.max_rows = 0
        self.content = []
        self.measurements: List[MeasurementResult] = []

    def get_size(self, save_orientation: bool):
        if save_orientation:
            return self.max_rows, len(self.header)

        return len(self.header), self.max_rows

    def set_expand(self, expand):
        if self.expand != expand:
            self.expand = expand
            self.refresh()

    def set_show_units(self, show_units):
        if self.show_units != show_units:
            self.show_units = show_units
            self.refresh()

    def refresh(self):
        self.header = []
        self.content = []
        self.max_rows = 0
        previous_labels = []
        previous_units = []
        for data in self.measurements:
            labels = data.get_labels(self.expand)
            if labels != previous_labels:
                self.content.append(labels)
                self.header.append("Name")
                previous_labels = labels
            if self.expand:
                values = data.get_separated()
                units = data.get_units()
            else:
                values, units = zip(*list(data.values()))
                values = [values]
            self.max_rows = max(self.max_rows, len(values[0]))
            self.content.extend(values)
            self.header.extend(["Value" for _ in range(len(values))])
            if self.show_units and units != previous_units:
                previous_units = units
                self.content.append(previous_units)
                self.header.append("Units")

    def add_measurements(self, data: MeasurementResult):
        self.measurements.append(data)
        self.refresh()

    def get_val_as_str(self, x: int, y: int, save_orientation: bool) -> str:
        """get value from given index"""
        if not save_orientation:
            x, y = y, x
        if len(self.content) <= x:
            return ""
        sublist = self.content[x]
        if len(sublist) <= y:
            return ""
        val = sublist[y]
        if isinstance(val, float):
            return locale.str(val)
        return str(val)

    def get_header(self, save_orientation: bool) -> List[str]:
        if save_orientation:
            return [str(i) for i in range(self.max_rows)]

        return self.header

    def get_rows(self, save_orientation: bool) -> List[str]:
        return self.get_header(not save_orientation)


class MeasurementWidget(QWidget):
    """
    :type settings: Settings
    :type segment: Segment
    """

    def __init__(self, settings: PartSettings, segment=None):
        super().__init__()
        self.settings = settings
        self.segment = segment
        self.measurements_storage = MeasurementsStorage()
        self.recalculate_button = QPushButton("Recalculate and\n replace measurement", self)
        self.recalculate_button.clicked.connect(self.replace_measurement_result)
        self.recalculate_append_button = QPushButton("Recalculate and\n append measurement", self)
        self.recalculate_append_button.clicked.connect(self.append_measurement_result)
        self.copy_button = QPushButton("Copy to clipboard", self)
        self.copy_button.setToolTip("You cacn copy also with 'Ctrl+C'. To get raw copy copy with 'Ctrl+Shit+C'")
        self.horizontal_measurement_present = QCheckBox("Horizontal view", self)
        self.no_header = QCheckBox("No header", self)
        self.no_units = QCheckBox("No units", self)
        self.no_units.setChecked(True)
        self.no_units.clicked.connect(self.refresh_view)
        self.expand_mode = QCheckBox("Expand", self)
        self.expand_mode.setToolTip("Shows results for each component in separate entry")
        self.file_names = EnumComboBox(FileNamesEnum)
        self.file_names_label = QLabel("Add file name:")
        self.file_names.currentIndexChanged.connect(self.refresh_view)
        self.horizontal_measurement_present.stateChanged.connect(self.refresh_view)
        self.expand_mode.stateChanged.connect(self.refresh_view)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.measurement_type = SearchCombBox(self)
        # noinspection PyUnresolvedReferences
        self.measurement_type.currentIndexChanged.connect(self.measurement_profile_selection_changed)
        self.measurement_type.addItem("<none>")
        self.measurement_type.addItems(list(sorted(self.settings.measurement_profiles.keys())))
        self.measurement_type.setToolTip(
            'You can create new measurement profile in advanced window, in tab "Measurement settings"'
        )
        self.channels_chose = ChannelComboBox()
        self.units_choose = EnumComboBox(Units)
        self.units_choose.set_value(self.settings.get("units_value", Units.nm))
        self.info_field = QTableWidget(self)
        self.info_field.setColumnCount(3)
        self.info_field.setHorizontalHeaderLabels(["Name", "Value", "Units"])
        self.measurement_add_shift = 0
        layout = QVBoxLayout()
        # layout.addWidget(self.recalculate_button)
        v_butt_layout = QVBoxLayout()
        v_butt_layout.setSpacing(1)
        self.up_butt_layout = QHBoxLayout()
        self.up_butt_layout.addWidget(self.recalculate_button)
        self.up_butt_layout.addWidget(self.recalculate_append_button)
        self.butt_layout = QHBoxLayout()
        # self.butt_layout.setMargin(0)
        # self.butt_layout.setSpacing(10)
        self.butt_layout.addWidget(self.horizontal_measurement_present, 1)
        self.butt_layout.addWidget(self.no_header, 1)
        self.butt_layout.addWidget(self.no_units, 1)
        self.butt_layout.addWidget(self.expand_mode, 1)
        self.butt_layout.addWidget(self.file_names_label)
        self.butt_layout.addWidget(self.file_names, 1)
        self.butt_layout.addWidget(self.copy_button, 2)
        self.butt_layout2 = QHBoxLayout()
        self.butt_layout3 = QHBoxLayout()
        self.butt_layout3.addWidget(QLabel("Channel:"))
        self.butt_layout3.addWidget(self.channels_chose)
        self.butt_layout3.addWidget(QLabel("Units:"))
        self.butt_layout3.addWidget(self.units_choose)
        # self.butt_layout3.addWidget(QLabel("Noise removal:"))
        # self.butt_layout3.addWidget(self.noise_removal_method)
        self.butt_layout3.addWidget(QLabel("Measurement set:"))
        self.butt_layout3.addWidget(self.measurement_type, 2)
        v_butt_layout.addLayout(self.up_butt_layout)
        v_butt_layout.addLayout(self.butt_layout)
        v_butt_layout.addLayout(self.butt_layout2)
        v_butt_layout.addLayout(self.butt_layout3)
        layout.addLayout(v_butt_layout)
        # layout.addLayout(self.butt_layout)
        layout.addWidget(self.info_field)
        self.setLayout(layout)
        # noinspection PyArgumentList
        self.clip = QApplication.clipboard()
        self.settings.image_changed[int].connect(self.image_changed)
        self.previous_profile = None

    def check_if_measurement_can_be_calculated(self, name):
        if name == "<none>":
            return "<none>"
        profile: MeasurementProfile = self.settings.measurement_profiles.get(name)
        if profile.is_any_mask_measurement() and self.settings.mask is None:
            QMessageBox.information(
                self, "Need mask", "To use this measurement set please use data with mask loaded", QMessageBox.Ok
            )
            self.measurement_type.setCurrentIndex(0)
            return "<none>"
        if self.settings.roi is None:
            QMessageBox.information(
                self,
                "Need segmentation",
                'Before calculating please create segmentation ("Execute" button)',
                QMessageBox.Ok,
            )
            self.measurement_type.setCurrentIndex(0)
            return "<none>"
        return name

    def image_changed(self, channels_num):
        self.channels_chose.change_channels_num(channels_num)

    def measurement_profile_selection_changed(self, index):
        text = self.measurement_type.itemText(index)
        text = self.check_if_measurement_can_be_calculated(text)
        try:
            stat = self.settings.measurement_profiles[text]
            is_mask = stat.is_any_mask_measurement()
            disable = is_mask and (self.settings.mask is None)
        except KeyError:
            disable = True
        self.recalculate_button.setDisabled(disable)
        self.recalculate_append_button.setDisabled(disable)
        if disable:
            self.recalculate_button.setToolTip("Measurement profile contains mask measurements when mask is not loaded")
            self.recalculate_append_button.setToolTip(
                "Measurement profile contains mask measurements when mask is not loaded"
            )
        else:
            self.recalculate_button.setToolTip("")
            self.recalculate_append_button.setToolTip("")

    def copy_to_clipboard(self):
        s = ""
        for r in range(self.info_field.rowCount()):
            for c in range(self.info_field.columnCount()):
                try:
                    s += str(self.info_field.item(r, c).text()) + "\t"
                except AttributeError:
                    s += "\t"
            s = s[:-1] + "\n"  # eliminate last '\t'
        self.clip.setText(s)

    def replace_measurement_result(self):
        self.measurements_storage.clear()
        self.previous_profile = ""
        self.append_measurement_result()

    def refresh_view(self):
        self.measurements_storage.set_expand(self.expand_mode.isChecked())
        self.measurements_storage.set_show_units(not self.no_units.isChecked())
        self.info_field.clear()
        save_orientation = self.horizontal_measurement_present.isChecked()
        columns, rows = self.measurements_storage.get_size(save_orientation)
        self.info_field.setColumnCount(columns)
        self.info_field.setRowCount(rows)
        self.info_field.setHorizontalHeaderLabels(self.measurements_storage.get_header(save_orientation))
        self.info_field.setVerticalHeaderLabels(self.measurements_storage.get_rows(save_orientation))
        for x in range(rows):
            for y in range(columns):
                self.info_field.setItem(
                    x, y, QTableWidgetItem(self.measurements_storage.get_val_as_str(x, y, save_orientation))
                )
        if self.file_names.get_value() == FileNamesEnum.No:
            if save_orientation:
                self.info_field.removeColumn(0)
            else:
                self.info_field.removeRow(0)
        elif self.file_names.get_value() == FileNamesEnum.Short:
            if save_orientation:
                columns = 1
            else:
                rows = 1
            for x in range(rows):
                for y in range(columns):
                    item = self.info_field.item(x, y)
                    item.setText(os.path.basename(item.text()))

        self.info_field.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def append_measurement_result(self):
        try:
            compute_class = self.settings.measurement_profiles[self.measurement_type.currentText()]
        except KeyError:
            QMessageBox.warning(
                self,
                "Measurement profile not found",
                f"Measurement profile '{self.measurement_type.currentText()}' not found'",
            )
            return

        if self.settings.roi is None:
            return
        units = self.units_choose.get_value()

        # FIXME find which errors should be displayed as warning
        # def exception_hook(exception):
        #    QMessageBox.warning(self, "Calculation error", f"Error during calculation: {exception}")

        for num in compute_class.get_channels_num():
            if num >= self.settings.image.channels:
                QMessageBox.warning(
                    self,
                    "Measurement error",
                    "Cannot calculate this measurement because " f"image do not have channel {num+1}",
                )
                return

        thread = ExecuteFunctionThread(
            compute_class.calculate,
            [self.settings.image, self.channels_chose.currentIndex(), self.settings.roi_info, units],
        )
        dial = WaitingDialog(thread, "Measurement calculation")  # , exception_hook=exception_hook)
        dial.exec()
        stat: MeasurementResult = thread.result
        if stat is None:
            return
        stat.set_filename(self.settings.image_path)
        self.measurements_storage.add_measurements(stat)
        self.previous_profile = compute_class.name
        self.refresh_view()

    def keyPressEvent(self, e: QKeyEvent):
        if not e.modifiers() & Qt.ControlModifier:
            return
        selected = self.info_field.selectedRanges()

        if e.key() == Qt.Key_C:  # copy
            s = ""

            for r in range(selected[0].topRow(), selected[0].bottomRow() + 1):
                for c in range(selected[0].leftColumn(), selected[0].rightColumn() + 1):
                    try:
                        s += str(self.info_field.item(r, c).text()) + "\t"
                    except AttributeError:
                        s += "\t"
                s = s[:-1] + "\n"  # eliminate last '\t'
            self.clip.setText(s)

    def update_measurement_list(self):
        self.measurement_type.blockSignals(True)
        available = list(sorted(self.settings.measurement_profiles.keys()))
        text = self.measurement_type.currentText()
        try:
            index = available.index(text) + 1
        except ValueError:
            index = 0
        self.measurement_type.clear()
        self.measurement_type.addItem("<none>")
        self.measurement_type.addItems(available)
        self.measurement_type.setCurrentIndex(index)
        self.measurement_type.blockSignals(False)

    def showEvent(self, _):
        self.update_measurement_list()

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            self.update_measurement_list()
        return super().event(event)

    @staticmethod
    def _move_widgets(widgets_list: List[Tuple[QWidget, int]], layout1: QBoxLayout, layout2: QBoxLayout):
        for el in widgets_list:
            layout1.removeWidget(el[0])
            layout2.addWidget(el[0], el[1])

    def resizeEvent(self, _event: QResizeEvent) -> None:
        if self.width() < 800 and self.butt_layout2.count() == 0:
            self._move_widgets(
                [(self.file_names_label, 1), (self.file_names, 1), (self.copy_button, 2)],
                self.butt_layout,
                self.butt_layout2,
            )
        elif self.width() > 800 and self.butt_layout2.count() != 0:
            self._move_widgets(
                [(self.file_names_label, 1), (self.file_names, 1), (self.copy_button, 2)],
                self.butt_layout2,
                self.butt_layout,
            )
