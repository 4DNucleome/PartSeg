import logging
import sys

import numpy as np
from qtpy.QtCore import Qt, QEvent
from qtpy.QtWidgets import QWidget, QPushButton, QCheckBox, QComboBox, QTableWidget, QVBoxLayout, QHBoxLayout,\
    QLabel, QApplication, QTableWidgetItem, QMessageBox

from PartSeg.utils.analysis.statistics_calculation import StatisticProfile
from ..common_gui.universal_gui_part import ChannelComboBox, EnumComboBox
from ..common_gui.waiting_dialog import WaitingDialog
from .partseg_settings import PartSettings
from ..utils.universal_const import Units
from ..project_utils_qt.execute_function_thread import ExecuteFunctionThread


class StatisticsWidget(QWidget):
    """
    :type settings: Settings
    :type segment: Segment
    """

    def __init__(self, settings: PartSettings, segment=None):
        super(StatisticsWidget, self).__init__()
        self.settings = settings
        self.segment = segment
        self.recalculate_button = QPushButton("Recalculate and\n replace statistics", self)
        self.recalculate_button.clicked.connect(self.replace_statistics)
        self.recalculate_append_button = QPushButton("Recalculate and\n append statistics", self)
        self.recalculate_append_button.clicked.connect(self.append_statistics)
        self.copy_button = QPushButton("Copy to clipboard", self)
        self.horizontal_statistics = QCheckBox("Horizontal view", self)
        self.no_header = QCheckBox("No header", self)
        self.no_units = QCheckBox("No units", self)
        self.no_units.setChecked(True)
        self.horizontal_statistics.stateChanged.connect(self.horizontal_changed)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.statistic_type = QComboBox(self)
        # noinspection PyUnresolvedReferences
        self.statistic_type.currentTextChanged.connect(self.statistic_selection_changed)
        self.statistic_type.addItem("<none>")
        self.statistic_type.addItems(list(sorted(self.settings.statistic_profiles.keys())))
        self.statistic_type.setToolTip(
            "You can create new statistic profile in advanced window, in tab \"Statistic settings\"")
        self.channels_chose = ChannelComboBox()
        self.units_choose = EnumComboBox(Units)
        self.units_choose.set_value(self.settings.get("units_value", Units.nm))
        self.info_field = QTableWidget(self)
        self.info_field.setColumnCount(3)
        self.info_field.setHorizontalHeaderLabels(["Name", "Value", "Units"])
        self.statistic_shift = 0
        layout = QVBoxLayout()
        # layout.addWidget(self.recalculate_button)
        v_butt_layout = QVBoxLayout()
        v_butt_layout.setSpacing(1)
        self.up_butt_layout = QHBoxLayout()
        self.up_butt_layout.addWidget(self.recalculate_button)
        self.up_butt_layout.addWidget(self.recalculate_append_button)
        butt_layout = QHBoxLayout()
        # butt_layout.setMargin(0)
        butt_layout.setSpacing(10)
        butt_layout.addWidget(self.horizontal_statistics, 1)
        butt_layout.addWidget(self.no_header, 1)
        butt_layout.addWidget(self.no_units, 1)
        butt_layout.addWidget(self.copy_button, 2)
        butt_layout2 = QHBoxLayout()
        butt_layout2.addWidget(QLabel("Channel:"))
        butt_layout2.addWidget(self.channels_chose)
        butt_layout2.addWidget(QLabel("Units:"))
        butt_layout2.addWidget(self.units_choose)
        # butt_layout2.addWidget(QLabel("Noise removal:"))
        # butt_layout2.addWidget(self.noise_removal_method)
        butt_layout2.addWidget(QLabel("Profile:"))
        butt_layout2.addWidget(self.statistic_type, 2)
        v_butt_layout.addLayout(self.up_butt_layout)
        v_butt_layout.addLayout(butt_layout)
        v_butt_layout.addLayout(butt_layout2)
        layout.addLayout(v_butt_layout)
        # layout.addLayout(butt_layout)
        layout.addWidget(self.info_field)
        self.setLayout(layout)
        # noinspection PyArgumentList
        self.clip = QApplication.clipboard()
        self.settings.image_changed[int].connect(self.image_changed)
        self.previous_profile = None
        # self.update_statistics()

    def check_statistics(self, name):
        if name == "<none>":
            return "<none>"
        profile: StatisticProfile = self.settings.statistic_profiles.get(name)
        if profile.is_any_mask_statistic() and self.settings.mask is None:
            QMessageBox.information(self, "Need mask",
                                    "To use this measurement set please use data with mask loaded", QMessageBox.Ok)
            self.statistic_type.setCurrentIndex(0)
            return "<none>"
        if self.settings.segmentation is None:
            QMessageBox.information(self, "Need segmentation",
                                    "Before calculating please create segmentation (\"Execute\" button)",
                                    QMessageBox.Ok)
            self.statistic_type.setCurrentIndex(0)
            return "<none>"
        return name

    def image_changed(self, channels_num):
        self.channels_chose.change_channels_num(channels_num)

    def statistic_selection_changed(self, text):
        text = self.check_statistics(text)
        try:
            stat = self.settings.statistic_profiles[text]
            is_mask = stat.is_any_mask_statistic()
            disable = is_mask and (self.settings.mask is None)
        except KeyError:
            disable = True
        self.recalculate_button.setDisabled(disable)
        self.recalculate_append_button.setDisabled(disable)
        if disable:
            self.recalculate_button.setToolTip("Statistics contains mask statistic when mask is not loaded")
            self.recalculate_append_button.setToolTip("Statistics contains mask statistic when mask is not loaded")
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
                    logging.info("Copy problem")
            s = s[:-1] + "\n"  # eliminate last '\t'
        self.clip.setText(s)

    def replace_statistics(self):
        self.statistic_shift = 0
        self.info_field.setRowCount(0)
        self.info_field.setColumnCount(0)
        self.previous_profile = None
        self.append_statistics()

    def horizontal_changed(self):
        rows = self.info_field.rowCount()
        columns = self.info_field.columnCount()
        ob_array = np.zeros((rows, columns), dtype=object)
        for x in range(rows):
            for y in range(columns):
                field = self.info_field.item(x, y)
                if field is not None:
                    ob_array[x, y] = field.text()

        hor_headers = [self.info_field.horizontalHeaderItem(x).text() for x in range(columns)]
        ver_headers = [self.info_field.verticalHeaderItem(x).text() for x in range(rows)]
        self.info_field.setColumnCount(rows)
        self.info_field.setRowCount(columns)
        self.info_field.setHorizontalHeaderLabels(ver_headers)
        self.info_field.setVerticalHeaderLabels(hor_headers)
        for x in range(rows):
            for y in range(columns):
                self.info_field.setItem(y, x, QTableWidgetItem(ob_array[x, y]))

    def append_statistics(self):
        try:
            compute_class = self.settings.statistic_profiles[self.statistic_type.currentText()]
        except KeyError:
            QMessageBox.warning(self, "Statistic profile not found",
                                f"Statistic profile '{self.statistic_type.currentText()}' not found'")
            return
        channel = self.settings.image.get_channel(self.channels_chose.currentIndex())
        segmentation = self.settings.segmentation
        if segmentation is None:
            return
        full_mask = self.settings.full_segmentation
        base_mask = self.settings.mask
        units = self.units_choose.get_value()
        units_name = str(units)

        def exception_hook(exception):
            QMessageBox.warning(self, "Calculation error", f"Error during calculation: {exception}")

        kwargs = {}
        for num in compute_class.get_channels_num():
            if num >= self.settings.image.channels:
                QMessageBox.warning(self, "Measurement error", "Cannot calculate this statistics because "
                                                               f"image do not have channel {num+1}")
                return
            kwargs[f"channel+{num}"] = self.settings.image.get_channel(num)


        thread = ExecuteFunctionThread(compute_class.calculate, [channel, segmentation, full_mask, base_mask,
                                                                 self.settings.image.spacing, units], kwargs)
        dial = WaitingDialog(thread, "Statistic calculation", exception_hook=exception_hook)
        dial.exec()
        stat = thread.result
        if stat is None:
            return
        if self.no_header.isChecked() or self.previous_profile == compute_class.name:
            self.statistic_shift -= 1
        if self.no_units.isChecked():
            header_grow = self.statistic_shift - 1
        else:
            header_grow = self.statistic_shift
        if self.horizontal_statistics.isChecked():
            ver_headers = [self.info_field.verticalHeaderItem(x).text() for x in range(self.info_field.rowCount())]
            self.info_field.setRowCount(3 + header_grow)
            self.info_field.setColumnCount(max(len(stat), self.info_field.columnCount()))
            if not self.no_header.isChecked() and (self.previous_profile != compute_class.name):
                ver_headers.append("Name")
            ver_headers.extend(["Value"])
            if not self.no_units.isChecked():
                ver_headers.append("Units")
            self.info_field.setVerticalHeaderLabels(ver_headers)
            self.info_field.setHorizontalHeaderLabels([str(x) for x in range(len(stat))])
            for i, (key, (val, unit)) in enumerate(stat.items()):
                if not self.no_header.isChecked() and (self.previous_profile != compute_class.name):
                    self.info_field.setItem(self.statistic_shift + 0, i, QTableWidgetItem(key))
                self.info_field.setItem(self.statistic_shift + 1, i, QTableWidgetItem(str(val)))
                if not self.no_units.isChecked():
                    try:
                        self.info_field.setItem(self.statistic_shift + 2, i,
                                                QTableWidgetItem(str(unit)))
                    except KeyError as k:
                        print(k, sys.stderr)
        else:
            hor_headers = [self.info_field.horizontalHeaderItem(x).text() for x in range(self.info_field.columnCount())]
            self.info_field.setRowCount(max(len(stat), self.info_field.rowCount()))
            self.info_field.setColumnCount(3 + header_grow)
            self.info_field.setVerticalHeaderLabels([str(x) for x in range(len(stat))])
            if not self.no_header.isChecked() and (self.previous_profile != compute_class.name):
                hor_headers.append("Name")
            hor_headers.extend(["Value"])
            if not self.no_units.isChecked():
                hor_headers.append("Units")
            self.info_field.setHorizontalHeaderLabels(hor_headers)
            for i, (key, (val, unit)) in enumerate(stat.items()):
                # print(i, key, val)
                if not self.no_header.isChecked() and (self.previous_profile != compute_class.name):
                    self.info_field.setItem(i, self.statistic_shift + 0, QTableWidgetItem(key))
                self.info_field.setItem(i, self.statistic_shift + 1, QTableWidgetItem(str(val)))
                if not self.no_units.isChecked():
                    try:
                        self.info_field.setItem(i, self.statistic_shift + 2,
                                                QTableWidgetItem(str(unit)))
                    except KeyError as k:
                        print(k, file=sys.stderr)
        if self.no_units.isChecked():
            self.statistic_shift -= 1
        self.statistic_shift += 3
        self.previous_profile = compute_class.name
        self.info_field.resizeColumnsToContents()

    def keyPressEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            selected = self.info_field.selectedRanges()

            if e.key() == Qt.Key_C:  # copy
                s = ""

                for r in range(selected[0].topRow(), selected[0].bottomRow() + 1):
                    for c in range(selected[0].leftColumn(), selected[0].rightColumn() + 1):
                        try:
                            s += str(self.info_field.item(r, c).text()) + "\t"
                        except AttributeError:
                            s += "\t"
                            logging.info("Copy problem")
                    s = s[:-1] + "\n"  # eliminate last '\t'
                self.clip.setText(s)

    def update_statistic_list(self):
        self.statistic_type.blockSignals(True)
        avali = list(sorted(self.settings.statistic_profiles.keys()))
        # avali.insert(0, "Emish statistics (oryginal)")
        text = self.statistic_type.currentText()
        try:
            index = avali.index(text) + 1
        except ValueError:
            index = 0
        self.statistic_type.clear()
        self.statistic_type.addItem("<none>")
        self.statistic_type.addItems(avali)
        self.statistic_type.setCurrentIndex(index)
        self.statistic_type.blockSignals(False)

    def showEvent(self, _):
        self.update_statistic_list()

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            self.update_statistic_list()
        return super().event(event)
