# coding=utf-8
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer, QByteArray
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QPushButton, QListWidget, QVBoxLayout, QHBoxLayout, QLineEdit, \
    QFileDialog, QWidget, QDialog, QLabel, QMessageBox, QProgressBar, QSpinBox, QGridLayout, QComboBox, \
    QTabWidget, QTreeWidget, QTreeWidgetItem

from PartSeg.project_utils_qt.list_item_widget_for_exception import ExceptionList, ExceptionListItem
from ..common_gui.select_multiple_files import AddFiles
from .partseg_settings import PartSettings
from PartSeg.utils.analysis.batch_processing.batch_backend import CalculationManager
from PartSeg.utils.analysis.calculation_plan import CalculationPlan, MaskFile, MaskMapper, Calculation
from .prepare_plan_widget import CalculatePlaner
from ..utils.global_settings import static_file_folder
from ..utils.universal_const import Units
from ..common_gui.universal_gui_part import Spacing, right_label

__author__ = "Grzegorz Bokota"


class ProgressView(QWidget):
    """
    :type batch_manager: CalculationManager
    """
    def __init__(self, parent, batch_manager):
        QWidget.__init__(self, parent)
        self.batch_manager = batch_manager
        self.whole_progress = QProgressBar(self)
        self.whole_progress.setMinimum(0)
        self.whole_progress.setMaximum(1)
        self.whole_progress.setFormat("%v of %m")
        self.whole_progress.setTextVisible(True)
        self.part_progress = QProgressBar(self)
        self.part_progress.setMinimum(0)
        self.part_progress.setMaximum(1)
        self.part_progress.setFormat("%v of %m")
        self.whole_label = QLabel("Whole progress:", self)
        self.part_label = QLabel("Part progress:", self)
        self.logs = ExceptionList(self)
        self.logs.setToolTip("Logs")
        self.task_que = QListWidget()
        self.process_num_timer = QTimer()
        self.process_num_timer.setInterval(1000)
        self.process_num_timer.setSingleShot(True)
        self.process_num_timer.timeout.connect(self.change_number_of_workers)
        self.number_of_process = QSpinBox(self)
        self.number_of_process.setRange(1, multiprocessing.cpu_count())
        self.number_of_process.setValue(1)
        self.number_of_process.setToolTip("Number of process used in batch calculation")
        self.number_of_process.valueChanged.connect(self.process_num_timer_start)
        layout = QGridLayout()
        layout.addWidget(self.whole_label, 0, 0, Qt.AlignRight)
        layout.addWidget(self.whole_progress, 0, 1, 1, 2)
        layout.addWidget(self.part_label, 1, 0, Qt.AlignRight)
        layout.addWidget(self.part_progress, 1, 1, 1, 2)
        lab = QLabel("Number of process:")
        lab.setToolTip("Number of process used in batch calculation")
        layout.addWidget(lab, 2, 0)
        layout.addWidget(self.number_of_process, 2, 1)
        layout.addWidget(self.logs, 3, 0, 1, 3)
        layout.addWidget(self.task_que, 0, 4, 0, 1)
        layout.setColumnMinimumWidth(2, 10)
        layout.setColumnStretch(2, 1)
        self.setLayout(layout)
        self.preview_timer = QTimer()
        self.preview_timer.setInterval(1000)
        self.preview_timer.timeout.connect(self.update_info)

    def new_task(self):
        self.whole_progress.setMaximum(self.batch_manager.calculation_size)
        if not self.preview_timer.isActive():
            self.update_info()
            self.preview_timer.start()

    def update_info(self):
        errors, total, parts = self.batch_manager.get_results()
        for el in errors:
            ExceptionListItem(el, self.logs)
        self.whole_progress.setValue(total)
        working_search = True
        for i, (progress, total) in enumerate(parts):
            if working_search and progress != total:
                self.part_progress.setMaximum(total)
                self.part_progress.setValue(progress)
                working_search = False
            if i < self.task_que.count():
                item = self.task_que.item(i)
                item.setText("Task {} ({}/{})".format(i, progress, total))
            else:
                self.task_que.addItem("Task {} ({}/{})".format(i, progress, total))
        if not self.batch_manager.has_work:
            self.part_progress.setValue(self.part_progress.maximum())
            self.preview_timer.stop()
            logging.info("Progress stop")

    def process_num_timer_start(self):
        self.process_num_timer.start()

    def update_progress(self, total_progress, part_progress):
        self.whole_progress.setValue(total_progress)
        self.part_progress.setValue(part_progress)

    def set_total_size(self, size):
        self.whole_progress.setMaximum(size)

    def set_part_size(self, size):
        self.part_progress.setMaximum(size)

    def change_number_of_workers(self):
        self.batch_manager.set_number_of_workers(self.number_of_process.value())


class FileChoose(QWidget):
    """
    :type batch_manager: CalculationManager
    """
    def __init__(self, settings: PartSettings, batch_manager, parent=None):
        QWidget.__init__(self, parent)
        self.files_to_proceed = set()
        self.settings = settings
        self.batch_manager = batch_manager
        self.files_widget = AddFiles(settings, self)
        self.progress = ProgressView(self, batch_manager)
        self.run_button = QPushButton("Run calculation")
        self.run_button.setDisabled(True)
        self.calculation_choose = QComboBox()
        self.calculation_choose.addItem("<no calculation>")
        self.calculation_choose.currentIndexChanged[str].connect(self.change_situation)
        self.result_file = QLineEdit(self)
        self.result_file.setAlignment(Qt.AlignRight)
        self.result_file.setReadOnly(True)
        self.chose_result = QPushButton("Choose save place", self)
        self.chose_result.clicked.connect(self.chose_result_file)

        self.run_button.clicked.connect(self.prepare_calculation)
        self.files_widget.file_list_changed.connect(self.change_situation)

        layout = QVBoxLayout()
        layout.addWidget(self.files_widget)
        calc_layout = QHBoxLayout()
        calc_layout.addWidget(QLabel("Calculation profile:"))
        calc_layout.addWidget(self.calculation_choose)
        calc_layout.addWidget(self.result_file)
        calc_layout.addWidget(self.chose_result)
        calc_layout.addStretch()
        calc_layout.addWidget(self.run_button)
        layout.addLayout(calc_layout)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def prepare_calculation(self):
        plan = self.settings.batch_plans[str(self.calculation_choose.currentText())]
        dial = CalculationPrepare(self.files_widget.get_paths(), plan, str(self.result_file.text()), self.settings,
                                  self.batch_manager)
        if dial.exec_():
            self.batch_manager.add_calculation(dial.get_data())
            self.progress.new_task()

    def showEvent(self, _):
        current_calc = str(self.calculation_choose.currentText())
        new_list = ["<no calculation>"] + list(sorted(self.settings.batch_plans.keys()))
        try:
            index = new_list.index(current_calc)
        except ValueError:
            index = 0
        self.calculation_choose.clear()
        self.calculation_choose.addItems(new_list)
        self.calculation_choose.setCurrentIndex(index)

    def change_situation(self):
        if str(self.calculation_choose.currentText()) != "<no calculation>" and \
                        len(self.files_widget.files_to_proceed) != 0 and str(self.result_file.text()) != "":
            self.run_button.setEnabled(True)
        else:
            self.run_button.setDisabled(True)

    def chose_result_file(self):
        dial = QFileDialog(self, "Select result file")
        dial.setDirectory(
            self.settings.get("io.save_directory", self.settings.get("io.open_directory", str(Path.home()))))
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setNameFilter("Excel file (*.xlsx)")
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            if os.path.splitext(file_path)[1] == '':
                file_path += ".xlsx"
            self.result_file.setText(file_path)
            self.change_situation()


class BatchWindow(QTabWidget):
    """
    :type settings: PartSettings
    """
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch processing")
        self.settings = settings
        self.batch_manager = CalculationManager()
        self.file_choose = FileChoose(self.settings, self.batch_manager, self)
        self.calculate_planer = CalculatePlaner(self.settings, self)
        self.addTab(self.file_choose, "Choose files")
        self.addTab(self.calculate_planer, "Prepare plan")
        self.working = False
        try:
            geometry = self.settings.get_from_profile("batch_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, 'ascii')))
        except KeyError:
            pass

    def focusInEvent(self, event):
        self.calculate_planer.showEvent(event)

    def is_working(self):
        return self.working

    def terminate(self):
        self.working = False

    def closeEvent(self, event):
        if self.is_working():
            ret = QMessageBox.warning(self, "Batch work", "Batch work is not finished. "
                                                          "Would you like to terminate it?",
                                      QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                self.terminate()
            else:
                event.ignore()
        self.settings.set_in_profile("batch_window_geometry", bytes(self.saveGeometry().toHex()).decode('ascii'))


class CalculationPrepare(QDialog):
    """
    :type mask_path_list: list[QLineEdit]
    :type mask_mapper_list: list[MaskMapper]
    """
    def __init__(self, file_list, calculation_plan, statistic_file_path, settings, batch_manager):
        """

        :param file_list: list of files to proceed
        :type file_list: list[str]
        :param calculation_plan: calculation plan for this run
        :type calculation_plan: CalculationPlan
        :param statistic_file_path: path to statistic file
        :type statistic_file_path: str
        :param settings: settings object
        :type settings: PartSettings
        :type batch_manager: CalculationManager
        """
        super().__init__()
        self.setWindowTitle("Calculation start")
        self.file_list = file_list
        self.calculation_plan = calculation_plan
        self.statistic_file_path = statistic_file_path
        self.settings = settings
        self.batch_manager = batch_manager
        self.info_label = QLabel("Information, <i><font color='blue'>warnings</font></i>, "
                                 "<b><font color='red'>errors</font><b>")
        self.voxel_size = Spacing("Voxel size", settings.image.spacing, settings.get("units_value", Units.nm))
        all_prefix = os.path.commonprefix(file_list)
        if not os.path.exists(all_prefix):
            all_prefix = os.path.dirname(all_prefix)
        if not os.path.isdir(all_prefix):
            all_prefix = os.path.dirname(all_prefix)
        self.base_prefix = QLineEdit(all_prefix, self)
        self.base_prefix.setReadOnly(True)
        self.result_prefix = QLineEdit(all_prefix, self)
        self.result_prefix.setReadOnly(True)
        self.base_prefix_btn = QPushButton("Choose data prefix")
        self.base_prefix_btn.clicked.connect(self.choose_data_prefix)
        self.result_prefix_btn = QPushButton("Choose save prefix")
        self.result_prefix_btn.clicked.connect(self.choose_result_prefix)
        self.sheet_name = QLineEdit("Sheet1")
        self.sheet_name.textChanged.connect(self.verify_data)
        self.statistic_file_path_view = QLineEdit(statistic_file_path)
        self.statistic_file_path_view.setReadOnly(True)

        self.mask_path_list = []
        self.mask_mapper_list = self.calculation_plan.get_list_file_mask()
        mask_file_list = []
        for i, el in enumerate(self.mask_mapper_list):
            if isinstance(el, MaskFile):
                mask_file_list.append((i, el))
        mask_path_layout = QGridLayout()
        for i, (pos, mask_file) in enumerate(mask_file_list):
            if mask_file.name == "":
                mask_path_layout.addWidget(right_label("Path to file {} with mask mapping".format(i+1)))
            else:
                mask_path_layout.addWidget(right_label("Path to file {} with mask mapping for name: {}".format(i + 1, mask_file.name)))
            mask_path = QLineEdit(self)
            mask_path.setReadOnly(True)
            self.mask_path_list.append(mask_path)
            set_path = QPushButton("Choose file", self)
            set_path.clicked.connect(self.set_mapping_mask(i, pos))
            mask_path_layout.addWidget(mask_path, i, 1)
            mask_path_layout.addWidget(set_path, i, 2)

        self.state_list = np.zeros((len(self.file_list), len(self.mask_mapper_list)), dtype=np.uint8)

        self.file_list_widget = QTreeWidget()
        self.file_list_widget.header().close()

        self.execute_btn = QPushButton("Execute")
        self.execute_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)

        layout = QGridLayout()
        layout.addWidget(self.info_label, 0, 0, 1, 5)
        layout.addWidget(self.voxel_size, 1, 0, 1, 5)
        layout.addWidget(right_label("Statistics sheet name:"), 3, 3)
        layout.addWidget(self.sheet_name, 3, 4)
        layout.addWidget(right_label("Statistics file path:"), 2, 3)
        layout.addWidget(self.statistic_file_path_view, 2, 4)

        layout.addWidget(right_label("Data prefix:"), 2, 0)
        layout.addWidget(self.base_prefix, 2, 1)
        layout.addWidget(self.base_prefix_btn, 2, 2)
        layout.addWidget(right_label("Save prefix:"), 3, 0)
        layout.addWidget(self.result_prefix, 3, 1)
        layout.addWidget(self.result_prefix_btn, 3, 2)
        layout.addLayout(mask_path_layout, 4, 0, 1, 0)

        layout.addWidget(self.file_list_widget, 5, 0, 3, 6)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.execute_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout, 8, 0, 1, 0)
        self.setLayout(layout)
        self.verify_data()

    def choose_data_prefix(self):
        dial = QFileDialog()
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setFileMode(QFileDialog.Directory)
        dial.setDirectory(self.base_prefix.text())
        dial.setHistory(dial.history() + self.settings.get_path_history())
        if dial.exec_():
            dir_path = str(dial.selectedFiles()[0])
            self.base_prefix.setText(dir_path)

    def choose_result_prefix(self):
        dial = QFileDialog()
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setFileMode(QFileDialog.Directory)
        dial.setDirectory(self.result_prefix.text())
        dial.setHistory(dial.history() + self.settings.get_path_history())
        if dial.exec_():
            dir_path = str(dial.selectedFiles()[0])
            self.result_prefix.setText(dir_path)

    def set_mapping_mask(self, i, pos):
        def mapping_dialog():
            dial = QFileDialog(self, "Select file")
            dial.setHistory(dial.history() + self.settings.get_path_history())
            base_path = str(self.base_prefix.text()).strip()
            if base_path != "":
                dial.setDirectory(base_path)
            dial.setFileMode(QFileDialog.ExistingFile)
            if dial.exec_():
                path = str(dial.selectedFiles())
                self.mask_path_list[i].setText(path)
                file_mapper = self.mask_mapper_list[pos]
                """:type : MaskFile"""
                file_mapper.set_map_path(path)
        return mapping_dialog

    def get_data(self):
        res = {"file_list": self.file_list, "base_prefix": str(self.base_prefix.text()),
               "result_prefix": str(self.result_prefix.text()),
               "statistic_file_path": str(self.statistic_file_path_view.text()),
               "sheet_name": str(self.sheet_name.text()), "calculation_plan": self.calculation_plan,
               "voxel_size": self.voxel_size.get_values()}
        return Calculation(**res)

    def verify_data(self):
        self.execute_btn.setEnabled(True)
        text = "information, <i><font color='blue'>warnings</font></i>, <b><font color='red'>errors</font></b><br>" \
               "The voxel size is for file in which metadata do not contains this information<br>"
        if not self.batch_manager.is_valid_sheet_name(str(self.statistic_file_path_view.text()),
                                                      str(self.sheet_name.text())):
            text += "<i><font color='blue'>Sheet name already in use</i></font><br>"
            self.execute_btn.setDisabled(True)
        if self.state_list.size > 0:
            val = np.unique(self.state_list)
            if 1 in val:
                self.execute_btn.setDisabled(True)
                text += "<i><font color='blue'>Some mask map file are not set</font></i><br>"
            if 2 in val:
                self.execute_btn.setDisabled(True)
                text += "<b><font color='red'>Some mask do not exists</font><b><br>"

        text = text[:-4]
        self.info_label.setText(text)

    def showEvent(self, event):
        super().showEvent(event)
        ok_icon = QIcon(os.path.join(static_file_folder, "icons", "task-accepted.png"))
        bad_icon = QIcon(os.path.join(static_file_folder, "icons", "task-reject.png"))
        warn_icon = QIcon(os.path.join(static_file_folder, "icons", "task-attempt.png"))
        all_prefix = os.path.commonprefix(self.file_list)
        if not os.path.exists(all_prefix):
            all_prefix = os.path.dirname(all_prefix)
        for file_num, file_path in enumerate(self.file_list):
            widget = QTreeWidgetItem(self.file_list_widget)
            widget.setText(0, os.path.relpath(file_path, all_prefix))
            if not os.path.exists(file_path):
                widget.setIcon(0, bad_icon)
                widget.setToolTip(0, "File do not exists")
                sub_widget = QTreeWidgetItem(widget)
                sub_widget.setText(0, "File do not exists")
                continue
            for mask_num, mask_mapper in enumerate(self.mask_mapper_list):
                if mask_mapper.is_ready():
                    mask_path = mask_mapper.get_mask_path(file_path)
                    exist = os.path.exists(mask_path)
                    if exist:
                        sub_widget = QTreeWidgetItem(widget)
                        sub_widget.setText(0, "Mask {} ok".format(mask_mapper.name))
                        sub_widget.setIcon(0, ok_icon)
                        self.state_list[file_num, mask_num] = 0
                    else:
                        sub_widget = QTreeWidgetItem(widget)
                        sub_widget.setText(0, "Mask {} do not exists (path: {})".format(
                            mask_mapper.name, os.path.relpath(mask_path, all_prefix)))
                        sub_widget.setIcon(0, bad_icon)
                        self.state_list[file_num, mask_num] = 2
                else:
                    sub_widget = QTreeWidgetItem(widget)
                    sub_widget.setText(0, "Mask {} unknown".format(mask_mapper.name))
                    sub_widget.setIcon(0, warn_icon)
                    self.state_list[file_num, mask_num] = 1
            if self.state_list.shape[1] == 0:
                state = 0
            else:
                state = self.state_list[file_num].max()

            if state == 0:
                widget.setIcon(0, ok_icon)
            elif state == 1:
                widget.setIcon(0, warn_icon)
            else:
                widget.setIcon(0, bad_icon)


