import os
from glob import glob
import multiprocessing
from backend import Settings

from qt_import import *

__author__ = "Grzegorz Bokota"


class AcceptFiles(QDialog):
    def __init__(self, files):
        super(AcceptFiles, self).__init__()
        self.ok = QPushButton("Add", self)
        self.ok.pyqtConfigure(clicked=self.accept)
        discard = QPushButton("Discard", self)
        discard.pyqtConfigure(clicked=self.close)
        self.files = QListWidget(self)
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for file_name in files:
            self.files.addItem(file_name)
        for i in range(self.files.count()):
            self.files.item(i).setSelected(True)
        self.ok.setDefault(True)
        self.ok.setAutoDefault(True)

        layout = QVBoxLayout()
        layout.addWidget(self.files)
        butt_layout = QHBoxLayout()
        butt_layout.addWidget(discard)
        butt_layout.addStretch()
        butt_layout.addWidget(self.ok)
        layout.addLayout(butt_layout)
        self.setLayout(layout)

    def selection_changed(self):
        if self.files.selectedItems().count() == 0:
            self.ok.setDisabled(True)
        else:
            self.ok.setEnabled(True)

    def get_files(self):
        return [str(item.text()) for item in self.files.selectedItems()]


class AddFiles(QWidget):

    """Docstring for AddFiles. """

    def __init__(self, settings, parent):
        """TODO: to be defined1. """
        QWidget.__init__(self, parent)
        self.settings = settings
        self.files_to_proceed = set()
        self.paths = QLineEdit(self)
        self.selected_files = QListWidget(self)
        self.selected_files.itemSelectionChanged.connect(self.file_chosen)
        self.found_button = QPushButton("Find all", self)
        self.found_button.pyqtConfigure(clicked=self.find_all)
        self.select_files_button = QPushButton("Select files")
        self.select_dir_button = QPushButton("Select directory")
        self.select_files_button.pyqtConfigure(clicked=self.select_files)
        self.select_dir_button.pyqtConfigure(clicked=self.select_directory)
        self.delete_button = QPushButton("Remove file from list", self)
        self.delete_button.setDisabled(True)
        self.delete_button.pyqtConfigure(clicked=self.delete_element)
        layout = QVBoxLayout()
        layout.addWidget(self.paths)
        select_layout = QHBoxLayout()
        select_layout.addWidget(self.found_button)
        select_layout.addWidget(self.select_files_button)
        select_layout.addWidget(self.select_dir_button)
        select_layout.addStretch()
        select_layout.addWidget(self.delete_button)
        layout.addLayout(select_layout)
        layout.addWidget(self.selected_files)
        self.setLayout(layout)

    def find_all(self):
        paths = glob(str(self.paths.text()))
        paths = sorted(list(set(paths) - self.files_to_proceed))
        if len(paths) > 0:
            dialog = AcceptFiles(paths)
            if dialog.exec_():
                new_paths = dialog.get_files()
                for path in new_paths:
                    lwi = QListWidgetItem(path)
                    lwi.setTextAlignment(Qt.AlignRight)
                    self.selected_files.addItem(lwi)
                self.files_to_proceed.update(new_paths)
        else:
            QMessageBox.warning(self, "No new files", "No new files found", QMessageBox.Ok)

    def select_files(self):
        dial = QFileDialog(self, "Select files")
        if self.settings.batch_directory is not None:
            dial.setDirectory(self.settings.batch_directory)
        dial.setFileMode(QFileDialog.ExistingFiles)
        if dial.exec_():
            new_paths = sorted(set(map(str, dial.selectedFiles())) - self.files_to_proceed)
            for path in new_paths:
                lwi = QListWidgetItem(path)
                lwi.setTextAlignment(Qt.AlignRight)
                self.selected_files.addItem(lwi)
            self.files_to_proceed.update(new_paths)

    def select_directory(self):
        dial = QFileDialog(self, "Select directory")
        if self.settings.batch_directory is not None:
            dial.setDirectory(self.settings.batch_directory)
        dial.setFileMode(QFileDialog.Directory)
        if dial.exec_():
            self.paths.setText(dial.selectedFiles()[0])

    def file_chosen(self):
        self.delete_button.setEnabled(True)

    def delete_element(self):
        self.selected_files.takeItem(self.selected_files.currentRow())
        if self.selected_files.count() == 0:
            self.delete_button.setDisabled(True)


class ProgressView(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
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
        self.logs = QTextEdit(self)
        #self.logs.setMaximumHeight(50)
        self.logs.setReadOnly(True)
        self.logs.setToolTip("Logs")
        self.task_que = QTextEdit()
        self.task_que.setReadOnly(True)
        self.number_of_process = QSpinBox(self)
        self.number_of_process.setRange(1, multiprocessing.cpu_count())
        self.number_of_process.setValue(1)
        layout = QGridLayout()
        layout.addWidget(self.whole_label, 0, 0, Qt.AlignRight)
        layout.addWidget(self.whole_progress, 0, 1, 1, 2)
        layout.addWidget(self.part_label, 1, 0, Qt.AlignRight)
        layout.addWidget(self.part_progress, 1, 1, 1, 2)
        layout.addWidget(QLabel("Process number:"), 2, 0)
        layout.addWidget(self.number_of_process, 2, 1)
        layout.addWidget(self.logs, 3, 0, 1, 3)
        layout.addWidget(self.task_que, 0, 4, 0, 1)
        layout.setColumnMinimumWidth(2, 10)
        layout.setColumnStretch(2, 1)
        self.setLayout(layout)

    def update_progress(self, total_progress, part_progress):
        self.whole_progress.setValue(total_progress)
        self.part_progress.setValue(part_progress)

    def set_total_size(self, size):
        self.whole_progress.setMaximum(size)

    def set_part_size(self, size):
        self.part_progress.setMaximum(size)


class FileChoose(QWidget):
    def __init__(self, settings, parent=None):
        QWidget.__init__(self, parent)
        self.files_to_proceed = set()
        self.settings = settings
        self.files_widget = AddFiles(settings, self)
        self.progress = ProgressView(self)
        self.run_button = QPushButton("Run calculation")
        self.run_button.setDisabled(True)
        self.calculation_choose = QComboBox()
        self.calculation_choose.addItem("<no calculation>")
        self.calculation_choose.currentIndexChanged[str_type].connect(self.change_plan)
        self.result_file = QLineEdit(self)
        self.result_file.setAlignment(Qt.AlignRight)
        self.chose_result = QPushButton("Choose save place", self)
        self.chose_result.clicked.connect(self.chose_result_file)

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

    def showEvent(self, _):
        current_calc = str(self.calculation_choose.currentText())
        new_list = ["<no calculation>", "aa"] + self.settings.batch_plans
        try:
            index = new_list.index(current_calc)
        except ValueError:
            index = 0
        self.calculation_choose.clear()
        self.calculation_choose.addItems(new_list)
        self.calculation_choose.setCurrentIndex(index)

    def change_plan(self, text):
        if str(text) == "<no calculation>":
            self.run_button.setDisabled(True)
        else:
            self.run_button.setEnabled(True)

    def chose_result_file(self):
        dial = QFileDialog(self, "Select result.file")
        if self.settings.batch_directory is not None:
            dial.setDirectory(self.settings.batch_directory)
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setFilter("(*.xlsx Excel file")
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            if os.path.splitext(file_path)[1] == '':
                file_path += ".xlsx"
            self.result_file.setText(file_path)


group_sheet = """
QGroupBox {
    border: 1px solid gray;
    border-radius: 9px;
    margin-top: 0.5em;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 3px 0 3px;
}
"""


class CreatePlan(QWidget):
    def __init__(self, settings):
        super(CreatePlan, self).__init__()
        self.settings = settings
        # self.statistics = StatisticWidget(settings)
        self.plan = QListWidget()
        self.save_plan = QPushButton("Save plan")
        self.clean_plan = QPushButton("Clean plan")
        self.segment_profile = QListWidget()
        self.generate_mask = QPushButton("Generate mask")
        self.generate_mask.setToolTip("Mask need to have unique name")
        self.reuse_mask = QPushButton("Reuse mask")
        self.mask_name = QLineEdit()
        self.chose_profile = QPushButton("Segment Profile")
        self.statistic_list = QListWidget(self)
        self.statistic_name_prefix = QLineEdit(self)
        self.add_calculation = QPushButton("Add statistic calculation")
        self.information = QTextEdit()
        self.information.setReadOnly(True)
        self.protect = False

        self.statistic_list.currentTextChanged[str_type].connect(self.show_statistics)
        self.segment_profile.currentTextChanged[str_type].connect(self.show_segment)

        plan_box = QGroupBox("Calculate plan:")
        lay = QVBoxLayout()
        lay.addWidget(self.plan)
        bt_lay = QHBoxLayout()
        bt_lay.addWidget(self.save_plan)
        bt_lay.addWidget(self.clean_plan)
        lay.addLayout(bt_lay)
        plan_box.setLayout(lay)
        plan_box.setStyleSheet(group_sheet)

        mask_box = QGroupBox("Mask")
        mask_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.mask_name)
        bt_lay = QHBoxLayout()
        bt_lay.addWidget(self.generate_mask)
        bt_lay.addWidget(self.reuse_mask)
        lay.addLayout(bt_lay)
        mask_box.setLayout(lay)

        segment_box = QGroupBox("Segmentation")
        segment_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.segment_profile)
        lay.addWidget(self.chose_profile)
        segment_box.setLayout(lay)

        statistic_box = QGroupBox("Statistics")
        statistic_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.statistic_list)
        lab = QLabel("Name prefix:")
        lab.setToolTip("Prefix added before each column name")
        lay.addWidget(lab)
        lay.addWidget(self.statistic_name_prefix)
        lay.addWidget(self.add_calculation)
        statistic_box.setLayout(lay)

        info_box = QGroupBox("Information")
        info_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.information)
        info_box.setLayout(lay)

        layout = QGridLayout()
        layout.addWidget(plan_box, 0, 0, 2, 1)
        layout.addWidget(mask_box, 0, 1)
        layout.addWidget(segment_box, 1, 1)
        layout.addWidget(statistic_box, 2, 0)
        layout.addWidget(info_box, 2, 1)
        self.setLayout(layout)

    def showEvent(self, _):
        print("buka")
        new_statistics = list(sorted(self.settings.statistics_profile_dict.keys()))
        new_segment = list(sorted(self.settings.profiles.keys()))
        if self.statistic_list.currentItem() is not None:
            text = str(self.statistic_list.currentItem().text())
            try:
                statistic_index = new_statistics.index(text)
            except ValueError:
                statistic_index = -1
        else:
            statistic_index = -1
        if self.segment_profile.currentItem() is not None:
            text = str(self.segment_profile.currentItem().text())
            try:
                segment_index = new_segment.index(text)
            except ValueError:
                segment_index = -1
        else:
            segment_index = -1
        self.protect = True
        self.statistic_list.clear()
        self.statistic_list.addItems(new_statistics)
        if statistic_index != -1:
            self.statistic_list.setCurrentRow(statistic_index)

        self.segment_profile.clear()
        self.segment_profile.addItems(new_segment)
        if segment_index != -1:
            self.segment_profile.setCurrentRow(segment_index)
        self.protect = False

    def show_statistics(self, text):
        if self.protect:
            return
        if str(text) != "":
            self.information.setText(str(self.settings.statistics_profile_dict[str(text)]))

    def show_segment(self, text):
        if self.protect:
            return
        if str(text) != "":
            self.information.setText(str(self.settings.profiles[str(text)]))


class StatisticWidget(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings):
        super(StatisticWidget, self).__init__()
        self.settings = settings


        layout = QVBoxLayout()
        layout.addWidget(QLabel("Statistics list:"))
        layout.addWidget(self.statistic_list)
        layout.addWidget(QLabel("Name prefix:"))
        layout.addWidget(self.name_prefix)
        layout.addWidget(self.add_calculation)
        self.setLayout(layout)

    def showEvent(self, _):
        new_items = list(self.settings.statistics_profile_dict.keys())
        if self.statistic_list.currentItem() is not None:
            text = str(self.statistic_list.currentItem().text())
            try:
                index = new_items.index(text)
            except ValueError:
                index = -1
        else:
            index = -1

        self.statistic_list.clear()
        self.statistic_list.addItems(new_items)
        if index != -1:
            self.statistic_list.setCurrentRow(index)


class CalculateInfo(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings):
        super(CalculateInfo, self).__init__()
        self.settings = settings
        self.calculate_plans = QListWidget(self)
        self.plan_view = QTextEdit(self)
        self.plan_view.setReadOnly(True)
        self.delete_plan = QPushButton("Delete plan")
        self.edit_plan = QPushButton("Edit plan")
        info_layout = QVBoxLayout()
        info_butt_layout = QHBoxLayout()
        info_butt_layout.addWidget(self.delete_plan)
        info_butt_layout.addWidget(self.edit_plan)
        info_layout.addLayout(info_butt_layout)
        info_chose_layout = QHBoxLayout()
        info_chose_layout.addWidget(self.calculate_plans)
        info_chose_layout.addWidget(self.plan_view)
        info_layout.addLayout(info_chose_layout)
        self.setLayout(info_layout)


class CalculatePlaner(QSplitter):
    """
    :type settings: Settings
    """
    def __init__(self, settings, parent):
        QWidget.__init__(self, parent)
        self.settings = settings
        info_widget = CalculateInfo(settings)
        self.addWidget(info_widget)
        self.create_plan = CreatePlan(settings)
        self.addWidget(self.create_plan)


class BatchWindow(QTabWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings):
        QTabWidget.__init__(self)
        self.setWindowTitle("Batch processing")
        self.settings = settings
        self.file_choose = FileChoose(self.settings, self)
        self.calculate_planer = CalculatePlaner(self.settings, self)
        self.addTab(self.file_choose, "Choose files")
        self.addTab(self.calculate_planer, "Calculate settings")
        self.working = False

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



