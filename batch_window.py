import os
from glob import glob
import multiprocessing
from backend import Settings
from batch_backed import CalculationPlan, MaskCreate, MaskUse
from copy import copy

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
        new_list = ["<no calculation>", "aa"] + list(self.settings.batch_plans.keys())
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
    """
    :type settings: Settings
    """

    plan_created = pyqtSignal()

    def __init__(self, settings):
        super(CreatePlan, self).__init__()
        self.settings = settings
        # self.statistics = StatisticWidget(settings)
        self.plan = QListWidget()
        self.save_plan_btn = QPushButton("Save plan")
        self.clean_plan_btn = QPushButton("Clean plan")
        self.remove_last_btn = QPushButton("Remove last")
        self.forgot_mask_btn = QPushButton("Forgot mask")
        self.cmap_save_btn = QPushButton("Save to cmap")
        self.project_save_btn = QPushButton("Save to project")
        self.forgot_mask_btn.setToolTip("Return to state on begin")
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
        self.mapping_file_button = QPushButton("Mask mapping file")
        self.swap_mask_name_button = QPushButton("Name Substitution")
        self.suffix_mask_name_button = QPushButton("Name suffix")
        self.base_mask_name = QLineEdit()
        self.swap_mask_name = QLineEdit()
        self.protect = False
        self.mask_set = set()
        self.calculation_plan = CalculationPlan()

        self.statistic_list.currentTextChanged[str_type].connect(self.show_statistics)
        self.segment_profile.currentTextChanged[str_type].connect(self.show_segment)
        self.mask_name.textChanged[str_type].connect(self.mask_name_changed)
        self.generate_mask.clicked.connect(self.create_mask)
        self.reuse_mask.clicked.connect(self.use_mask)
        self.clean_plan_btn.clicked.connect(self.clean_plan)
        self.remove_last_btn.clicked.connect(self.remove_last)
        self.base_mask_name.textChanged.connect(self.file_mask_text_changed)
        self.swap_mask_name.textChanged.connect(self.file_mask_text_changed)
        self.chose_profile.clicked.connect(self.add_segmentation)
        self.add_calculation.clicked.connect(self.add_statistics)
        self.save_plan_btn.clicked.connect(self.add_calculation_plan)

        plan_box = QGroupBox("Calculate plan:")
        lay = QVBoxLayout()
        lay.addWidget(self.plan)
        bt_lay = QGridLayout()
        bt_lay.addWidget(self.save_plan_btn, 0, 0)
        bt_lay.addWidget(self.clean_plan_btn, 0, 1)
        bt_lay.addWidget(self.remove_last_btn, 1, 0)
        bt_lay.addWidget(self.forgot_mask_btn, 1, 1)
        bt_lay.addWidget(self.cmap_save_btn, 2, 0)
        bt_lay.addWidget(self.project_save_btn, 2, 1)
        lay.addLayout(bt_lay)
        plan_box.setLayout(lay)
        plan_box.setStyleSheet(group_sheet)

        file_mask_box = QGroupBox("Mask from file")
        file_mask_box.setStyleSheet(group_sheet)
        lay = QGridLayout()
        lay.addWidget(self.mapping_file_button, 0, 0, 1, 2)
        lay.addWidget(self.base_mask_name, 1, 0)
        lay.addWidget(self.swap_mask_name, 1, 1)
        lay.addWidget(self.suffix_mask_name_button, 2, 0)
        lay.addWidget(self.swap_mask_name_button, 2, 1)
        file_mask_box.setLayout(lay)

        mask_box = QGroupBox("Mask from segmentation")
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
        layout.addWidget(plan_box, 0, 0, 3, 1)
        layout.addWidget(file_mask_box, 0, 1)
        layout.addWidget(mask_box, 1, 1)
        layout.addWidget(segment_box, 2, 1)
        layout.addWidget(statistic_box, 3, 0)
        layout.addWidget(info_box, 3, 1)
        self.setLayout(layout)

        self.reuse_mask.setDisabled(True)
        self.generate_mask.setDisabled(True)
        self.chose_profile.setDisabled(True)
        self.add_calculation.setDisabled(True)
        self.swap_mask_name_button.setDisabled(True)
        self.suffix_mask_name_button.setDisabled(True)

    def create_mask(self):
        text = str(self.mask_name.text())
        if text in self.mask_set:
            QMessageBox.warning(self, "Already exists", "Mask with this name already exists", QMessageBox.Ok)
            return
        self.plan.addItem("Create mask: {}".format(text))
        self.mask_set.add(text)
        self.calculation_plan.add_step(MaskCreate(text))
        self.generate_mask.setDisabled(True)
        self.reuse_mask.setDisabled(False)

    def use_mask(self):
        text = str(self.mask_name.text())
        if text not in self.mask_set:
            QMessageBox.warning(self, "Don`t exists", "Mask with this name do not exists", QMessageBox.Ok)
            return
        self.plan.addItem("Use mask: {}".format(text))
        self.calculation_plan.add_step(MaskUse(text))

    def mask_name_changed(self, text):
        if str(text) in self.mask_set:
            self.generate_mask.setDisabled(True)
            self.reuse_mask.setDisabled(False)
        else:
            self.generate_mask.setDisabled(False)
            self.reuse_mask.setDisabled(True)

    def add_segmentation(self):
        text = str(self.segment_profile.currentItem().text())
        profile = self.settings.segmentation_profiles_dict[text]
        self.plan.addItem("Segmentation: {}".format(profile.name))
        self.calculation_plan.add_step(profile)

    def add_statistics(self):
        text = str(self.statistic_list.currentItem().text())
        statistics = self.settings.statistics_profile_dict[text]
        prefix = str(self.statistic_name_prefix.text()).strip()
        if prefix == "":
            self.plan.addItem("Statistics: {}".format(statistics.name))
            self.calculation_plan.add_step(statistics)
        else:
            statistics = copy(statistics)
            statistics.name_prefix = prefix
            self.plan.addItem("Statistics: {} with prefix: {}".format(statistics.name, prefix))
            self.calculation_plan.add_step(statistics)

    def remove_last(self):
        if len(self.calculation_plan) > 0:
            self.calculation_plan.pop()
            # TODO Something better if need more information
            self.plan.takeItem(len(self.calculation_plan))

    def clean_plan(self):
        self.calculation_plan = CalculationPlan()
        self.plan.clear()
        self.mask_set = set()

    def file_mask_text_changed(self):
        if str(self.base_mask_name.text()).strip() != "":
            self.suffix_mask_name_button.setEnabled(True)
            if str(self.swap_mask_name.text()).strip() != "":
                self.swap_mask_name_button.setEnabled(True)
            else:
                self.swap_mask_name_button.setDisabled(True)
        else:
            self.suffix_mask_name_button.setDisabled(True)
            self.swap_mask_name_button.setDisabled(True)

    def add_calculation_plan(self, used_text=None):
        print(used_text)
        if used_text is None or isinstance(used_text, bool):
            text, ok = QInputDialog.getText(self, "Plan title", "Set plan title")
        else:
            text, ok = QInputDialog.getText(self, "Plan title", "Set plan title. Previous ({}) "
                                                                "is already in use".format(used_text))
        if ok:
            text = str(text)
            if text in self.settings.batch_plans:
                self.add_calculation_plan(text)
                return
            plan = copy(self.calculation_plan)
            plan.set_name(text)
            self.settings.batch_plans[text] = plan
            self.plan_created.emit()

    def showEvent(self, _):
        new_statistics = list(sorted(self.settings.statistics_profile_dict.keys()))
        new_segment = list(sorted(self.settings.segmentation_profiles_dict.keys()))
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
            if self.calculation_plan.is_segmentation():
                self.add_calculation.setEnabled(True)
            else:
                self.add_calculation.setDisabled(True)
        else:
            self.add_calculation.setDisabled(True)

    def show_segment(self, text):
        if self.protect:
            return
        if str(text) != "":
            self.information.setText(str(self.settings.segmentation_profiles_dict[str(text)]))
            self.chose_profile.setEnabled(True)
        else:
            self.chose_profile.setDisabled(True)


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
        self.protect = False

    def update_plan_list(self):
        new_plan_list = list(sorted(self.settings.batch_plans.keys()))
        if self.calculate_plans.currentItem() is not None:
            text = str(self.calculate_plans.currentItem().text())
            try:
                index = new_plan_list.index(text)
            except ValueError:
                index = -1
        else:
            index = -1
        self.protect = True
        self.calculate_plans.clear()
        self.calculate_plans.addItems(new_plan_list)
        if index != -1:
            self.calculate_plans.setCurrentRow(index)
        else:
            self.plan_view.setText("")
        self.protect = False


class CalculatePlaner(QSplitter):
    """
    :type settings: Settings
    """
    def __init__(self, settings, parent):
        QWidget.__init__(self, parent)
        self.settings = settings
        self.info_widget = CalculateInfo(settings)
        self.addWidget(self.info_widget)
        self.create_plan = CreatePlan(settings)
        self.create_plan.plan_created.connect(self.info_widget.update_plan_list)
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



