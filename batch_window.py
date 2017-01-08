# coding=utf-8
import os
from glob import glob
import multiprocessing
from backend import Settings, CalculationPlan, MaskCreate, MaskUse, Operations, CmapProfile, MaskSuffix, MaskSub, \
    MaskFile, ProjectSave, UNITS_LIST
from batch_backed import BatchManager
from copy import copy
from io_functions import GaussUse
from backend import StatisticProfile, SegmentationProfile
from universal_gui_part import Spacing, right_label
from global_settings import file_folder

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

    def get_paths(self):
        return list(sorted(self.files_to_proceed))


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

        self.run_button.clicked.connect(self.prepare_calculation)

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
        dial = CalculationPrepare(self.files_widget.get_paths(), plan, str(self.result_file.text()), self.settings)
        if dial.exec_():
            final_settings = dial.get_data()


    def showEvent(self, _):
        current_calc = str(self.calculation_choose.currentText())
        new_list = ["<no calculation>"] + list(self.settings.batch_plans.keys())
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
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setFilter("(*.xlsx Excel file")
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            if os.path.splitext(file_path)[1] == '':
                file_path += ".xlsx"
            self.result_file.setText(file_path)


class CmapSavePrepare(QDialog):
    """
    :type settings: Settings
    """
    def __init__(self, text, title="Cmap settings"):
        super(CmapSavePrepare, self).__init__()
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        self.setWindowTitle(title)
        self.gauss_type = QComboBox(self)
        self.gauss_type.addItems(["No gauss", "2d gauss", "2d + 3d gauss"])
        self.center_data = QCheckBox(self)
        self.center_data.setChecked(True)
        self.with_statistics = QCheckBox(self)
        self.with_statistics.setChecked(True)
        self.rotation_axis = QComboBox(self)
        self.rotation_axis.addItems(["None", "x", "y", "z"])
        self.cut_data = QCheckBox(self)
        self.cut_data.setChecked(True)
        self.suffix_text = QLineEdit()
        grid = QGridLayout()
        grid.addWidget(QLabel("Gauss type"), 0, 0)
        grid.addWidget(self.gauss_type, 0, 1)
        grid.addWidget(QLabel("Center data"), 1, 0)
        grid.addWidget(self.center_data, 1, 1)
        # grid.addWidget(QLabel("With statistics"), 2, 0)
        # grid.addWidget(self.with_statistics, 2, 1)
        grid.addWidget(QLabel("Rotation axis"), 3, 0)
        grid.addWidget(self.rotation_axis, 3, 1)
        grid.addWidget(QLabel("Cut obsolete area"), 4, 0)
        grid.addWidget(self.cut_data, 4, 1)
        grid.addWidget(QLabel("Suffix_text"), 5, 0)
        grid.addWidget(self.suffix_text, 5, 1)

        close = QPushButton("Cancel")
        close.clicked.connect(self.close)
        save = QPushButton("Accept")
        save.clicked.connect(self.save)

        button_layout = QHBoxLayout()
        button_layout.addWidget(close)
        button_layout.addStretch()
        button_layout.addWidget(save)

        layout = QVBoxLayout()
        layout.addWidget(text_label)
        layout.addLayout(grid)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def save(self):
        self.accept()

    def get_result(self):
        options = {"No gauss": GaussUse.no_gauss, "2d gauss": GaussUse.gauss_2d, "2d + 3d gauss": GaussUse.gauss_3d}
        return CmapProfile(suffix=str(self.suffix_text.text()), gauss_type=options[str(self.gauss_type.currentText())],
                           center_data=self.center_data.isChecked(), rotation_axis=str(self.rotation_axis.currentText())
                           , cut_obsolete_are=self.cut_data.isChecked())

group_sheet = "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em;} " \
              "QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px;}"


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
        self.reuse_mask = QPushButton("Use mask")
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
        self.file_mask_name = QLineEdit()
        self.file_mask_name.setToolTip("Name of mask")
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
        self.file_mask_name.textChanged.connect(self.file_mask_text_changed)
        self.chose_profile.clicked.connect(self.add_segmentation)
        self.add_calculation.clicked.connect(self.add_statistics)
        self.save_plan_btn.clicked.connect(self.add_calculation_plan)
        self.forgot_mask_btn.clicked.connect(self.forgot_mask)
        self.cmap_save_btn.clicked.connect(self.save_to_cmap)
        self.swap_mask_name_button.clicked.connect(self.mask_by_substitution)
        self.suffix_mask_name_button.clicked.connect(self.mask_by_suffix)
        self.mapping_file_button.clicked.connect(self.mask_by_mapping)
        self.project_save_btn.clicked.connect(self.save_to_project)

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
        lay.addWidget(QLabel("Mask name:"), 0, 0)
        lay.addWidget(self.file_mask_name, 1, 0, 1, 2)
        lay.addWidget(self.mapping_file_button, 2, 0, 1, 2)
        lay.addWidget(QLabel("Suffix/Sub string:"), 3, 0)
        lay.addWidget(QLabel("Replace:"), 3, 1)
        lay.addWidget(self.base_mask_name, 4, 0)
        lay.addWidget(self.swap_mask_name, 4, 1)
        lay.addWidget(self.suffix_mask_name_button, 5, 0)
        lay.addWidget(self.swap_mask_name_button, 5, 1)
        file_mask_box.setLayout(lay)

        mask_box = QGroupBox("Mask from segmentation")
        mask_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Mask name:"))
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
        self.mapping_file_button.setDisabled(True)

    def save_to_project(self):
        suffix, ok = QInputDialog.getText(self, "Project file suffix", "Set project name suffix")
        if ok:
            suffix = str(suffix)
            text = self.calculation_plan.add_step(ProjectSave(suffix))
            self.plan.addItem(text)

    def mask_by_mapping(self):
        name = str(self.file_mask_name.text()).strip()
        text = self.calculation_plan.add_step(MaskFile(name, ""))
        self.plan.addItem(text)
        self.mask_set.add(name)
        self.file_mask_text_changed()
        self.mask_name_changed(self.mask_name.text)

    def mask_by_suffix(self):
        name = str(self.file_mask_name.text()).strip()
        suffix = str(self.base_mask_name.text()).strip()
        text = self.calculation_plan.add_step(MaskSuffix(name, suffix))
        self.plan.addItem(text)
        self.mask_set.add(name)
        self.file_mask_text_changed()
        self.mask_name_changed(self.mask_name.text)

    def mask_by_substitution(self):
        name = str(self.file_mask_name.text()).strip()
        base = str(self.base_mask_name.text()).strip()
        repl = str(self.swap_mask_name.text()).strip()
        text = self.calculation_plan.add_step(MaskSub(name, base, repl))
        self.plan.addItem(text)
        self.mask_set.add(name)
        self.file_mask_text_changed()
        self.mask_name_changed(self.mask_name.text)

    def save_to_cmap(self):
        dial = CmapSavePrepare("Settings for cmap create")
        if dial.exec_():
            text = self.calculation_plan.add_step(dial.get_result())
            self.plan.addItem(text)

    def forgot_mask(self):
        text = self.calculation_plan.add_step(Operations.clean_mask)
        self.plan.addItem(text)

    def create_mask(self):
        text = str(self.mask_name.text())
        if text in self.mask_set:
            QMessageBox.warning(self, "Already exists", "Mask with this name already exists", QMessageBox.Ok)
            return
        self.mask_set.add(text)
        text = self.calculation_plan.add_step(MaskCreate(text))
        self.plan.addItem(text)
        self.generate_mask.setDisabled(True)
        self.reuse_mask.setDisabled(False)

    def use_mask(self):
        text = str(self.mask_name.text())
        if text not in self.mask_set:
            QMessageBox.warning(self, "Don`t exists", "Mask with this name do not exists", QMessageBox.Ok)
            return
        text = self.calculation_plan.add_step(MaskUse(text))
        self.plan.addItem(text)

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
        text = self.calculation_plan.add_step(profile)
        self.plan.addItem(text)

    def add_statistics(self):
        text = str(self.statistic_list.currentItem().text())
        statistics = self.settings.statistics_profile_dict[text]
        statistics_copy = copy(statistics)
        prefix = str(self.statistic_name_prefix.text()).strip()
        statistics_copy.name_prefix = prefix
        text = self.calculation_plan.add_step(statistics_copy)
        self.plan.addItem(text)

    def remove_last(self):
        if len(self.calculation_plan) > 0:
            self.calculation_plan.pop()
            # TODO Something better if need more information
            el = self.plan.takeItem(len(self.calculation_plan))
            if isinstance(el, MaskCreate):
                self.mask_set.remove(el.name)

    def clean_plan(self):
        self.calculation_plan = CalculationPlan()
        self.plan.clear()
        self.mask_set = set()

    def file_mask_text_changed(self):
        name = str(self.file_mask_name.text()).strip()
        if name == "" or name in self.mask_set:
            self.suffix_mask_name_button.setDisabled(True)
            self.swap_mask_name_button.setDisabled(True)
            self.mapping_file_button.setDisabled(True)
        else:
            self.mapping_file_button.setEnabled(True)
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

    def edit_plan(self):
        plan = self.sender().plan_to_edit  # type: CalculationPlan
        self.calculation_plan = plan
        self.plan.clear()
        self.mask_set.clear()
        for el in plan.execution_list:
            self.plan.addItem(CalculationPlan.get_el_name(el))
            if isinstance(el, MaskCreate):
                self.mask_set.add(el.name)


class CalculateInfo(QWidget):
    """
    :type settings: Settings
    """
    plan_to_edit_signal = pyqtSignal()

    def __init__(self, settings):
        super(CalculateInfo, self).__init__()
        self.settings = settings
        self.calculate_plans = QListWidget(self)
        self.plan_view = QTreeWidget(self)
        self.delete_plan_btn = QPushButton("Delete plan")
        self.edit_plan_btn = QPushButton("Edit plan")
        self.export_plans_btn = QPushButton("Export plans")
        self.import_plans_btn = QPushButton("Import plans")
        info_layout = QVBoxLayout()
        info_butt_layout = QHBoxLayout()
        info_butt_layout.addWidget(self.delete_plan_btn)
        info_butt_layout.addWidget(self.import_plans_btn)
        # info_layout.addLayout(info_butt_layout)
        # info_butt_layout = QHBoxLayout()
        info_butt_layout.addWidget(self.export_plans_btn)
        info_butt_layout.addWidget(self.edit_plan_btn)
        info_layout.addLayout(info_butt_layout)
        info_chose_layout = QGridLayout()
        info_chose_layout.addWidget(QLabel("List o plans:"), 0, 0)
        info_chose_layout.addWidget(QLabel("Plan preview:"), 0, 1)
        info_chose_layout.addWidget(self.calculate_plans, 1, 0)
        info_chose_layout.addWidget(self.plan_view, 1, 1)
        info_layout.addLayout(info_chose_layout)
        self.setLayout(info_layout)
        self.calculate_plans.addItems(list(sorted(self.settings.batch_plans.keys())))
        self.protect = False
        self.plan_to_edit = None

        self.plan_view.header().close()
        self.calculate_plans.currentTextChanged.connect(self.plan_preview)
        self.delete_plan_btn.clicked.connect(self.delete_plan)
        self.edit_plan_btn.clicked.connect(self.edit_plan)
        self.export_plans_btn.clicked.connect(self.export_plans)
        self.import_plans_btn.clicked.connect(self.import_plans)

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
            pass
            # self.plan_view.setText("")
        self.protect = False

    def export_plans(self):
        dial = QFileDialog(self, "Export calculation plans")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        if self.settings.save_directory is not None:
            dial.setDirectory(self.settings.save_directory)
        dial.setFilter("Calculation plans (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("calculation_plans.json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self.settings.dump_calculation_plans(file_path)

    def import_plans(self):
        dial = QFileDialog(self, "Export calculation plans")
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        if self.settings.open_directory is not None:
            dial.setDirectory(self.settings.save_directory)
        dial.setFilter("Calculation plans (*.json)")
        dial.setDefaultSuffix("json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self.settings.load_calculation_plans(file_path)
            self.update_plan_list()

    def delete_plan(self):
        if self.calculate_plans.currentItem() is None:
            return
        text = str(self.calculate_plans.currentItem().text())
        if text == "":
            return
        if text in self.settings.batch_plans:
            del self.settings.batch_plans[text]
        self.update_plan_list()
        self.plan_view.clear()

    def edit_plan(self):
        if self.calculate_plans.currentItem() is None:
            return
        text = str(self.calculate_plans.currentItem().text())
        if text == "":
            return
        if text in self.settings.batch_plans:
            self.plan_to_edit = self.settings.batch_plans[text]
            self.plan_to_edit_signal.emit()

    def plan_preview(self, text):
        if self.protect:
            return
        text = str(text)
        if text.strip() == "":
            return
        plan = self.settings.batch_plans[str(text)]  # type: CalculationPlan
        self.plan_view.clear()
        for el in plan.execution_list:
            widget = QTreeWidgetItem(self.plan_view)
            widget.setText(0, CalculationPlan.get_el_name(el))
            if isinstance(el, StatisticProfile) or isinstance(el, SegmentationProfile):
                description = str(el).split("\n")
                for line in description[1:]:
                    if line.strip() == "":
                        continue
                    w = QTreeWidgetItem(widget)
                    w.setText(0, line)


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
        self.info_widget.plan_to_edit_signal.connect(self.create_plan.edit_plan)
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
        self.addTab(self.calculate_planer, "Prepare plan")
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


class CalculationPrepare(QDialog):
    def __init__(self, file_list, calculation_plan, statistic_file_path, settings):
        """

        :param file_list: list of files to proceed
        :type file_list: list[str]
        :param calculation_plan: calculation plan for this run
        :type calculation_plan: CalculationPlan
        :param statistic_file_path: path to statistic file
        :type statistic_file_path: str
        :param settings: settings object
        :type settings: Settings
        """
        super(CalculationPrepare, self).__init__()
        self.setWindowTitle("Calculation start")
        self.file_list = file_list
        self.calculation_plan = calculation_plan
        self.statistic_file_path = statistic_file_path
        self.settings = settings
        self.info_label = QLabel("information, <i><font color='blue'>warnings</font></i>, "
                                 "<b><font color='red'>errors</font><b>")
        self.voxel_size = Spacing("Voxel size", zip(['x:', 'y:', 'z:'], settings.voxel_size), self, units=UNITS_LIST,
                                  units_index=UNITS_LIST.index(settings.size_unit))
        all_prefix = os.path.commonprefix(file_list)
        if not os.path.exists(all_prefix):
            all_prefix = os.path.dirname(all_prefix)
        self.base_prefix = QLineEdit(all_prefix, self)
        self.result_prefix = QLineEdit(all_prefix, self)
        self.base_prefix_btn = QPushButton("Choose data prefix")
        self.result_prefix_btn = QPushButton("Choose save prefix")
        self.sheet_name = QLineEdit("Sheet1")
        self.statistic_file_path = QLineEdit(statistic_file_path)

        self.file_list_widget = QTreeWidget()
        self.file_list_widget.header().close()

        self.execute_btn = QPushButton("Execute")
        self.execute_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)

        layout = QGridLayout()
        layout.addWidget(self.info_label, 0, 0, 1, 0)
        layout.addWidget(self.voxel_size, 1, 0, 1, 3)
        layout.addWidget(right_label("Statistics sheet name:"), 3, 3)
        layout.addWidget(self.sheet_name, 3, 4)
        layout.addWidget(right_label("Statistics file path:"), 2, 3)
        layout.addWidget(self.statistic_file_path, 2, 4)

        layout.addWidget(right_label("Data prefix:"), 2, 0)
        layout.addWidget(self.base_prefix, 2, 1)
        layout.addWidget(self.base_prefix_btn, 2, 2)
        layout.addWidget(right_label("Save prefix:"), 3, 0)
        layout.addWidget(self.result_prefix, 3, 1)
        layout.addWidget(self.result_prefix_btn, 3, 2)

        layout.addWidget(self.file_list_widget, 5, 0, 3, 6)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.execute_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout, 8, 0, 1, 3)
        self.setLayout(layout)

    def get_data(self):
        res = {"file_list": self.file_list, "base_prefix": str(self.base_prefix.text()),
               "result_prefix": str(self.result_prefix.text()),
               "statistic_file_path": str(self.statistic_file_path.text()),
               "sheet_name": str(self.sheet_name.text()), "calculation_plan": self.calculation_plan}
        return res

    def verify_data(self):
        pass

    def showEvent(self, event):
        super(CalculationPrepare, self).showEvent(event)
        ok_icon = QIcon(os.path.join(file_folder, "icons", "task-accepted.png"))
        bad_icon = QIcon(os.path.join(file_folder, "icons", "task-reject.png"))
        all_prefix = os.path.commonprefix(self.file_list)
        if not os.path.exists(all_prefix):
            all_prefix = os.path.dirname(all_prefix)
        for file_path in self.file_list:
            widget = QTreeWidgetItem(self.file_list_widget)
            widget.setText(0, os.path.realpath(file_path, all_prefix))
            if not os.path.exists(file_path):
                widget.setIcon(0, bad_icon)
                widget.setToolTip(0, "File do not exists")
                sub_widget = QTreeWidgetItem(widget)
                sub_widget.setText(0, "File do not exists")
                continue

            widget.setIcon(0, ok_icon)
            widget.setToolTip(0, "Ok")


