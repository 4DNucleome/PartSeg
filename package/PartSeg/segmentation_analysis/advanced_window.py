import json
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Union, Optional, Tuple

from qtpy.QtCore import QByteArray, Qt, QEvent
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QTabWidget, QWidget, QListWidget, QTextEdit, QPushButton, QLineEdit, \
    QVBoxLayout, QLabel, QHBoxLayout, QListWidgetItem, QDialog, QDoubleSpinBox, QSpinBox, QGridLayout, QApplication, \
    QMessageBox, QFileDialog, QComboBox, QAbstractSpinBox, QInputDialog, \
    QPlainTextEdit, QFrame, QCheckBox

from PartSeg.utils.analysis.algorithm_description import SegmentationProfile, analysis_algorithm_dict
from ..common_gui.universal_gui_part import EnumComboBox
from ..common_gui.colors_choose import ColorSelector
from ..common_gui.custom_save_dialog import FormDialog
from ..common_gui.lock_checkbox import LockCheckBox
from .partseg_settings import PartSettings, MASK_COLORS
from .profile_export import ExportDialog, StringViewer, ImportDialog, ProfileDictViewer
from .statistic_widget import StatisticsWidget
from PartSeg.utils.analysis.statistics_calculation import StatisticProfile, STATISTIC_DICT, Node, Leaf, AreaType, PerComponent, \
    StatisticEntry
from ..utils.global_settings import static_file_folder
from ..utils.universal_const import UNIT_SCALE, Units


def h_line():
    toto = QFrame()
    toto.setFrameShape(QFrame.HLine)
    toto.setFrameShadow(QFrame.Sunken)
    return toto


class AdvancedSettings(QWidget):
    def __init__(self, settings: PartSettings):
        super().__init__()
        self._settings = settings
        self.export_btn = QPushButton("Export profile")
        self.export_btn.clicked.connect(self.export_profile)
        self.import_btn = QPushButton("Import profile")
        self.import_btn.clicked.connect(self.import_profiles)
        self.export_pipeline_btn = QPushButton("Export pipeline")
        self.export_pipeline_btn.clicked.connect(self.export_pipeline)
        self.import_pipeline_btn = QPushButton("Import pipeline")
        self.import_pipeline_btn.clicked.connect(self.import_pipeline)
        self.delete_btn = QPushButton("Delete profile")
        self.delete_btn.setDisabled(True)
        self.delete_btn.clicked.connect(self.delete_profile)
        self.multiple_files_chk = QCheckBox("Show multiple files widget")
        self.multiple_files_chk.setChecked(self._settings.get("multiple_files", False))
        self.multiple_files_chk.stateChanged.connect(partial(self._settings.set, "multiple_files"))
        self.rename_btn = QPushButton("Rename Profile")
        self.rename_btn.clicked.connect(self.rename_profile)
        self.rename_btn.setDisabled(True)
        self.voxel_size_label = QLabel()
        self.info_label = QPlainTextEdit()
        self.info_label.setReadOnly(True)
        self.profile_list = QListWidget()
        self.profile_list.currentTextChanged.connect(self.profile_chosen)
        self.pipeline_list = QListWidget()
        self.pipeline_list.currentTextChanged.connect(self.profile_chosen)
        self.spacing = [QDoubleSpinBox() for _ in range(3)]
        self.lock_spacing = LockCheckBox()
        self.lock_spacing.stateChanged.connect(self.spacing[1].setDisabled)
        self.lock_spacing.stateChanged.connect(self.synchronize_spacing)
        # noinspection PyUnresolvedReferences
        self.spacing[2].valueChanged.connect(self.synchronize_spacing)
        units_value = self._settings.get("units_value", Units.nm)
        for i, el in enumerate(self.spacing):
            el.setAlignment(Qt.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.NoButtons)
            el.setRange(0, 1000000)
            # noinspection PyUnresolvedReferences
            el.valueChanged.connect(self.image_spacing_change)
        self.units = EnumComboBox(Units)
        self.units.set_value(units_value)
        # noinspection PyUnresolvedReferences
        self.units.currentIndexChanged.connect(self.update_spacing)

        color, opacity = self._settings.get_from_profile("mask_presentation", (list(MASK_COLORS.keys())[0], 0.6))
        self.mask_color = QComboBox()
        self.mask_color.addItems(MASK_COLORS.keys())
        self.mask_opacity = QDoubleSpinBox()
        self.mask_opacity.setRange(0, 1)
        self.mask_opacity.setSingleStep(0.1)
        try:
            index = list(MASK_COLORS.keys()).index(color)
        except IndexError:
            index = 0
        self.mask_color.setCurrentIndex(index)
        self.mask_opacity.setValue(opacity)
        # noinspection PyUnresolvedReferences
        self.mask_opacity.valueChanged.connect(self.mask_prop_changed)
        # noinspection PyUnresolvedReferences
        self.mask_color.currentIndexChanged.connect(self.mask_prop_changed)

        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(self.lock_spacing)
        for txt, el in zip(["x", "y", "z"], self.spacing[::-1]):
            spacing_layout.addWidget(QLabel(txt + ":"))
            spacing_layout.addWidget(el)
        spacing_layout.addWidget(self.units)
        spacing_layout.addStretch(1)
        voxel_size_layout = QHBoxLayout()
        voxel_size_layout.addWidget(self.voxel_size_label)
        voxel_size_layout.addSpacing(30)
        mask_layout = QHBoxLayout()
        mask_layout.addWidget(QLabel("Mask mark color"))
        mask_layout.addWidget(self.mask_color)
        mask_layout.addWidget(QLabel("Mask mark opacity"))
        mask_layout.addWidget(self.mask_opacity)
        mask_layout.addStretch(1)
        profile_layout = QGridLayout()
        profile_layout.setSpacing(0)
        profile_layout.addWidget(QLabel("Profiles:"), 0, 0)
        profile_layout.addWidget(self.profile_list, 1, 0)
        profile_layout.addWidget(QLabel("Pipelines:"), 2, 0)
        profile_layout.addWidget(self.pipeline_list, 3, 0, 4, 1)
        profile_layout.addWidget(self.info_label, 1, 1, 3, 2)
        profile_layout.addWidget(self.export_btn, 4, 1)
        profile_layout.addWidget(self.import_btn, 4, 2)
        profile_layout.addWidget(self.export_pipeline_btn, 5, 1)
        profile_layout.addWidget(self.import_pipeline_btn, 5, 2)
        profile_layout.addWidget(self.delete_btn, 6, 1)
        profile_layout.addWidget(self.rename_btn, 6, 2)
        layout = QVBoxLayout()
        layout.addLayout(spacing_layout)
        layout.addLayout(voxel_size_layout)
        layout.addLayout(mask_layout)
        layout.addWidget(self.multiple_files_chk)

        layout.addLayout(profile_layout, 1)
        self.setLayout(layout)

    def mask_prop_changed(self):
        self._settings.set_in_profile("mask_presentation", (self.mask_color.currentText(), self.mask_opacity.value()))

    def profile_chosen(self, text):
        if text == "":
            self.delete_btn.setEnabled(False)
            self.rename_btn.setEnabled(False)
            self.info_label.setPlainText("")
            return
        try:
            if self.sender() == self.profile_list:
                profile = self._settings.segmentation_profiles[text]
                self.pipeline_list.selectionModel().clear()
            elif self.sender() == self.pipeline_list:
                profile = self._settings.segmentation_pipelines[text]
                self.profile_list.selectionModel().clear()
            else:
                return
        except KeyError:
            return

        # TODO update with knowledge from profile dict
        self.delete_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)
        self.info_label.setPlainText(profile.pretty_print(analysis_algorithm_dict))


    def synchronize_spacing(self):
        if self.lock_spacing.isChecked():
            self.spacing[1].setValue(self.spacing[2].value())

    def image_spacing_change(self):
        spacing = [el.value() / UNIT_SCALE[self.units.currentIndex()] for i, el in enumerate(self.spacing)]
        if not self.spacing[0].isEnabled():
            spacing = spacing[1:]
        self._settings.image_spacing = spacing

        voxel_size = 1
        for el in self._settings.image_spacing:
            voxel_size *= el * UNIT_SCALE[self.units.currentIndex()]
        self.voxel_size_label.setText(f"Voxel_size: {voxel_size} {self.units.get_value().name}"
                                      f"<sup>{len(self._settings.image_spacing)}</sup>")

    def update_spacing(self, index=None):
        voxel_size = 1
        value = self.units.get_value()
        if index is not None:
            self._settings.set("units_value", value)
        for i, (el, sp) in enumerate(zip(self.spacing[::-1], self._settings.image_spacing[::-1])):
            el.blockSignals(True)
            current_size = sp * UNIT_SCALE[self.units.currentIndex()]
            voxel_size *= current_size
            el.setValue(current_size)
            el.blockSignals(False)
        self.spacing[0].setDisabled(len(self._settings.image_spacing) == 2)
        self.voxel_size_label.setText(f"Voxel_size: {voxel_size} {value.name}"
                                      f"<sup>{len(self._settings.image_spacing)}</sup>")

    def update_profile_list(self):
        current_names = set(self._settings.segmentation_profiles.keys())
        self.profile_list.clear()
        self.profile_list.addItems(sorted(current_names))
        self.pipeline_list.clear()
        self.pipeline_list.addItems(sorted(set(self._settings.segmentation_pipelines.keys())))
        self.delete_btn.setDisabled(True)
        self.rename_btn.setDisabled(True)
        self.info_label.setPlainText("")

    def showEvent(self, a0):
        self.update_profile_list()
        self.update_spacing()

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate and self.isVisible():
            self.update_profile_list()
            self.update_spacing()
        return super().event(event)

    def delete_profile(self):
        text, dkt = "", {}
        if self.profile_list.selectedItems():
            text = self.profile_list.selectedItems()[0].text()
            dkt = self._settings.segmentation_profiles
        elif self.pipeline_list.selectedItems():
            text = self.pipeline_list.selectedItems()[0].text()
            dkt = self._settings.segmentation_pipelines
        if text != "":
            self.delete_btn.setDisabled(True)
            del dkt[text]
            self.update_profile_list()

    def export_profile(self):
        exp = ExportDialog(self._settings.segmentation_profiles, ProfileDictViewer)
        if not exp.exec_():
            return
        dial = QFileDialog(self, "Export profile segment")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setDirectory(self._settings.get("io.save_directory", str(Path.home())))
        dial.setNameFilter("Segment profile (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("segment_profile.json")
        dial.setHistory(dial.history() + self._settings.get_path_history())
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self._settings.set("io.save_directory", os.path.dirname(file_path))
            self._settings.add_path_history(os.path.dirname(file_path))
            data = dict([(x, self._settings.segmentation_profiles[x]) for x in exp.get_export_list()])
            with open(file_path, 'w') as ff:
                json.dump(data, ff, cls=self._settings.json_encoder_class, indent=2)

    def import_profiles(self):
        dial = QFileDialog(self, "Import profile segment")
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setDirectory(self._settings.get("io.save_directory", str(Path.home())))
        dial.setNameFilter("Segment profile (*.json)")
        dial.setHistory(dial.history() + self._settings.get_path_history())
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            save_dir = os.path.dirname(file_path)
            self._settings.set("io.save_directory", save_dir)
            self._settings.add_path_history(save_dir)
            profs,err = self._settings.load_part(file_path)
            if err:
                QMessageBox.warning(self, "Import error", "error during importing, part of data were filtered.")
            profiles_dict = self._settings.segmentation_profiles
            imp = ImportDialog(profs, profiles_dict, ProfileDictViewer)
            if not imp.exec_():
                return
            for original_name, final_name in imp.get_import_list():
                profiles_dict[final_name] = profs[original_name]
            self._settings.dump()
            self.update_profile_list()

    def export_pipeline(self):
        exp = ExportDialog(self._settings.segmentation_pipelines, ProfileDictViewer)
        if not exp.exec_():
            return
        dial = QFileDialog(self, "Export pipeline segment")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setDirectory(self._settings.get("io.save_directory", ""))
        dial.setNameFilter("Segment pipeline (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("segment_pipeline.json")
        dial.setHistory(dial.history() + self._settings.get_path_history())
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            data = dict([(x, self._settings.segmentation_pipelines[x]) for x in exp.get_export_list()])
            with open(file_path, 'w') as ff:
                json.dump(data, ff, cls=self._settings.json_encoder_class, indent=2)
            self._settings.set("io.save_directory", os.path.dirname(file_path))
            self._settings.add_path_history(os.path.dirname(file_path))

    def import_pipeline(self):
        dial = QFileDialog(self, "Import pipeline segment")
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setDirectory(self._settings.get("io.save_directory", ""))
        dial.setNameFilter("Segment pipeline (*.json)")
        dial.setHistory(dial.history() + self._settings.get_path_history())
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self._settings.set("io.save_directory", os.path.dirname(file_path))
            self._settings.add_path_history(os.path.dirname(file_path))
            profs, err = self._settings.load_part(file_path)
            if err:
                QMessageBox.warning(self, "Import error", "error during importing, part of data were filtered.")
            profiles_dict = self._settings.segmentation_pipelines
            imp = ImportDialog(profs, profiles_dict, ProfileDictViewer)
            if not imp.exec_():
                return
            for original_name, final_name in imp.get_import_list():
                profiles_dict[final_name] = profs[original_name]
            self._settings.dump()
            self.update_profile_list()

    def rename_profile(self):
        profile_name, profiles_dict = "", {}
        if self.profile_list.selectedItems():
            profile_name = self.profile_list.selectedItems()[0].text()
            profiles_dict = self._settings.segmentation_profiles
        elif self.pipeline_list.selectedItems():
            profile_name = self.pipeline_list.selectedItems()[0].text()
            profiles_dict = self._settings.segmentation_pipelines
        if profile_name == "":
            return
        text, ok = QInputDialog.getText(self, "New profile name", f"New name for {profile_name}", text=profile_name)
        if ok:
            text = text.strip()
            if text in profiles_dict.keys():
                res = QMessageBox.warning(self, "Already exist",
                                          f"Profile with name {text} already exist. Would you like to overwrite?",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if res == QMessageBox.No:
                    self.rename_profile()
                    return
            profiles_dict[text] = profiles_dict.pop(profile_name)
            self._settings.dump()
            self.update_profile_list()


class StatisticListWidgetItem(QListWidgetItem):
    def __init__(self, stat: Union[Node, Leaf], *args, **kwargs):
        super().__init__(stat.pretty_print(STATISTIC_DICT), *args, **kwargs)
        self.stat = stat


class StatisticsSettings(QWidget):
    """
    :type settings: Settings
    """

    def __init__(self, settings: PartSettings):
        super(StatisticsSettings, self).__init__()
        self.chosen_element: Optional[StatisticListWidgetItem] = None
        self.chosen_element_area: Optional[Tuple[AreaType, float]] = None
        self.settings = settings
        self.profile_list = QListWidget(self)
        self.profile_description = QTextEdit(self)
        self.profile_description.setReadOnly(True)
        self.profile_options = QListWidget()
        self.profile_options_chosen = QListWidget()
        self.statistic_area_choose = EnumComboBox(AreaType)
        self.per_component = EnumComboBox(PerComponent)
        self.power_num = QDoubleSpinBox()
        self.power_num.setDecimals(3)
        self.power_num.setRange(-100, 100)
        self.power_num.setValue(1)
        # self.statistic_object_choose.addItems(["Mask", "Segmentation", "Mask without segmentation"])
        self.choose_butt = QPushButton(u"→", self)
        self.discard_butt = QPushButton(u"←", self)
        self.proportion_butt = QPushButton(u"Ratio", self)
        self.proportion_butt.setToolTip("Create proportion from two statistics")
        self.move_up = QPushButton(u"↑", self)
        self.move_down = QPushButton(u"↓", self)
        self.remove_button = QPushButton("Remove")
        self.save_butt = QPushButton("Save  profile")
        self.save_butt.setToolTip("Set name for profile and choose at least one statistic")
        self.save_butt_with_name = QPushButton("Save profile with custom parameters names")
        self.save_butt_with_name.setToolTip("Set name for profile and choose at least one statistic")
        self.reset_butt = QPushButton("Clear")
        self.soft_reset_butt = QPushButton("Remove user statistics")
        self.profile_name = QLineEdit(self)

        self.delete_profile_butt = QPushButton("Delete profile")
        self.restore_builtin_profiles = QPushButton("Restore builtin profiles")
        self.export_profiles_butt = QPushButton("Export profiles")
        self.import_profiles_butt = QPushButton("Import profiles")
        self.edit_profile_butt = QPushButton("Edit profile")

        self.choose_butt.setDisabled(True)
        self.choose_butt.clicked.connect(self.choose_option)
        self.discard_butt.setDisabled(True)
        self.discard_butt.clicked.connect(self.discard_option)
        self.proportion_butt.setDisabled(True)
        self.proportion_butt.clicked.connect(self.proportion_action)
        self.save_butt.setDisabled(True)
        self.save_butt.clicked.connect(self.save_action)
        self.save_butt_with_name.setDisabled(True)
        self.save_butt_with_name.clicked.connect(self.named_save_action)
        self.profile_name.textChanged.connect(self.name_changed)
        self.move_down.setDisabled(True)
        self.move_down.clicked.connect(self.move_down_fun)
        self.move_up.setDisabled(True)
        self.move_up.clicked.connect(self.move_up_fun)
        self.remove_button.setDisabled(True)
        self.remove_button.clicked.connect(self.remove_element)
        self.reset_butt.clicked.connect(self.reset_action)
        self.soft_reset_butt.clicked.connect(self.soft_reset)
        self.delete_profile_butt.setDisabled(True)
        self.delete_profile_butt.clicked.connect(self.delete_profile)
        self.export_profiles_butt.clicked.connect(self.export_statistic_profiles)
        self.import_profiles_butt.clicked.connect(self.import_statistic_profiles)
        self.edit_profile_butt.clicked.connect(self.edit_profile)

        self.profile_list.itemSelectionChanged.connect(self.profile_chosen)
        self.profile_options.itemSelectionChanged.connect(self.create_selection_changed)
        self.profile_options_chosen.itemSelectionChanged.connect(self.create_selection_chosen_changed)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Defined profiles list:"))
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(self.profile_list)
        profile_layout.addWidget(self.profile_description)
        profile_buttons_layout = QHBoxLayout()
        profile_buttons_layout.addWidget(self.delete_profile_butt)
        profile_buttons_layout.addWidget(self.restore_builtin_profiles)
        self.restore_builtin_profiles.setHidden(True)
        profile_buttons_layout.addWidget(self.export_profiles_butt)
        profile_buttons_layout.addWidget(self.import_profiles_butt)
        profile_buttons_layout.addWidget(self.edit_profile_butt)
        profile_buttons_layout.addStretch()
        layout.addLayout(profile_layout)
        layout.addLayout(profile_buttons_layout)
        heading_layout = QHBoxLayout()
        heading_layout.addWidget(QLabel("Create profile"), 1)
        heading_layout.addWidget(h_line(), 6)
        layout.addLayout(heading_layout)
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Profile name:"))
        name_layout.addWidget(self.profile_name)
        name_layout.addStretch()
        name_layout.addWidget(QLabel("Per component:"))
        name_layout.addWidget(self.per_component)
        name_layout.addWidget(QLabel("Area:"))
        name_layout.addWidget(self.statistic_area_choose)
        name_layout.addWidget(QLabel("to power:"))
        name_layout.addWidget(self.power_num)
        """name_layout.addWidget(self.reversed_brightness)
        name_layout.addWidget(QLabel("Gauss image:"))
        name_layout.addWidget(self.gauss_img)
        name_layout.addWidget(QLabel("Gauss radius (pix):"))
        name_layout.addWidget(self.gauss_radius)"""
        layout.addLayout(name_layout)
        create_layout = QHBoxLayout()
        create_layout.addWidget(self.profile_options)
        butt_op_layout = QVBoxLayout()
        butt_op_layout.addStretch()
        butt_op_layout.addWidget(self.choose_butt)
        butt_op_layout.addWidget(self.discard_butt)
        butt_op_layout.addWidget(self.proportion_butt)
        butt_op_layout.addWidget(self.reset_butt)
        butt_op_layout.addStretch()
        create_layout.addLayout(butt_op_layout)
        create_layout.addWidget(self.profile_options_chosen)
        butt_move_layout = QVBoxLayout()
        butt_move_layout.addStretch()
        butt_move_layout.addWidget(self.move_up)
        butt_move_layout.addWidget(self.move_down)
        butt_move_layout.addWidget(self.remove_button)
        butt_move_layout.addStretch()
        create_layout.addLayout(butt_move_layout)
        layout.addLayout(create_layout)
        save_butt_layout = QHBoxLayout()
        save_butt_layout.addWidget(self.soft_reset_butt)
        save_butt_layout.addStretch()
        save_butt_layout.addWidget(self.save_butt)
        save_butt_layout.addWidget(self.save_butt_with_name)
        layout.addLayout(save_butt_layout)
        self.setLayout(layout)

        for name, profile in STATISTIC_DICT.items():
            help_text = profile.get_description()
            lw = StatisticListWidgetItem(profile.get_starting_leaf())
            lw.setToolTip(help_text)
            self.profile_options.addItem(lw)
        self.profile_list.addItems(list(sorted(self.settings.statistic_profiles.keys())))
        if self.profile_list.count() == 0:
            self.export_profiles_butt.setDisabled(True)

    def remove_element(self):
        elem = self.profile_options_chosen.currentItem()
        if elem is None:
            return
        index = self.profile_options_chosen.currentRow()
        self.profile_options_chosen.takeItem(index)
        if self.profile_options_chosen.count() == 0:
            self.move_down.setDisabled(True)
            self.move_up.setDisabled(True)
            self.remove_button.setDisabled(True)
            self.discard_butt.setDisabled(True)
            self.save_butt.setDisabled(True)
            self.save_butt_with_name.setDisabled(True)
        pass

    def delete_profile(self):
        row = self.profile_list.currentRow()
        item = self.profile_list.currentItem()
        del self.settings.statistic_profiles[str(item.text())]
        self.profile_list.takeItem(row)
        if self.profile_list.count() == 0:
            self.delete_profile_butt.setDisabled(True)

    def profile_chosen(self):
        self.delete_profile_butt.setEnabled(True)
        if self.profile_list.count() == 0:
            self.profile_description.setText("")
            return
        item = self.profile_list.currentItem()
        if item is None:
            self.profile_description.setText("")
            return
        profile = self.settings.statistic_profiles[item.text()]
        self.profile_description.setText(str(profile))

    def create_selection_changed(self):
        self.choose_butt.setEnabled(True)
        self.proportion_butt.setEnabled(True)

    def proportion_action(self):
        # TODO use get_parameters
        if self.chosen_element is None:
            item = self.profile_options.currentItem()
            self.chosen_element_area = \
                self.get_parameters(deepcopy(item.stat), self.statistic_area_choose.get_value(),
                                    self.per_component.get_value(), self.power_num.value())
            if self.chosen_element_area is None:
                return
            self.chosen_element = item
            item.setIcon(QIcon(os.path.join(static_file_folder, "icons", "task-accepted.png")))
            # self.statistic_area_choose.get_value(), self.per_component.get_value(), self.power_num.value()
        elif self.profile_options.currentItem() == self.chosen_element and \
                self.statistic_area_choose.get_value() == self.chosen_element_area.area and \
                self.per_component.get_value() == self.chosen_element_area.per_component:
            self.chosen_element.setIcon(QIcon())
            self.chosen_element = None
        else:
            item: StatisticListWidgetItem = self.profile_options.currentItem()
            leaf = self.get_parameters(deepcopy(item.stat), self.statistic_area_choose.get_value(),
                                       self.per_component.get_value(), self.power_num.value())
            if leaf is None:
                return 
            lw = StatisticListWidgetItem(
                Node(op="/", left=self.chosen_element_area,
                     right=leaf))
            lw.setToolTip("User defined")
            self.profile_options_chosen.addItem(lw)
            self.chosen_element.setIcon(QIcon())
            self.chosen_element = None
            self.chosen_element_area = None
            if self.good_name():
                self.save_butt.setEnabled(True)
                self.save_butt_with_name.setEnabled(True)
            if self.profile_options.count() == 0:
                self.choose_butt.setDisabled(True)

    def create_selection_chosen_changed(self):
        # print(self.profile_options_chosen.count())
        self.remove_button.setEnabled(True)
        if self.profile_options_chosen.count() == 0:
            self.move_down.setDisabled(True)
            self.move_up.setDisabled(True)
            self.remove_button.setDisabled(True)
            return
        self.discard_butt.setEnabled(True)
        if self.profile_options_chosen.currentRow() != 0:
            self.move_up.setEnabled(True)
        else:
            self.move_up.setDisabled(True)
        if self.profile_options_chosen.currentRow() != self.profile_options_chosen.count() - 1:
            self.move_down.setEnabled(True)
        else:
            self.move_down.setDisabled(True)

    def good_name(self):
        if str(self.profile_name.text()).strip() == "":
            return False
        return True

    def move_down_fun(self):
        row = self.profile_options_chosen.currentRow()
        item = self.profile_options_chosen.takeItem(row)
        self.profile_options_chosen.insertItem(row + 1, item)
        self.profile_options_chosen.setCurrentRow(row + 1)
        self.create_selection_chosen_changed()

    def move_up_fun(self):
        row = self.profile_options_chosen.currentRow()
        item = self.profile_options_chosen.takeItem(row)
        self.profile_options_chosen.insertItem(row - 1, item)
        self.profile_options_chosen.setCurrentRow(row - 1)
        self.create_selection_chosen_changed()

    def name_changed(self):
        if self.good_name() and self.profile_options_chosen.count() > 0:
            self.save_butt.setEnabled(True)
            self.save_butt_with_name.setEnabled(True)
        else:
            self.save_butt.setDisabled(True)
            self.save_butt_with_name.setDisabled(True)

    @staticmethod
    def get_parameters(node: Union[Node, Leaf], area: AreaType, component: PerComponent, power: float):
        if isinstance(node, Node):
            return node
        node = node.replace_(power=power)
        if node.area is None:
            node = node.replace_(area=area)
        if node.per_component is None:
            node = node.replace_(per_component=component)
        try:
            arguments = STATISTIC_DICT[str(node.name)].get_fields()
            if len(arguments) > 0 and len(node.dict) == 0:
                dial = FormDialog(arguments)
                if dial.exec():
                    node = node._replace(dict=dial.get_values())
                else:
                    return
        except KeyError:
            pass
        return node

    def choose_option(self):
        selected_item = self.profile_options.currentItem()
        # selected_row = self.profile_options.currentRow()
        assert isinstance(selected_item, StatisticListWidgetItem)
        node = deepcopy(selected_item.stat)
        # noinspection PyTypeChecker
        node = self.get_parameters(node, self.statistic_area_choose.get_value(), self.per_component.get_value(),
                                   self.power_num.value())
        if node is None:
            return
        lw = StatisticListWidgetItem(node)
        for i in range(self.profile_options_chosen.count()):
            if lw.text() == self.profile_options_chosen.item(i).text():
                return
        lw.setToolTip(selected_item.toolTip())
        self.profile_options_chosen.addItem(lw)
        if self.good_name():
            self.save_butt.setEnabled(True)
            self.save_butt_with_name.setEnabled(True)
        if self.profile_options.count() == 0:
            self.choose_butt.setDisabled(True)

    def discard_option(self):
        selected_item: StatisticListWidgetItem = self.profile_options_chosen.currentItem()
        #  selected_row = self.profile_options_chosen.currentRow()
        lw = StatisticListWidgetItem(deepcopy(selected_item.stat))
        lw.setToolTip(selected_item.toolTip())
        self.create_selection_chosen_changed()
        for i in range(self.profile_options.count()):
            if lw.text() == self.profile_options.item(i).text():
                return
        self.profile_options.addItem(lw)

    def edit_profile(self):
        item = self.profile_list.currentItem()
        if item is None:
            return
        profile = self.settings.statistic_profiles[str(item.text())]
        self.profile_options_chosen.clear()
        self.profile_name.setText(item.text())
        for ch in profile.chosen_fields:
            self.profile_options_chosen.addItem(StatisticListWidgetItem(ch.calculation_tree))
        # self.gauss_img.setChecked(profile.use_gauss_image)
        self.save_butt.setEnabled(True)
        self.save_butt_with_name.setEnabled(True)

    def save_action(self):
        for i in range(self.profile_list.count()):
            if self.profile_name.text() == self.profile_list.item(i).text():
                ret = QMessageBox.warning(self, "Profile exist", "Profile exist\nWould you like to overwrite it?",
                                          QMessageBox.No | QMessageBox.Yes)
                if ret == QMessageBox.No:
                    return
        selected_values = []
        for i in range(self.profile_options_chosen.count()):
            element: StatisticListWidgetItem = self.profile_options_chosen.item(i)
            selected_values.append(StatisticEntry(element.text(), element.stat))
        stat_prof = StatisticProfile(self.profile_name.text(), selected_values)
        if stat_prof.name not in self.settings.statistic_profiles:
            self.profile_list.addItem(stat_prof.name)
        self.settings.statistic_profiles[stat_prof.name] = stat_prof
        self.settings.dump()
        self.export_profiles_butt.setEnabled(True)

    def named_save_action(self):
        for i in range(self.profile_list.count()):
            if self.profile_name.text() == self.profile_list.item(i).text():
                ret = QMessageBox.warning(self, "Profile exist", "Profile exist\nWould you like to overwrite it?",
                                          QMessageBox.No | QMessageBox.Yes)
                if ret == QMessageBox.No:
                    return
        selected_values = []
        for i in range(self.profile_options_chosen.count()):
            txt = str(self.profile_options_chosen.item(i).text())
            selected_values.append((txt, str, txt))
        val_dialog = MultipleInput("Set fields name", list(selected_values))
        if val_dialog.exec_():
            selected_values = []
            for i in range(self.profile_options_chosen.count()):
                element: StatisticListWidgetItem = self.profile_options_chosen.item(i)
                selected_values.append(StatisticEntry(val_dialog.result[element.text()], element.stat))
            stat_prof = StatisticProfile(self.profile_name.text(), selected_values)
            if stat_prof.name not in self.settings.statistic_profiles:
                self.profile_list.addItem(stat_prof.name)
            self.settings.statistic_profiles[stat_prof.name] = stat_prof
            self.export_profiles_butt.setEnabled(True)

    def reset_action(self):
        self.profile_options.clear()
        self.profile_options_chosen.clear()
        self.profile_name.setText("")
        self.save_butt.setDisabled(True)
        self.save_butt_with_name.setDisabled(True)
        self.move_down.setDisabled(True)
        self.move_up.setDisabled(True)
        self.proportion_butt.setDisabled(True)
        self.choose_butt.setDisabled(True)
        self.discard_butt.setDisabled(True)
        for name, profile in STATISTIC_DICT.items():
            help_text = profile.get_description()
            lw = StatisticListWidgetItem(profile.get_starting_leaf())
            lw.setToolTip(help_text)
            self.profile_options.addItem(lw)

    def soft_reset(self):
        # TODO rim should not be removed
        shift = 0
        for i in range(self.profile_options.count()):
            item = self.profile_options.item(i - shift)
            if str(item.text()) not in STATISTIC_DICT:
                self.profile_options.takeItem(i - shift)
                shift += 1
        self.create_selection_changed()

    def export_statistic_profiles(self):
        exp = ExportDialog(self.settings.statistic_profiles, StringViewer)
        if not exp.exec_():
            return
        dial = QFileDialog(self, "Export settings profiles")
        dial.setDirectory(self.settings.get("io.export_directory", ""))
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setNameFilter("statistic profile (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("measurements_profile.json")

        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.set("io.export_directory", file_path)
            data = dict([(x, self.settings.statistic_profiles[x]) for x in exp.get_export_list()])
            with open(file_path, 'w') as ff:
                json.dump(data, ff, cls=self.settings.json_encoder_class, indent=2)
            self.settings.set("io.save_directory", os.path.dirname(file_path))

    def import_statistic_profiles(self):
        dial = QFileDialog(self, "Import settings profiles")
        dial.setDirectory(self.settings.get("io.export_directory", ""))
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setNameFilter("statistic profile (*.json)")
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.set("io.export_directory", file_path)
            stat, err = self.settings.load_part(file_path)
            if err:
                QMessageBox.warning(self, "Import error", "error during importing, part of data were filtered.")
            statistic_dict = self.settings.statistic_profiles
            imp = ImportDialog(stat, statistic_dict, StringViewer)
            if not imp.exec_():
                return
            for original_name, final_name in imp.get_import_list():
                statistic_dict[final_name] = stat[original_name]
            self.profile_list.clear()
            self.profile_list.addItems(list(sorted(statistic_dict.keys())))
            self.settings.dump()


class AdvancedWindow(QTabWidget):
    """
    :type settings: Settings
    """

    def __init__(self, settings, parent=None):
        super(AdvancedWindow, self).__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings and Measurement")
        self.advanced_settings = AdvancedSettings(settings)
        self.colormap_settings = ColorSelector(settings, ["result_control"])
        self.statistics = StatisticsWidget(settings)
        self.statistics_settings = StatisticsSettings(settings)
        self.addTab(self.advanced_settings, "Properties")
        self.addTab(self.colormap_settings, "Color maps")
        self.addTab(self.statistics_settings, "Measurements settings")
        self.addTab(self.statistics, "Measurements")
        """if settings.advanced_menu_geometry is not None:
            self.restoreGeometry(settings.advanced_menu_geometry)"""
        try:
            geometry = self.settings.get_from_profile("advanced_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, 'ascii')))
        except KeyError:
            pass

    def closeEvent(self, *args, **kwargs):
        self.settings.set_in_profile("advanced_window_geometry", bytes(self.saveGeometry().toHex()).decode('ascii'))
        super(AdvancedWindow, self).closeEvent(*args, **kwargs)


class MultipleInput(QDialog):
    def __init__(self, text, help_text, objects_list=None):
        if objects_list is None:
            objects_list = help_text
            help_text = ""

        def create_input_float(obj, ob2=None):
            if ob2 is not None:
                val = obj
                obj = ob2
            else:
                val = 0
            res = QDoubleSpinBox(obj)
            res.setRange(-1000000, 1000000)
            res.setValue(val)
            return res

        def create_input_int(obj, ob2=None):
            if ob2 is not None:
                val = obj
                obj = ob2
            else:
                val = 0
            res = QSpinBox(obj)
            res.setRange(-1000000, 1000000)
            res.setValue(val)
            return res

        field_dict = {str: QLineEdit, float: create_input_float, int: create_input_int}
        QDialog.__init__(self)
        ok_butt = QPushButton("Ok", self)
        cancel_butt = QPushButton("Cancel", self)
        self.object_dict = dict()
        self.result = None
        ok_butt.clicked.connect(self.accept_response)
        cancel_butt.clicked.connect(self.close)
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignVCenter)
        for i, info in enumerate(objects_list):
            name = info[0]
            type_of = info[1]
            name_label = QLabel(name)
            name_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            layout.addWidget(name_label, i, 0)
            if len(info) == 3:
                item = field_dict[type_of](type_of(info[2]), self)
            else:
                item = field_dict[type_of](self)
            self.object_dict[name] = (type_of, item)
            layout.addWidget(item, i, 1)
        main_layout = QVBoxLayout()
        main_text = QLabel(text)
        main_text.setWordWrap(True)
        font = QApplication.font()
        font.setBold(True)
        main_text.setFont(font)
        main_layout.addWidget(main_text)
        if help_text != "":
            help_label = QLabel(help_text)
            help_label.setWordWrap(True)
            main_layout.addWidget(help_label)
        main_layout.addLayout(layout)
        button_layout = QHBoxLayout()
        button_layout.addWidget(cancel_butt)
        button_layout.addStretch()
        button_layout.addWidget(ok_butt)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def accept_response(self):
        res = dict()
        for name, (type_of, item) in self.object_dict.items():
            if type_of == str:
                val = str(item.text())
                if val.strip() != "":
                    res[name] = val
                else:
                    QMessageBox.warning(self, "Not all fields filled", "")
                    return
            else:
                val = type_of(item.value())
                res[name] = val
        self.result = res
        self.accept()

    @property
    def get_response(self):
        return self.result
