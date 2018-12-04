import logging
import os
import numpy as np
from PyQt5.QtCore import QByteArray, Qt, QEvent
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QTabWidget, QWidget, QListWidget, QTextEdit, QPushButton, QCheckBox, QLineEdit, QVBoxLayout, \
    QLabel, QHBoxLayout, QListWidgetItem, QDialog, QDoubleSpinBox, QSpinBox, QGridLayout, QApplication, QMessageBox, \
    QFileDialog, QComboBox, QTableWidget, QTableWidgetItem, QAbstractSpinBox, QInputDialog, QPlainTextEdit, QFrame

from common_gui.colors_choose import ColorSelector
from common_gui.lock_checkbox import LockCheckBox
from partseg2.partseg_settings import PartSettings, MASK_COLORS
from partseg2.profile_export import ExportDialog, StringViewer, ImportDialog, ProfileDictViewer
from partseg2.statistics_calculation import StatisticProfile
from project_utils.global_settings import static_file_folder
from project_utils.segmentation.denoising import noise_removal_dict
from project_utils.settings import BaseSettings
from project_utils.universal_const import UNITS_DICT, UNIT_SCALE, UNITS_LIST
from common_gui.dim_combobox import DimComboBox

def h_line():
    toto = QFrame()
    toto.setFrameShape(QFrame.HLine)
    toto.setFrameShadow(QFrame.Sunken)
    return toto


class AdvancedSettings(QWidget):
    def __init__(self, settings: PartSettings):
        super().__init__()
        self._settings = settings
        self.read_spacing_chk = QCheckBox("Read voxel size from file")
        self.export_btn = QPushButton("Export profile")
        self.export_btn.clicked.connect(self.export_profile)
        self.import_btn = QPushButton("Import profile")
        self.import_btn.clicked.connect(self.import_profiles)
        self.delete_btn = QPushButton("Delete profile")
        self.delete_btn.setDisabled(True)
        self.delete_btn.clicked.connect(self.delete_profile)
        self.rename_btn = QPushButton("Rename Profile")
        self.rename_btn.clicked.connect(self.rename_profile)
        self.rename_btn.setDisabled(True)
        self.use_physical_unit_chk = QCheckBox()
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
        self.spacing[2].valueChanged.connect(self.synchronize_spacing)
        units_index = self._settings.get("units_index", 2)
        for i, el in enumerate(self.spacing):
            el.setAlignment(Qt.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.NoButtons)
            el.setRange(0, 1000000)
            el.valueChanged.connect(self.image_spacing_change)
        self.units = QComboBox()
        self.units.addItems(UNITS_LIST)
        self.units.setCurrentIndex(units_index)
        self.units.currentIndexChanged.connect(self.update_spacing)

        color, opacity = self._settings.get_from_profile("mask_presentation", (list(MASK_COLORS.keys())[0], 1))
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
        self.mask_opacity.valueChanged.connect(self.mask_prop_changed)
        self.mask_color.currentIndexChanged.connect(self.mask_prop_changed)

        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(self.lock_spacing)
        for txt, el in zip(["x", "y", "z"], self.spacing[::-1]):
            spacing_layout.addWidget(QLabel(txt+":"))
            spacing_layout.addWidget(el)
        spacing_layout.addWidget(self.units)
        spacing_layout.addWidget(self.read_spacing_chk)
        spacing_layout.addStretch(1)
        voxel_size_layout = QHBoxLayout()
        voxel_size_layout.addWidget(self.voxel_size_label)
        voxel_size_layout.addSpacing(30)
        voxel_size_layout.addWidget(QLabel("Use physical units in minimum size:"))
        voxel_size_layout.addWidget(self.use_physical_unit_chk)
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
        profile_layout.addWidget(self.pipeline_list, 3, 0, 3, 1)
        profile_layout.addWidget(self.info_label, 1, 1, 3, 2)
        profile_layout.addWidget(self.export_btn, 4, 1)
        profile_layout.addWidget(self.import_btn, 4, 2)
        profile_layout.addWidget(self.delete_btn, 5, 1)
        profile_layout.addWidget(self.rename_btn, 5, 2)
        layout = QVBoxLayout()
        layout.addLayout(spacing_layout)
        layout.addLayout(voxel_size_layout)
        layout.addLayout(mask_layout)

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
        self.info_label.setPlainText(str(profile))
        self.delete_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)

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
        self.voxel_size_label.setText(f"Voxel_size: {voxel_size} {UNITS_LIST[self.units.currentIndex()]}"
                                      f"<sup>{len(self._settings.image_spacing)}</sup>")


    def update_spacing(self, index=None):
        voxel_size = 1
        if index is not None:
            self._settings.set("units_index", index)
        for i, (el, sp) in enumerate(zip(self.spacing[::-1], self._settings.image_spacing[::-1])):
            el.blockSignals(True)
            current_size = sp * UNIT_SCALE[self.units.currentIndex()]
            voxel_size *= current_size
            el.setValue(current_size)
            el.blockSignals(False)
        self.spacing[0].setDisabled(len(self._settings.image_spacing) == 2)
        self.voxel_size_label.setText(f"Voxel_size: {voxel_size} {UNITS_LIST[self.units.currentIndex()]}"
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
        exp = ExportDialog( self._settings.get(f"segmentation_profiles", dict()), ProfileDictViewer)
        if not exp.exec_():
            return
        dial = QFileDialog(self, "Export profile segment")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setDirectory(self._settings.get("io.save_directory", ""))
        dial.setNameFilter("Segment profile (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("segment_profile.json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self._settings.dump_part(file_path, "segmentation_profiles", exp.get_export_list())
            self._settings.set("io.save_directory", os.path.dirname(file_path))

    def import_profiles(self):
        dial = QFileDialog(self, "Import profile segment")
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setDirectory(self._settings.get("io.save_directory", ""))
        dial.setNameFilter("Segment profile (*.json)")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self._settings.set("io.save_directory", os.path.dirname(file_path))
            profs = self._settings.load_part(file_path)
            profiles_dict = self._settings.get(f"segmentation_profiles", dict())
            imp = ImportDialog(profs, profiles_dict, ProfileDictViewer)
            if not imp.exec_():
                return
            for original_name, final_name in imp.get_import_list():
                profiles_dict[final_name] = profs[original_name]
            self._settings.dump()
            self.update_profile_list()

    def rename_profile(self):
        profile_name = self.profile_list.currentItem().text()
        text, ok = QInputDialog.getText(self, "New profile name", f"New name for {profile_name}", text=profile_name)
        if ok:
            text = text.strip()
            profiles_dict = self._settings.get(f"segmentation_profiles", dict())
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



class StatisticsSettings(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings: PartSettings):
        super(StatisticsSettings, self).__init__()
        self.chosen_element = None
        self.settings = settings
        self.profile_list = QListWidget(self)
        self.profile_description = QTextEdit(self)
        self.profile_description.setReadOnly(True)
        self.profile_options = QListWidget()
        self.profile_options_chosen = QListWidget()
        self.choose_butt = QPushButton(u"→", self)
        self.discard_butt = QPushButton(u"←", self)
        self.proportion_butt = QPushButton(u"∺", self)
        self.proportion_butt.setToolTip("Create proportion from two statistics")
        self.move_up = QPushButton(u"↑", self)
        self.move_down = QPushButton(u"↓", self)
        self.save_butt = QPushButton("Save statistic profile")
        self.save_butt.setToolTip("Set name for profile and choose at least one statistic")
        self.save_butt_with_name = QPushButton("Save statistic profile with name")
        self.save_butt_with_name.setToolTip("Set name for profile and choose at least one statistic")
        self.reset_butt = QPushButton("Clear")
        self.soft_reset_butt = QPushButton("Remove user statistics")
        self.profile_name = QLineEdit(self)
        self.reversed_brightness = QCheckBox("Reversed image", self)
        self.reversed_brightness.setToolTip("This is option usefull for electrom microscope images")
        self.gauss_img = DimComboBox(self)
        self.gauss_radius = QDoubleSpinBox(self)# QCheckBox("2d gauss image", self)
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
        self.proportion_butt.clicked.connect(self.choose_element)
        self.save_butt.setDisabled(True)
        self.save_butt.clicked.connect(self.save_action)
        self.save_butt_with_name.setDisabled(True)
        self.save_butt_with_name.clicked.connect(self.named_save_action)
        self.profile_name.textChanged.connect(self.name_changed)
        self.move_down.setDisabled(True)
        self.move_down.clicked.connect(self.move_down_fun)
        self.move_up.setDisabled(True)
        self.move_up.clicked.connect(self.move_up_fun)
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
        layout.addWidget(QLabel("Defined statistics profiles list:"))
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
        """name_layout.addWidget(self.reversed_brightness)
        name_layout.addWidget(QLabel("Gauss image:"))
        name_layout.addWidget(self.gauss_img)
        name_layout.addWidget(QLabel("Gauss radius (pix):"))
        name_layout.addWidget(self.gauss_radius)"""
        self.gauss_radius.setHidden(True)
        self.gauss_img.setHidden(True)
        self.reversed_brightness.setHidden(True)
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

        for name, profile in sorted(StatisticProfile.STATISTIC_DICT.items()):
            help_text = profile.help_message
            lw = QListWidgetItem(name)
            lw.setToolTip(help_text)
            self.profile_options.addItem(lw)
        self.profile_list.addItems(list(sorted(self.settings.statistic_profiles.keys())))
        if self.profile_list.count() == 0:
            self.export_profiles_butt.setDisabled(True)

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

    def choose_element(self):
        if self.chosen_element is None:
            item = self.profile_options.currentItem()
            item.setIcon(QIcon(os.path.join(static_file_folder, "icons", "task-accepted.png")))
            self.chosen_element = item
        elif self.profile_options.currentItem() == self.chosen_element:
            self.chosen_element.setIcon(QIcon())
            self.chosen_element = None
        else:
            text1 = self.chosen_element.text()
            if "/" in text1:
                text1 = "({})".format(text1)
            text2 = self.profile_options.currentItem().text()
            if "/" in text2:
                text2 = "({})".format(text2)
            lw = QListWidgetItem("{}/{}".format(text1, text2))
            lw.setToolTip("User defined")
            self.profile_options_chosen.addItem(lw)
            self.chosen_element.setIcon(QIcon())
            self.chosen_element = None

    def create_selection_chosen_changed(self):
        # print(self.profile_options_chosen.count())
        if self.profile_options_chosen.count() == 0:
            self.move_down.setDisabled(True)
            self.move_up.setDisabled(True)
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

    def choose_option(self):
        selected_item = self.profile_options.currentItem()
        # selected_row = self.profile_options.currentRow()
        if str(selected_item.text()) in StatisticProfile.STATISTIC_DICT:
            arguments = StatisticProfile.STATISTIC_DICT[str(selected_item.text())].arguments
        else:
            arguments = None
        if arguments is not None:
            val_dialog = MultipleInput("Set parameters:",
                                       StatisticProfile.STATISTIC_DICT[str(selected_item.text())].help_message,
                                       list(arguments.items()))
            if val_dialog.exec_():
                res = ""
                for name, val in val_dialog.get_response.items():
                    res += "{}={},".format(name, val)
                lw = QListWidgetItem(selected_item.text() + "[{}]".format(res[:-1]))
            else:
                return
        else:
            lw = QListWidgetItem(selected_item.text())
            # self.profile_options.takeItem(selected_row)
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
        selected_item = self.profile_options_chosen.currentItem()
        selected_row = self.profile_options_chosen.currentRow()
        lw = QListWidgetItem(selected_item.text())
        lw.setToolTip(selected_item.toolTip())
        self.profile_options_chosen.takeItem(selected_row)
        if self.profile_options_chosen.count() == 0:
            self.save_butt.setDisabled(True)
            self.save_butt_with_name.setDisabled(True)
            self.discard_butt.setDisabled(True)
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
            self.profile_options_chosen.addItem(profile.flat_tree(ch[0]))
        # self.gauss_img.setChecked(profile.use_gauss_image)
        self.reversed_brightness.setChecked(profile.reversed_brightness)
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
            txt = str(self.profile_options_chosen.item(i).text())
            selected_values.append((txt, txt))
        stat_prof = StatisticProfile(str(self.profile_name.text()), selected_values,
                                     self.reversed_brightness.isChecked(),
                                     False)
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
                txt = str(self.profile_options_chosen.item(i).text())
                selected_values.append((txt, val_dialog.result[txt]))
            stat_prof = StatisticProfile(str(self.profile_name.text()), selected_values,
                                         self.reversed_brightness.isChecked(),
                                         (self.gauss_img.value() ,self.gauss_radius.value()))
            if stat_prof.name not in self.settings.statistic_profiles:
                self.profile_list.addItem(stat_prof.name)
            self.self.settings.statistic_profiles[stat_prof.name] = stat_prof
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
        self.profile_options.addItems(list(sorted(StatisticProfile.STATISTIC_DICT.keys())))

    def soft_reset(self):
        shift = 0
        for i in range(self.profile_options.count()):
            item = self.profile_options.item(i - shift)
            if str(item.text()) not in StatisticProfile.STATISTIC_DICT:
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
        dial.selectFile("statistic_profile.json")

        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.set("io.export_directory", file_path)
            self.settings.dump_part(file_path, "statistic_profiles", exp.get_export_list())

    def import_statistic_profiles(self):
        dial = QFileDialog(self, "Import settings profiles")
        dial.setDirectory(self.settings.get("io.export_directory", ""))
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setNameFilter("statistic profile (*.json)")
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.set("io.export_directory", file_path)
            stat = self.settings.load_part(file_path)
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
        self.setWindowTitle("Settings and statistics")
        self.advanced_settings = AdvancedSettings(settings)
        self.colormap_settings = ColorSelector(settings, ["result_control"])
        self.statistics = StatisticsWindow(settings)
        self.statistics_settings = StatisticsSettings(settings)
        self.addTab(self.advanced_settings, "Settings")
        self.addTab(self.colormap_settings, "Color maps")
        self.addTab(self.statistics, "Statistics")
        self.addTab(self.statistics_settings, "Statistic settings")
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

class StatisticsWindow(QWidget):
    """
    :type settings: Settings
    :type segment: Segment
    """
    def __init__(self, settings: PartSettings, segment=None):
        super(StatisticsWindow, self).__init__()
        self.settings = settings
        self.segment = segment
        self.recalculate_button = QPushButton("Recalculate and\n replace statistics", self)
        self.recalculate_button.clicked.connect(self.replace_statistics)
        self.recalculate_append_button = QPushButton("Recalculate and\n append statistics", self)
        self.recalculate_append_button.clicked.connect(self.append_statistics)
        self.copy_button = QPushButton("Copy to clipboard", self)
        self.horizontal_statistics = QCheckBox("Horizontal", self)
        self.no_header = QCheckBox("No header", self)
        self.no_units = QCheckBox("No units", self)
        self.no_units.setChecked(True)
        self.horizontal_statistics.stateChanged.connect(self.horizontal_changed)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.statistic_type = QComboBox(self)
        # self.statistic_type.addItem("Emish statistics (oryginal)")
        self.statistic_type.currentIndexChanged[str].connect(self.statistic_selection_changed)
        self.statistic_type.addItem("<none>")
        self.statistic_type.addItems(list(sorted(self.settings.statistic_profiles.keys())))
        self.statistic_type.setToolTip(
            "You can create new statistic profile in advanced window, in tab \"Statistic settings\"")
        self.channels_chose = QComboBox()
        self.channels_chose.addItems(map(str, range(self.settings.channels)))
        #self.noise_removal_method = QComboBox()
        #self.noise_removal_method.addItem("Like segmentation")
        #self.noise_removal_method.addItems(noise_removal_dict.keys())
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
        # butt_layout2.addWidget(QLabel("Noise removal:"))
        # butt_layout2.addWidget(self.noise_removal_method)
        butt_layout2.addWidget(QLabel("Statistic:"))
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

    def image_changed(self, channels_num):
        ind = self.channels_chose.currentIndex()
        self.channels_chose.clear()
        self.channels_chose.addItems(map(str, range(channels_num)))
        if ind < 0 or ind > channels_num:
            ind = 0
        self.channels_chose.setCurrentIndex(ind)

    def statistic_selection_changed(self, text):
        text = str(text)
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
                self.info_field.setItem(y, x,  QTableWidgetItem(ob_array[x, y]))

    def append_statistics(self):
        try:
            compute_class = self.settings.statistic_profiles[self.statistic_type.currentText()]
        except KeyError:
            QMessageBox.warning(self, "Statistic profile not found",
                                f"Statistic profile '{self.statistic_type.currentText()}' not found'")
            return
        gauss_image = self.settings.noise_remove_image_part
        image = self.settings.image.get_channel(self.channels_chose.currentIndex())
        segmentation = self.settings.segmentation
        full_mask = self.settings.full_segmentation
        base_mask = self.settings.mask
        try:
            stat = compute_class.calculate(image, gauss_image, segmentation,
                                           full_mask, base_mask, self.settings.image_spacing)
        except ValueError as e:
            logging.error(e)
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
            if not self.no_header.isChecked()  and (self.previous_profile != compute_class.name):
                ver_headers.append("Name")
            ver_headers.extend(["Value"])
            if not self.no_units.isChecked():
                ver_headers.append("Units")
            self.info_field.setVerticalHeaderLabels(ver_headers)
            self.info_field.setHorizontalHeaderLabels([str(x) for x in range(len(stat))])
            for i, (key, val) in enumerate(stat.items()):
                print(i, key, val)
                if not self.no_header.isChecked() and (self.previous_profile != compute_class.name):
                    self.info_field.setItem(self.statistic_shift + 0, i, QTableWidgetItem(key))
                self.info_field.setItem(self.statistic_shift + 1, i, QTableWidgetItem(str(val)))
                if not self.no_units.isChecked():
                    try:
                        self.info_field.setItem(self.statistic_shift + 2, i,
                                                QTableWidgetItem(UNITS_DICT[key].format(self.settings.size_unit)))
                    except KeyError as k:
                        logging.warning(k.message)
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
            for i, (key, val) in enumerate(stat.items()):
                # print(i, key, val)
                if not self.no_header.isChecked() and (self.previous_profile != compute_class.name):
                    self.info_field.setItem(i, self.statistic_shift + 0, QTableWidgetItem(key))
                self.info_field.setItem(i, self.statistic_shift + 1, QTableWidgetItem(str(val)))
                if not self.no_units.isChecked():
                    try:
                        self.info_field.setItem(i, self.statistic_shift + 2,
                                                QTableWidgetItem(UNITS_DICT[key].format(self.settings.size_unit)))
                    except KeyError as k:
                        logging.warning(k.message)
        if self.no_units.isChecked():
            self.statistic_shift -= 1
        self.statistic_shift += 3
        self.previous_profile = compute_class.name

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
