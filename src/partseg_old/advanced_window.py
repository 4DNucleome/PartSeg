# coding=utf-8

import numpy as np
import os

from partseg_old.backend import StatisticProfile, get_segmented_data, calculate_statistic_from_image, \
    SegmentationProfile, Settings
from partseg_old.profile_export import ExportDialog, ImportDialog, StringViewer
from partseg_utils.global_settings import static_file_folder, button_height
from partseg_utils.universal_const import UNITS_DICT, UNITS_LIST
from partseg_old.qt_import import *

__author__ = "Grzegorz Bokota"


class AdvancedWindow(QTabWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings, segment, parent=None):
        super(AdvancedWindow, self).__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings and statistics")
        self.advanced_settings = AdvancedSettings(settings)
        self.colormap_settings = ColormapSettings(settings)
        self.statistics = StatisticsWindow(settings, segment)
        self.statistics_settings = StatisticsSettings(settings)
        self.addTab(self.advanced_settings, "Settings")
        self.addTab(self.colormap_settings, "Color maps")
        self.addTab(self.statistics, "Statistics")
        self.addTab(self.statistics_settings, "Statistic settings")
        if settings.advanced_menu_geometry is not None:
            self.restoreGeometry(settings.advanced_menu_geometry)

    def resizeEvent(self, resize_event):
        super(AdvancedWindow, self).resizeEvent(resize_event)
        """:type new_size: QSize"""
        w = resize_event.size().width()
        h = resize_event.size().height()
        ht = self.tabBar().size().height()
        h -= ht
        self.colormap_settings.resize(w, h)
        self.statistics.resize(w, h)
        self.advanced_settings.resize(w, h)

    def closeEvent(self, *args, **kwargs):
        self.colormap_settings.clean()
        self.settings.advanced_menu_geometry = self.saveGeometry()
        super(AdvancedWindow, self).closeEvent(*args, **kwargs)


class StatisticsSettings(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings):
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
        self.reversed_brightness = QCheckBox("Reversed image (for electron microscope)", self)
        self.gauss_img = QCheckBox("2d gauss image", self)
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
        name_layout.addWidget(self.reversed_brightness)
        name_layout.addWidget(self.gauss_img)
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
        self.profile_list.addItems(list(sorted([x[0] for x in self.settings.statistics_profile_dict.items()])))
        if self.profile_list.count() == 0:
            self.export_profiles_butt.setDisabled(True)

    def delete_profile(self):
        row = self.profile_list.currentRow()
        item = self.profile_list.currentItem()
        del self.settings.statistics_profile_dict[str(item.text())]
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
        text = "Profile name: {}\n".format(item.text())
        profile = self.settings.statistics_profile_dict[str(item.text())]  # type: StatisticProfile
        text += "Reversed image [{}]\n".format(profile.reversed_brightness)
        text += "Gaussed image [{}]\n".format(profile.use_gauss_image)
        text += "statistics list:\n"
        for el in profile.chosen_fields:
            if el[2] is not None:
                text += "{}: {}\n".format(el[1], el[2])
            else:
                text += "{}\n".format(el[1])
        self.profile_description.setText(text)

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
        profile = self.settings.statistics_profile_dict[str(item.text())]  # type: StatisticProfile
        self.profile_options_chosen.clear()
        self.profile_name.setText(item.text())
        for ch in profile.chosen_fields:
            self.profile_options_chosen.addItem(profile.flat_tree(ch[0]))
        self.gauss_img.setChecked(profile.use_gauss_image)
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
                                     self.gauss_img.isChecked())
        if stat_prof.name not in self.settings.statistics_profile_dict:
            self.profile_list.addItem(stat_prof.name)
        self.settings.statistics_profile_dict[stat_prof.name] = stat_prof
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
                                         self.gauss_img.isChecked())
            if stat_prof.name not in self.settings.statistics_profile_dict:
                self.profile_list.addItem(stat_prof.name)
            self.settings.statistics_profile_dict[stat_prof.name] = stat_prof
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
        exp = ExportDialog(self.settings.statistics_profile_dict, StringViewer)
        if not exp.exec_():
            return
        dial = QFileDialog(self, "Export settings profiles")
        if self.settings.statistic_dirs is not None:
            dial.setDirectory(self.settings.statistic_dirs)
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setNameFilter("statistic profile (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("statistic_profile.json")

        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.statistic_dirs = file_path
            self.settings.dump_statistics(file_path, exp.get_export_list())

    def import_statistic_profiles(self):
        dial = QFileDialog(self, "Import settings profiles")
        if self.settings.statistic_dirs is not None:
            dial.setDirectory(self.settings.statistic_dirs)
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setNameFilter("statistic profile (*.json)")
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.statistic_dirs = file_path
            stat = self.settings.load_statistics(file_path)
            imp = ImportDialog(stat, self.settings.statistics_profile_dict, StringViewer)
            if not imp.exec_():
                return
            self.settings.add_statistics(stat, imp.get_import_list())
            self.profile_list.clear()
            self.profile_list.addItems(list(sorted(self.settings.statistics_profile_dict.keys())))


class StatisticsWindow(QWidget):
    """
    :type settings: Settings
    :type segment: Segment
    """
    def __init__(self, settings, segment):
        super(StatisticsWindow, self).__init__()
        self.settings = settings
        self.segment = segment
        self.recalculate_button = QPushButton("Recalculate and replace statistics", self)
        self.recalculate_button.clicked.connect(self.replace_statistics)
        self.recalculate_append_button = QPushButton("Recalculate and append statistics", self)
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
        self.statistic_type.addItems(list(sorted(self.settings.statistics_profile_dict.keys())))
        self.statistic_type.currentIndexChanged[str_type].connect(self.statistic_selection_changed)
        self.info_field = QTableWidget(self)
        self.info_field.setColumnCount(3)
        self.info_field.setHorizontalHeaderLabels(["Name", "Value", "Units"])
        self.statistic_shift = 0
        self._protect = False
        layout = QVBoxLayout()
        # layout.addWidget(self.recalculate_button)
        v_butt_layout = QVBoxLayout()
        v_butt_layout.setSpacing(1)
        up_butt_layout = QHBoxLayout()
        up_butt_layout.addWidget(self.recalculate_button)
        up_butt_layout.addWidget(self.recalculate_append_button)
        butt_layout = QHBoxLayout()
        # butt_layout.setMargin(0)
        butt_layout.setSpacing(10)
        butt_layout.addWidget(self.horizontal_statistics, 1)
        butt_layout.addWidget(self.no_header, 1)
        butt_layout.addWidget(self.no_units, 1)
        butt_layout.addWidget(self.copy_button, 2)
        butt_layout.addWidget(self.statistic_type, 2)
        v_butt_layout.addLayout(up_butt_layout)
        v_butt_layout.addLayout(butt_layout)
        layout.addLayout(v_butt_layout)
        # layout.addLayout(butt_layout)
        layout.addWidget(self.info_field)
        self.setLayout(layout)
        # noinspection PyArgumentList
        self.clip = QApplication.clipboard()
        # self.update_statistics()

    def statistic_selection_changed(self, text):
        if self._protect:
            return
        text = str(text)
        try:
            stat = self.settings.statistics_profile_dict[text]
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
        if self.statistic_type.currentText() == "Emish statistics (oryginal)":
            image, mask = get_segmented_data(np.copy(self.settings.image), self.settings, self.segment)
            stat = calculate_statistic_from_image(image, mask, self.settings)

        else:
            compute_class = self.settings.statistics_profile_dict[str(self.statistic_type.currentText())]
            gauss_image = self.settings.gauss_image
            image = self.settings.image
            mask = self.segment.get_segmentation()
            full_mask = self.segment.get_full_segmentation()
            base_mask = self.settings.mask
            try:
                stat = compute_class.calculate(image, gauss_image, mask,
                                               full_mask, base_mask, self.settings.voxel_size)
            except ValueError as e:
                logging.error(e)
        if self.no_header.isChecked():
            self.statistic_shift -= 1
        if self.no_units.isChecked():
            header_grow = self.statistic_shift - 1
        else:
            header_grow = self.statistic_shift
        if self.horizontal_statistics.isChecked():
            ver_headers = [self.info_field.verticalHeaderItem(x).text() for x in range(self.info_field.rowCount())]
            self.info_field.setRowCount(3 + header_grow)
            self.info_field.setColumnCount(max(len(stat), self.info_field.columnCount()))
            if not self.no_header.isChecked():
                ver_headers.append("Name")
            ver_headers.extend(["Value"])
            if not self.no_units.isChecked():
                ver_headers.append("Units")
            self.info_field.setVerticalHeaderLabels(ver_headers)
            self.info_field.setHorizontalHeaderLabels([str(x) for x in range(len(stat))])
            for i, (key, val) in enumerate(stat.items()):
                print(i, key, val)
                if not self.no_header.isChecked():
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
            if not self.no_header.isChecked():
                hor_headers.append("Name")
            hor_headers.extend(["Value"])
            if not self.no_units.isChecked():
                hor_headers.append("Units")
            self.info_field.setHorizontalHeaderLabels(hor_headers)
            for i, (key, val) in enumerate(stat.items()):
                # print(i, key, val)
                if not self.no_header.isChecked():
                    self.info_field.setItem(i, self.statistic_shift + 0, QTableWidgetItem(key))
                self.info_field.setItem(i, self.statistic_shift + 1, QTableWidgetItem(str(val)))
                if not self.no_units.isChecked():
                    try:
                        key_f = key.split("[")[0]
                        self.info_field.setItem(i, self.statistic_shift + 2,
                                                QTableWidgetItem(UNITS_DICT[key_f].format(self.settings.size_unit)))
                    except KeyError as k:
                        logging.warning(k)
        if self.no_units.isChecked():
            self.statistic_shift -= 1
        self.statistic_shift += 3

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

    def showEvent(self, _):
        self._protect = True
        avali = list(sorted(self.settings.statistics_profile_dict.keys()))
        # avali.insert(0, "Emish statistics (oryginal)")
        text = self.statistic_type.currentText()
        try:
            index = avali.index(text)
        except ValueError:
            index = 0
        self.statistic_type.clear()
        self.statistic_type.addItems(avali)
        self.statistic_type.setCurrentIndex(index)
        self._protect = False


class AdvancedSettings(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings, parent=None):
        super(AdvancedSettings, self).__init__(parent)

        def add_label(label_text, up_layout, widget):
            lab = QLabel(label_text)
            layout = QHBoxLayout()
            layout.setSpacing(0)
            layout.addWidget(lab)
            layout.addWidget(widget)
            up_layout.addLayout(layout)
            return widget

        """def create_spacing(label_text, layout, num):
            spacing = QSpinBox()
            spacing.setRange(0, 100)
            spacing.setValue(settings.spacing[num])
            spacing.setSingleStep(1)
            spacing.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spacing.setAlignment(Qt.AlignRight)
            return add_label(label_text, layout, spacing)"""

        def create_voxel_size(label_text, layout, num):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 1000)
            spinbox.setValue(settings.voxel_size[num])
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(2)
            spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spinbox.setAlignment(Qt.AlignRight)
            return add_label(label_text, layout, spinbox)

        def create_overlay(label_text, layout, val):
            overlay = QDoubleSpinBox()
            overlay.setRange(0, 1)
            overlay.setSingleStep(0.1)
            overlay.setDecimals(2)
            overlay.setButtonSymbols(QAbstractSpinBox.NoButtons)
            overlay.setValue(val)
            return add_label(label_text, layout, overlay)

        def create_power_norm(label_text, layout, val):
            overlay = QDoubleSpinBox()
            overlay.setRange(0.01, 10)
            overlay.setSingleStep(0.1)
            overlay.setDecimals(2)
            overlay.setButtonSymbols(QAbstractSpinBox.NoButtons)
            overlay.setValue(val)
            return add_label(label_text, layout, overlay)

        self.settings = settings
        vertical_layout = QVBoxLayout()
        """spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing"))
        spacing_layout.addSpacing(11)
        self.x_spacing = create_spacing("x:", spacing_layout, 0)
        self.y_spacing = create_spacing("y:", spacing_layout, 1)
        self.z_spacing = create_spacing("z:", spacing_layout, 2)
        spacing_layout.addStretch()
        vertical_layout.addLayout(spacing_layout)"""

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Voxel size"))
        self.x_size = create_voxel_size("x:", size_layout, 0)
        self.y_size = create_voxel_size("y:", size_layout, 1)
        self.z_size = create_voxel_size("z:", size_layout, 2)
        self.units_size = QComboBox()
        self.units_size.addItems(UNITS_LIST)
        self.units_size.setCurrentIndex(2)
        for el in [self.x_size, self.y_size, self.z_size]:
            el.valueChanged.connect(self.update_volume)
        self.units_size.currentIndexChanged.connect(self.update_volume)
        size_layout.addWidget(self.units_size)
        size_layout.addStretch()
        vertical_layout.addLayout(size_layout)
        self.volume_info = QLabel()
        vertical_layout.addWidget(self.volume_info)
        overlay_layout = QHBoxLayout()
        self.mask_overlay = create_overlay("mask opacity:", overlay_layout, self.settings.mask_overlay)
        self.component_overlay = create_overlay("segmentation opacity:", overlay_layout, self.settings.overlay)
        self.power_norm = create_power_norm("norm parameter:", overlay_layout, self.settings.power_norm)
        overlay_layout.addStretch()
        gauss_layout = QHBoxLayout()
        self.gauss_radius = QDoubleSpinBox(self)
        self.gauss_radius.setRange(0.01, 10)
        self.gauss_radius.setValue(settings.gauss_radius)
        self.gauss_radius.setSingleStep(1)
        self.gauss_radius.setButtonSymbols(QAbstractSpinBox.NoButtons)
        gauss_layout.addWidget(QLabel("Gauss radius"))
        gauss_layout.addWidget(self.gauss_radius)
        self.zoom_scale = QDoubleSpinBox(self)
        self.zoom_scale.setRange(0.9, 1.1)
        self.zoom_scale.setSingleStep(0.01)
        self.zoom_scale.setDecimals(3)
        self.zoom_scale.setValue(self.settings.scale_factor)
        gauss_layout.addWidget(QLabel("Zoom scale"))
        gauss_layout.addWidget(self.zoom_scale)
        gauss_layout.addStretch()

        normalize_range = QHBoxLayout()
        self.min_range_value = QDoubleSpinBox(self)
        self.min_range_value.setRange(0, 10**6)
        self.min_range_value.setValue(self.settings.normalize_range[0])
        self.max_range_value = QDoubleSpinBox(self)
        self.max_range_value.setRange(0, 10**6)
        self.max_range_value.setValue(self.settings.normalize_range[1])
        self.use_const_range = QCheckBox(self)
        self.use_const_range.setChecked(self.settings.normalize_range[2])
        normalize_range.addWidget(QLabel("Min brightness"))
        normalize_range.addWidget(self.min_range_value)
        normalize_range.addWidget(QLabel("Max brightness"))
        normalize_range.addWidget(self.max_range_value)
        normalize_range.addWidget(QLabel("Use brightness_range"))
        normalize_range.addWidget(self.use_const_range)

        vertical_layout.addLayout(overlay_layout)
        vertical_layout.addLayout(gauss_layout)
        vertical_layout.addLayout(normalize_range)

        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.accept)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        butt_lay = QHBoxLayout()
        butt_lay.addWidget(accept_button)
        butt_lay.addWidget(reset_button)
        butt_lay.addStretch()

        vertical_layout.addLayout(butt_lay)
        vertical_layout.addWidget(h_line())

        profile_lay = QHBoxLayout()
        self.profile_list = QListWidget()
        profile_lay.addWidget(self.profile_list)
        self.profile_list.addItem("<current profile>")
        self.profile_list.addItems(list(sorted(self.settings.segmentation_profiles_dict.keys())))
        self.profile_list.setMaximumWidth(200)
        self.profile_list.currentTextChanged.connect(self.changed_profile)
        self.create_profile = QPushButton("Create profile", self)
        self.create_profile.clicked.connect(self.save_profile)
        self.delete_profile_butt = QPushButton("Delete profile", self)
        self.delete_profile_butt.setDisabled(True)
        self.delete_profile_butt.clicked.connect(self.delete_profile)
        self.current_profile = QLabel()
        self.current_profile.setWordWrap(True)
        self.export_profile_butt = QPushButton("Export profile")
        self.export_profile_butt.clicked.connect(self.export_profile)
        self.import_profile_butt = QPushButton("Import Profile")
        self.import_profile_butt.clicked.connect(self.import_profiles)
        profile_layout2 = QVBoxLayout()
        profile_lay_butt1 = QHBoxLayout()
        profile_lay_butt1.addWidget(self.create_profile)
        profile_lay_butt1.addWidget(self.delete_profile_butt)
        profile_lay_butt1.addStretch()
        profile_layout2.addLayout(profile_lay_butt1)
        profile_layout2.addWidget(self.current_profile)
        profile_lay_butt2 = QHBoxLayout()
        profile_lay_butt2.addWidget(self.export_profile_butt)
        profile_lay_butt2.addWidget(self.import_profile_butt)
        profile_lay_butt2.addStretch()
        profile_layout2.addLayout(profile_lay_butt2)
        profile_lay.addLayout(profile_layout2)
        text = str(SegmentationProfile("", **self.settings.get_profile_dict()))
        self.current_profile.setText(text)

        vertical_layout.addLayout(profile_lay)
        vertical_layout.addStretch()
        self.setLayout(vertical_layout)
        self.update_volume()

    def normalize_range_change(self):
        if self.use_const_range.isChecked():
            self.settings.normalize_range = (
                self.min_range_value.value(),
                self.max_range_value.value(),
                self.use_const_range.isChecked()
            )
        else:
            self.settings.normalize_range = None

    def export_profile(self):
        exp = ExportDialog(self.settings.segmentation_profiles_dict, StringViewer)
        if not exp.exec_():
            return
        dial = QFileDialog(self, "Export profile segment")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        if self.settings.save_directory is not None:
            dial.setDirectory(self.settings.save_directory)
        dial.setFilter("Segment profile (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("segment_profile.json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self.settings.dump_profiles(file_path, exp.get_export_list())

    def import_profiles(self):
        dial = QFileDialog(self, "Import profile segment")
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        if self.settings.open_directory is not None:
            dial.setDirectory(self.settings.open_directory)
        dial.setFilter("Segment profile (*.json)")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            profs = self.settings.load_profiles(file_path)
            imp = ImportDialog(profs, self.settings.segmentation_profiles_dict, StringViewer)
            if not imp.exec_():
                return
            self.settings.add_profiles(profs, imp.get_import_list())
            self.profile_list.clear()
            self.profile_list.addItems(["<current profile>"] +
                                       list(sorted(self.settings.segmentation_profiles_dict.keys())))

    def changed_profile(self, name):
        name = str(name)
        if name == "<current profile>" or name == u"":
            text = str(SegmentationProfile("", **self.settings.get_profile_dict()))
            self.current_profile.setText(text)
            self.delete_profile_butt.setDisabled(True)
        else:
            text = str(self.settings.get_profile(name))
            self.current_profile.setText(text)
            self.delete_profile_butt.setEnabled(True)

    def save_profile(self):
        # noinspection PyCallByClass
        text, ok = QInputDialog.getText(self, "Profile name", "Profile name", QLineEdit.Normal)
        text = str(text)
        if ok and text != "":
            profile = SegmentationProfile(text, **self.settings.get_profile_dict())
            self.settings.add_profile(profile)
            print("New profile", profile)
            self.profile_list.clear()
            self.profile_list.addItem("<current profile>")
            self.profile_list.addItems(list(sorted(self.settings.segmentation_profiles_dict.keys())))

    def delete_profile(self):
        chosen_profile = self.profile_list.currentItem()
        label = chosen_profile.text()
        if label != "<current profile>":
            self.delete_profile_butt.setDisabled(True)
            self.settings.delete_profile(label)
            self.profile_list.clear()
            self.profile_list.addItem("<current profile>")
            self.profile_list.addItems(list(sorted(self.settings.segmentation_profiles_dict.keys())))

    def update_volume(self):
        volume = self.x_size.value() * self.y_size.value() * self.z_size.value()
        text = u"Voxel size: {}{}³".format(volume, self.units_size.currentText())
        self.volume_info.setText(text)

    def reset(self):
        self.x_spacing.setValue(self.settings.spacing[0])
        self.y_spacing.setValue(self.settings.spacing[1])
        self.z_spacing.setValue(self.settings.spacing[2])
        self.x_size.setValue(self.settings.voxel_size[0])
        self.y_size.setValue(self.settings.voxel_size[1])
        self.z_size.setValue(self.settings.voxel_size[2])
        self.mask_overlay.setValue(self.settings.mask_overlay)
        self.component_overlay.setValue(self.settings.overlay)
        self.power_norm.setValue(self.settings.power_norm)
        self.gauss_radius.setValue(self.settings.gauss_radius)
        self.zoom_scale.setValue(self.settings.scale_factor)

    def accept(self):
        if self.zoom_scale.value() == 1:
            r = QMessageBox.warning(self, "", "Scroll zoom is inactive\nwould you like to save settings?",
                                    QMessageBox.Ok | QMessageBox.Cancel)
            if r == QMessageBox.Cancel:
                return
        self.settings.scale_factor = self.zoom_scale.value()
        # self.settings.spacing = self.x_spacing.value(), self.y_spacing.value(), self.z_spacing.value()
        self.settings.voxel_size = self.x_size.value(), self.y_size.value(), self.z_size.value()
        self.settings.mask_overlay = self.mask_overlay.value()
        self.settings.overlay = self.component_overlay.value()
        self.settings.power_norm = self.power_norm.value()
        self.settings.normalize_range = (
            self.min_range_value.value(),
            self.max_range_value.value(),
            self.use_const_range.isChecked()
        )
        if self.gauss_radius.value() != self.settings.gauss_radius:
            self.settings.gauss_radius = self.gauss_radius.value()
            self.settings.changed_gauss_radius()
        self.settings.advanced_settings_changed()


class ColormapSettings(QWidget):
    """
    :type cmap_list: list[QCheckBox]
    :type settings: Settings
    """

    def __init__(self, settings, parent=None):
        super(ColormapSettings, self).__init__(parent)
        self.accept = QPushButton("Accept", self)
        self.accept.clicked.connect(self.accept_click)
        set_button(self.accept, None)
        self.mark_all = QPushButton("Check all", self)
        self.mark_all.clicked.connect(self.mark_all_click)
        set_button(self.mark_all, self.accept, button_small_dist)
        self.uncheck_all = QPushButton("Uncheck all", self)
        self.uncheck_all.clicked.connect(self.un_mark_all_click)
        set_button(self.uncheck_all, self.mark_all, button_small_dist)
        self.settings = settings
        scroll_area = QScrollArea(self)
        scroll_area.move(0, button_height)
        self.scroll_area = scroll_area
        self.scroll_widget = QLabel()
        self.scroll_area.setWidget(self.scroll_widget)
        chosen = set(settings.colormap_list)
        all_cmap = settings.available_colormap_list
        self.cmap_list = []
        # noinspection PyArgumentList
        font_met = QFontMetrics(QApplication.font())
        max_len = 0
        for name in all_cmap:
            max_len = max(max_len, font_met.boundingRect(name).width())
            check = QCheckBox(self.scroll_widget)
            check.setText(name)
            if name in chosen:
                check.setChecked(True)
            if name == self.settings.color_map_name:
                check.setDisabled(True)
            self.cmap_list.append(check)
        self.columns = 0
        self.label_len = max_len
        self.update_positions()
        self.settings.add_colormap_callback(self.change_main_colormap)
        self.setMinimumSize(400, 400)

    def mark_all_click(self):
        for elem in self.cmap_list:
            if elem.isEnabled():
                elem.setChecked(True)

    def un_mark_all_click(self):
        for elem in self.cmap_list:
            if elem.isEnabled():
                elem.setChecked(False)

    def accept_click(self):
        chosen = []
        for elem in self.cmap_list:
            if elem.isChecked():
                chosen.append(str(elem.text()))
        self.settings.set_available_colormap(chosen)

    def update_positions(self):
        space = self.size().width()
        space -= 20  # scrollbar
        columns = int(space / float(self.label_len + 10))
        if columns == 0:
            columns = 1
        if columns == self.columns:
            return
        self.columns = columns
        elem = self.cmap_list[0]
        elem.move(0, 0)
        prev = elem
        for count, elem in enumerate(self.cmap_list[1:]):
            if ((count + 1) % columns) == 0:
                elem.move(0, prev.pos().y() + 20)
            else:
                elem.move(prev.pos().x() + self.label_len + 10, prev.pos().y())
            prev = elem
        height = prev.pos().y() + 20
        self.scroll_widget.resize(columns * (self.label_len + 10), height)

    def change_main_colormap(self):
        for elem in self.cmap_list:
            elem.setDisabled(False)
            if elem.text() == self.settings.color_map_name:
                elem.setChecked(True)
                elem.setDisabled(True)

    def resizeEvent(self, resize_event):
        w = resize_event.size().width()
        h = resize_event.size().height()
        w -= 4
        h -= button_height + 4
        self.scroll_area.resize(w, h)
        self.update_positions()

    def clean(self):
        self.settings.remove_colormap_callback(self.change_main_colormap)


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
            res.setDecimals(4)
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
                    QMessageBox.warning("Not all fields filled")
                    return
            else:
                val = type_of(item.value())
                res[name] = val
        self.result = res
        self.accept()

    @property
    def get_response(self):
        return self.result

