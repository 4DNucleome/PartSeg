import os

from PyQt5.QtCore import QByteArray, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QTabWidget, QWidget, QListWidget, QTextEdit, QPushButton, QCheckBox, QLineEdit, QVBoxLayout, \
    QLabel, QHBoxLayout, QListWidgetItem, QDialog, QDoubleSpinBox, QSpinBox, QGridLayout, QApplication, QMessageBox, \
    QFileDialog

from common_gui.colors_choose import ColorSelector
from partseg.statistics_calculation import StatisticProfile
from project_utils.global_settings import static_file_folder
from qt_import import h_line


class AdvancedSettings(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings


class StatisticsWindow(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings


"""class StatisticsSettings(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings"""

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
        #self.profile_list.addItems(list(sorted([x[0] for x in self.settings.statistics_profile_dict.items()])))
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
        if stat_prof.name not in self.settings.get("statistic_profiles", dict()):
            self.profile_list.addItem(stat_prof.name)
        self.settings.set_in_profile(f"statistic_profiles.{stat_prof.name}", stat_prof)
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
        dial.setFilter("statistic profile (*.json)")
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
        dial.setFilter("statistic profile (*.json)")
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



class AdvancedWindow(QTabWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings, parent=None):
        super(AdvancedWindow, self).__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings and statistics")
        self.advanced_settings = AdvancedSettings(settings)
        self.colormap_settings = ColorSelector(settings, ["raw_control", "result_control"])
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