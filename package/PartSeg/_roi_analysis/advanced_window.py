import json
import os
from contextlib import suppress
from copy import deepcopy
from typing import Optional, Tuple, Union, cast

from qtpy.QtCore import QEvent, Qt, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from superqt import QEnumComboBox, ensure_main_thread

from PartSeg.common_gui.advanced_tabs import AdvancedWindow
from PartSegCore.analysis.algorithm_description import AnalysisAlgorithmSelection
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, Node, PerComponent
from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT, MeasurementProfile
from PartSegCore.io_utils import LoadPlanJson
from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegData import icons_dir

from ..common_backend.base_settings import IO_SAVE_DIRECTORY
from ..common_gui.custom_load_dialog import PLoadDialog
from ..common_gui.custom_save_dialog import FormDialog, PSaveDialog
from ..common_gui.error_report import DataImportErrorDialog
from ..common_gui.lock_checkbox import LockCheckBox
from ..common_gui.searchable_list_widget import SearchableListWidget
from .measurement_widget import MeasurementWidget
from .partseg_settings import PartSettings
from .profile_export import ExportDialog, ImportDialog, ProfileDictViewer, StringViewer


def h_line():
    toto = QFrame()
    toto.setFrameShape(QFrame.HLine)
    toto.setFrameShadow(QFrame.Sunken)
    return toto


class Properties(QWidget):
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
        self.multiple_files_chk = QCheckBox("Show multiple files panel")
        self._update_measurement_chk()
        self.multiple_files_chk.stateChanged.connect(self.multiple_files_visibility)
        self.rename_btn = QPushButton("Rename profile")
        self.rename_btn.clicked.connect(self.rename_profile)
        self.rename_btn.setDisabled(True)
        self.voxel_size_label = QLabel()
        self.info_label = QPlainTextEdit()
        self.info_label.setReadOnly(True)
        self.profile_list = SearchableListWidget()
        self.profile_list.currentTextChanged.connect(self.profile_chosen)
        self.pipeline_list = SearchableListWidget()
        self.pipeline_list.currentTextChanged.connect(self.profile_chosen)
        self.spacing = [QDoubleSpinBox() for _ in range(3)]
        self.lock_spacing = LockCheckBox()
        self.lock_spacing.stateChanged.connect(self.spacing[1].setDisabled)
        self.lock_spacing.stateChanged.connect(self.synchronize_spacing)
        # noinspection PyUnresolvedReferences
        self.spacing[2].valueChanged.connect(self.synchronize_spacing)

        self._settings.roi_profiles_changed.connect(self.update_profile_list)
        self._settings.roi_pipelines_changed.connect(self.update_profile_list)
        self._settings.connect_("multiple_files_widget", self._update_measurement_chk)

        units_value = self._settings.get("units_value", Units.nm)
        for el in self.spacing:
            el.setAlignment(Qt.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.NoButtons)
            el.setRange(0, 1000000)
            # noinspection PyUnresolvedReferences
            el.valueChanged.connect(self.image_spacing_change)
        self.units = QEnumComboBox(enum_class=Units)
        self.units.setCurrentEnum(units_value)
        # noinspection PyUnresolvedReferences
        self.units.currentIndexChanged.connect(self.update_spacing)

        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(self.lock_spacing)
        for txt, el in zip(["x", "y", "z"], self.spacing[::-1]):
            spacing_layout.addWidget(QLabel(f"{txt}:"))
            spacing_layout.addWidget(el)
        spacing_layout.addWidget(self.units)
        spacing_layout.addStretch(1)
        voxel_size_layout = QHBoxLayout()
        voxel_size_layout.addWidget(self.voxel_size_label)
        voxel_size_layout.addSpacing(30)
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
        layout.addWidget(self.multiple_files_chk)

        layout.addLayout(profile_layout, 1)
        self.setLayout(layout)
        self.update_profile_list()

    @Slot(int)
    def multiple_files_visibility(self, val: int):
        self._settings.set("multiple_files_widget", val)

    # @Slot(str)  # PySide bug
    def profile_chosen(self, text):
        if text == "":
            self.delete_btn.setEnabled(False)
            self.rename_btn.setEnabled(False)
            self.info_label.setPlainText("")
            return
        try:
            if self.sender() == self.profile_list.list_widget:
                profile = self._settings.roi_profiles[text]
                self.pipeline_list.selectionModel().clear()
                self.delete_btn.setText("Delete profile")
                self.rename_btn.setText("Rename profile")
            elif self.sender() == self.pipeline_list.list_widget:
                profile = self._settings.roi_pipelines[text]
                self.profile_list.selectionModel().clear()
                self.delete_btn.setText("Delete pipeline")
                self.rename_btn.setText("Rename pipeline")
            else:
                return  # pragma: no cover
        except KeyError:  # pragma: no cover
            return

        # TODO update with knowledge from profile dict
        self.delete_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)
        self.info_label.setPlainText(profile.pretty_print(AnalysisAlgorithmSelection.__register__))

    def synchronize_spacing(self):
        if self.lock_spacing.isChecked():
            self.spacing[1].setValue(self.spacing[2].value())

    def image_spacing_change(self):
        spacing = [el.value() / UNIT_SCALE[self.units.currentIndex()] for el in self.spacing]

        if not self.spacing[0].isEnabled():
            spacing = spacing[1:]
        self._settings.image_spacing = spacing

        voxel_size = 1
        for el in self._settings.image_spacing:
            voxel_size *= el * UNIT_SCALE[self.units.currentIndex()]
        self.voxel_size_label.setText(
            f"Voxel_size: {voxel_size} {self.units.currentEnum().name} <sup>{len(self._settings.image_spacing)}</sup>"
        )

    def update_spacing(self, index=None):
        voxel_size = 1
        value = self.units.currentEnum()
        if index is not None:
            self._settings.set("units_value", value)
        for el, sp in zip(self.spacing[::-1], self._settings.image_spacing[::-1]):
            el.blockSignals(True)
            current_size = sp * UNIT_SCALE[self.units.currentIndex()]
            voxel_size *= current_size
            el.setValue(current_size)
            el.blockSignals(False)
        self.spacing[0].setDisabled(len(self._settings.image_spacing) == 2)
        self.voxel_size_label.setText(
            f"Voxel_size: {voxel_size} {value.name} <sup>{len(self._settings.image_spacing)}</sup>"
        )

    def update_profile_list(self):
        current_names = set(self._settings.roi_profiles.keys())
        self.profile_list.clear()
        self.profile_list.addItems(sorted(current_names))
        self.pipeline_list.clear()
        self.pipeline_list.addItems(sorted(set(self._settings.roi_pipelines.keys())))
        self.delete_btn.setDisabled(True)
        self.rename_btn.setDisabled(True)
        self.info_label.setPlainText("")

    def showEvent(self, _event):
        self.update_spacing()

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate and self.isVisible():
            self.update_spacing()
        return super().event(event)

    @ensure_main_thread
    def _update_measurement_chk(self):
        self.multiple_files_chk.setChecked(self._settings.get("multiple_files_widget", False))

    def delete_profile(self):
        text, dkt = "", {}
        if self.profile_list.selectedItems():
            text = self.profile_list.selectedItems()[0].text()
            dkt = self._settings.roi_profiles
        elif self.pipeline_list.selectedItems():
            text = self.pipeline_list.selectedItems()[0].text()
            dkt = self._settings.roi_pipelines
        if text != "":
            self.delete_btn.setDisabled(True)
            del dkt[text]
            self.update_profile_list()

    def export_profile(self):
        exp = ExportDialog(self._settings.roi_profiles, ProfileDictViewer)
        if not exp.exec_():
            return
        dial = PSaveDialog(
            "Segment profile (*.json)",
            settings=self._settings,
            path=IO_SAVE_DIRECTORY,
            caption="Export profile segment",
        )
        dial.selectFile("segment_profile.json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            data = {x: self._settings.roi_profiles[x] for x in exp.get_export_list()}
            with open(file_path, "w", encoding="utf-8") as ff:
                json.dump(data, ff, cls=self._settings.json_encoder_class, indent=2)

    def import_profiles(self):
        dial = PLoadDialog(
            LoadPlanJson,
            settings=self._settings,
            path=IO_SAVE_DIRECTORY,
            caption="Import profile segment",
        )
        if dial.exec_():
            res = dial.get_result()
            profs, err = res.load_class.load(res.load_location)
            if err:
                DataImportErrorDialog({res.load_location[0]: err}).exec_()
            if not profs:
                return
            profiles_dict = self._settings.roi_profiles
            imp = ImportDialog(profs, profiles_dict, ProfileDictViewer)
            if not imp.exec_():
                return
            for original_name, final_name in imp.get_import_list():
                profiles_dict[final_name] = profs[original_name]
            self._settings.dump()

    def export_pipeline(self):
        exp = ExportDialog(self._settings.roi_pipelines, ProfileDictViewer)
        if not exp.exec_():
            return
        dial = PSaveDialog(
            "Segment pipeline (*.json)",
            settings=self._settings,
            path=IO_SAVE_DIRECTORY,
            caption="Export pipeline segment",
        )
        dial.selectFile("segment_pipeline.json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            data = {x: self._settings.roi_pipelines[x] for x in exp.get_export_list()}
            with open(file_path, "w", encoding="utf-8") as ff:
                json.dump(data, ff, cls=self._settings.json_encoder_class, indent=2)

    def import_pipeline(self):
        dial = PLoadDialog(
            LoadPlanJson,
            settings=self._settings,
            path=IO_SAVE_DIRECTORY,
            caption="Import pipeline segment",
        )
        if dial.exec_():
            res = dial.get_result()
            profs, err = res.load_class.load(res.load_location)
            if err:
                QMessageBox.warning(self, "Import error", "error during importing, part of data were filtered.")
            profiles_dict = self._settings.roi_pipelines
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
            profiles_dict = self._settings.roi_profiles
        elif self.pipeline_list.selectedItems():
            profile_name = self.pipeline_list.selectedItems()[0].text()
            profiles_dict = self._settings.roi_pipelines
        if profile_name == "":
            return
        text, ok = QInputDialog.getText(self, "New profile name", f"New name for {profile_name}", text=profile_name)
        if ok:
            text = text.strip()
            if text in profiles_dict:
                res = QMessageBox.warning(
                    self,
                    "Already exist",
                    f"Profile with name {text} already exist. Would you like to overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if res == QMessageBox.No:
                    self.rename_profile()
                    return
            profiles_dict[text] = profiles_dict.pop(profile_name)
            self._settings.dump()
            self.update_profile_list()


class MeasurementListWidgetItem(QListWidgetItem):
    def __init__(self, stat: Union[Node, Leaf], *args, **kwargs):
        super().__init__(stat.pretty_print(MEASUREMENT_DICT), *args, **kwargs)
        self.stat = stat


class MeasurementSettings(QWidget):
    """
    :type settings: Settings
    """

    def __init__(self, settings: PartSettings, parent=None):
        super().__init__(parent)
        self.chosen_element: Optional[MeasurementListWidgetItem] = None
        self.chosen_element_area: Optional[Tuple[AreaType, float]] = None
        self.settings = settings
        self.profile_list = QListWidget(self)
        self.profile_description = QTextEdit(self)
        self.profile_description.setReadOnly(True)
        self.profile_options = QListWidget()
        self.profile_options_chosen = QListWidget()
        self.measurement_area_choose = QEnumComboBox(enum_class=AreaType)
        self.per_component = QEnumComboBox(enum_class=PerComponent)
        self.power_num = QDoubleSpinBox()
        self.power_num.setDecimals(3)
        self.power_num.setRange(-100, 100)
        self.power_num.setValue(1)
        self.choose_butt = QPushButton("→", self)
        self.discard_butt = QPushButton("←", self)
        self.proportion_butt = QPushButton("Ratio", self)
        self.proportion_butt.setToolTip("Create proportion from two parameter")
        self.move_up = QPushButton("↑", self)
        self.move_down = QPushButton("↓", self)
        self.remove_button = QPushButton("Remove")
        self.save_butt = QPushButton("Save")
        self.save_butt.setToolTip("Set name for set and choose at least one parameter")
        self.save_butt_with_name = QPushButton("Save with custom parameters designation")
        self.save_butt_with_name.setToolTip("Set name for set and choose at least one parameter")
        self.reset_butt = QPushButton("Clear")
        self.soft_reset_butt = QPushButton("Remove user parameters")
        self.profile_name = QLineEdit(self)

        self.delete_profile_butt = QPushButton("Delete")
        self.export_profiles_butt = QPushButton("Export")
        self.import_profiles_butt = QPushButton("Import")
        self.edit_profile_butt = QPushButton("Edit")

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
        self.export_profiles_butt.clicked.connect(self.export_measurement_profiles)
        self.import_profiles_butt.clicked.connect(self.import_measurement_profiles)
        self.edit_profile_butt.clicked.connect(self.edit_profile)

        self.profile_list.itemSelectionChanged.connect(self.profile_chosen)
        self.profile_options.itemSelectionChanged.connect(self.create_selection_changed)
        self.profile_options_chosen.itemSelectionChanged.connect(self.create_selection_chosen_changed)
        self.settings.measurement_profiles_changed.connect(self._refresh_profiles)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Measurement set:"))
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(self.profile_list)
        profile_layout.addWidget(self.profile_description)
        profile_buttons_layout = QHBoxLayout()
        profile_buttons_layout.addWidget(self.delete_profile_butt)
        profile_buttons_layout.addWidget(self.export_profiles_butt)
        profile_buttons_layout.addWidget(self.import_profiles_butt)
        profile_buttons_layout.addWidget(self.edit_profile_butt)
        profile_buttons_layout.addStretch()
        layout.addLayout(profile_layout)
        layout.addLayout(profile_buttons_layout)
        heading_layout = QHBoxLayout()
        heading_layout.addWidget(h_line(), 6)
        layout.addLayout(heading_layout)
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Set name:"))
        name_layout.addWidget(self.profile_name)
        name_layout.addStretch()
        name_layout.addWidget(QLabel("Per component:"))
        name_layout.addWidget(self.per_component)
        name_layout.addWidget(QLabel("Area:"))
        name_layout.addWidget(self.measurement_area_choose)
        name_layout.addWidget(QLabel("to power:"))
        name_layout.addWidget(self.power_num)
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

        for profile in MEASUREMENT_DICT.values():
            help_text = profile.get_description()
            lw = MeasurementListWidgetItem(profile.get_starting_leaf())
            lw.setToolTip(help_text)
            self.profile_options.addItem(lw)
        self._refresh_profiles()

    def _refresh_profiles(self):
        item = self.profile_list.currentItem()
        items = list(self.settings.measurement_profiles.keys())
        try:
            index = items.index(item.text())
        except (ValueError, AttributeError):
            index = -1

        self.profile_list.clear()
        self.profile_list.addItems(items)
        self.profile_list.setCurrentRow(index)

        if self.profile_list.count() == 0:
            self.export_profiles_butt.setDisabled(True)
        else:
            self.export_profiles_butt.setEnabled(True)

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

    def delete_profile(self):
        item = self.profile_list.currentItem()
        del self.settings.measurement_profiles[str(item.text())]

    def profile_chosen(self):
        self.delete_profile_butt.setEnabled(True)
        if self.profile_list.count() == 0:
            self.profile_description.setText("")
            return
        item = self.profile_list.currentItem()
        if item is None:
            self.profile_description.setText("")
            return
        profile = self.settings.measurement_profiles[item.text()]
        self.profile_description.setText(str(profile))

    def create_selection_changed(self):
        self.choose_butt.setEnabled(True)
        self.proportion_butt.setEnabled(True)

    def proportion_action(self):
        # TODO use get_parameters
        if self.chosen_element is None:
            item = self.profile_options.currentItem()
            self.chosen_element_area = self.get_parameters(
                deepcopy(item.stat),
                self.measurement_area_choose.currentEnum(),
                self.per_component.currentEnum(),
                self.power_num.value(),
            )
            if self.chosen_element_area is None:
                return
            self.chosen_element = item
            item.setIcon(QIcon(os.path.join(icons_dir, "task-accepted.png")))
        elif (
            self.profile_options.currentItem() is self.chosen_element
            and self.measurement_area_choose.currentEnum() == self.chosen_element_area.area
            and self.per_component.currentEnum() == self.chosen_element_area.per_component
        ):
            self.chosen_element.setIcon(QIcon())
            self.chosen_element = None
        else:
            item = cast(MeasurementListWidgetItem, self.profile_options.currentItem())
            leaf = self.get_parameters(
                deepcopy(item.stat),
                self.measurement_area_choose.currentEnum(),
                self.per_component.currentEnum(),
                self.power_num.value(),
            )
            if leaf is None:
                return
            lw = MeasurementListWidgetItem(Node(op="/", left=self.chosen_element_area, right=leaf))
            lw.setToolTip("User defined")
            self._add_option(lw)
            self.chosen_element.setIcon(QIcon())
            self.chosen_element = None
            self.chosen_element_area = None

    def _add_option(self, item: MeasurementListWidgetItem):
        for i in range(self.profile_options_chosen.count()):
            if item.text() == self.profile_options_chosen.item(i).text():
                return
        self.profile_options_chosen.addItem(item)
        if self.good_name():
            self.save_butt.setEnabled(True)
            self.save_butt_with_name.setEnabled(True)
        if self.profile_options.count() == 0:
            self.choose_butt.setDisabled(True)

    def create_selection_chosen_changed(self):
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
        return str(self.profile_name.text()).strip() != ""

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

    def form_dialog(self, arguments):
        return FormDialog(arguments, settings=self.settings, parent=self)

    def get_parameters(
        self, node: Union[Node, Leaf], area: AreaType, component: PerComponent, power: float
    ) -> Union[Leaf, Node, None]:
        if isinstance(node, Node):
            return node
        node = node.replace_(power=power)
        if node.area is None:
            node = node.replace_(area=area)
        if node.per_component is None:
            try:
                node = node.replace_(per_component=component)
            except ValueError as e:  # pragma: no cover
                QMessageBox().warning(self, "Problem in add measurement", str(e))
        with suppress(KeyError):
            arguments = MEASUREMENT_DICT[str(node.name)]._get_fields()  # pylint: disable=protected-access
            if len(arguments) > 0 and not dict(node.parameters):
                dial = self.form_dialog(arguments)
                if dial.exec_():
                    node = node.copy(update={"parameters": dial.get_values()})
                else:
                    return None
        return node

    def choose_option(self):
        selected_item = self.profile_options.currentItem()
        if not isinstance(selected_item, MeasurementListWidgetItem):  # pragma: no cover
            raise ValueError(f"Current item (type: {type(selected_item)} is not instance of MeasurementListWidgetItem")
        node = deepcopy(selected_item.stat)
        # noinspection PyTypeChecker
        node = self.get_parameters(
            node, self.measurement_area_choose.currentEnum(), self.per_component.currentEnum(), self.power_num.value()
        )
        if node is None:
            return
        lw = MeasurementListWidgetItem(node)
        lw.setToolTip(selected_item.toolTip())
        self._add_option(lw)

    def discard_option(self):
        selected_item: MeasurementListWidgetItem = self.profile_options_chosen.currentItem()
        lw = MeasurementListWidgetItem(deepcopy(selected_item.stat))
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
        profile = self.settings.measurement_profiles[str(item.text())]
        self.profile_options_chosen.clear()
        self.profile_name.setText(item.text())
        for ch in profile.chosen_fields:
            self.profile_options_chosen.addItem(MeasurementListWidgetItem(ch.calculation_tree))
        self.save_butt.setEnabled(True)
        self.save_butt_with_name.setEnabled(True)

    def save_action(self):
        if self.profile_name.text() in self.settings.measurement_profiles:
            ret = QMessageBox.warning(
                self,
                "Profile exist",
                "Profile exist\nWould you like to overwrite it?",
                QMessageBox.No | QMessageBox.Yes,
            )
            if ret == QMessageBox.No:
                return
        selected_values = []
        for i in range(self.profile_options_chosen.count()):
            element = cast(MeasurementListWidgetItem, self.profile_options_chosen.item(i))
            selected_values.append(MeasurementEntry(name=element.text(), calculation_tree=element.stat))
        stat_prof = MeasurementProfile(name=self.profile_name.text(), chosen_fields=selected_values)
        self.settings.measurement_profiles[stat_prof.name] = stat_prof
        self.settings.dump()
        self.export_profiles_butt.setEnabled(True)
        self.save_butt.setDisabled(True)

    def named_save_action(self):
        if self.profile_name.text() in self.settings.measurement_profiles:
            ret = QMessageBox.warning(
                self,
                "Profile exist",
                "Profile exist\nWould you like to overwrite it?",
                QMessageBox.No | QMessageBox.Yes,
            )
            if ret == QMessageBox.No:
                return
        selected_values = []
        for i in range(self.profile_options_chosen.count()):
            txt = str(self.profile_options_chosen.item(i).text())
            selected_values.append((txt, str, txt))
        val_dialog = MultipleInput("Set fields name", list(selected_values), parent=self)
        if val_dialog.exec_():
            selected_values = []
            for i in range(self.profile_options_chosen.count()):
                element = cast(MeasurementListWidgetItem, self.profile_options_chosen.item(i))
                selected_values.append(
                    MeasurementEntry(name=val_dialog.result[element.text()], calculation_tree=element.stat)
                )
            stat_prof = MeasurementProfile(name=self.profile_name.text(), chosen_fields=selected_values)
            self.settings.measurement_profiles[stat_prof.name] = stat_prof
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
        for profile in MEASUREMENT_DICT.values():
            help_text = profile.get_description()
            lw = MeasurementListWidgetItem(profile.get_starting_leaf())
            lw.setToolTip(help_text)
            self.profile_options.addItem(lw)

    def soft_reset(self):
        # TODO rim should not be removed
        shift = 0
        for i in range(self.profile_options.count()):
            item = self.profile_options.item(i - shift)
            if str(item.text()) not in MEASUREMENT_DICT:
                self.profile_options.takeItem(i - shift)
                if item == self.chosen_element:
                    self.chosen_element = None
                del item
                shift += 1
        self.create_selection_changed()

    def export_measurement_profiles(self):
        exp = ExportDialog(self.settings.measurement_profiles, StringViewer, parent=self)
        if not exp.exec_():
            return
        dial = PSaveDialog(
            "Measurement profile (*.json)",
            settings=self.settings,
            path="io.export_directory",
            caption="Export settings profiles",
        )
        dial.selectFile("measurements_profile.json")

        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            data = {x: self.settings.measurement_profiles[x] for x in exp.get_export_list()}
            with open(file_path, "w", encoding="utf-8") as ff:
                json.dump(data, ff, cls=self.settings.json_encoder_class, indent=2)

    def import_measurement_profiles(self):
        dial = PLoadDialog(
            LoadPlanJson,
            settings=self.settings,
            path="io.export_directory",
            caption="Import settings profiles",
            parent=self,
        )
        if dial.exec_():
            res = dial.get_result()
            stat, err = res.load_class.load(res.load_location)
            if err:
                QMessageBox.warning(self, "Import error", "error during importing, part of data were filtered.")
            measurement_dict = self.settings.measurement_profiles
            imp = ImportDialog(stat, measurement_dict, StringViewer)
            if not imp.exec_():
                return
            for original_name, final_name in imp.get_import_list():
                measurement_dict[final_name] = stat[original_name]
            self.settings.dump()


class SegAdvancedWindow(AdvancedWindow):
    """
    :type settings: Settings
    """

    def __init__(self, settings, parent=None, reload_list=None):
        super().__init__(settings, ["result_image", "raw_image"], reload_list, parent)
        self.settings = settings

        self.setWindowTitle("Settings and Measurement")
        self.advanced_settings = Properties(settings)
        self.measurement = MeasurementWidget(settings)
        self.measurement_settings = MeasurementSettings(settings)
        self.insertTab(0, self.advanced_settings, "Properties")
        self.addTab(self.measurement_settings, "Measurements settings")
        self.addTab(self.measurement, "Measurements")
        self.setCurrentWidget(self.advanced_settings)


class MultipleInput(QDialog):
    def __init__(self, text, help_text, objects_list=None, parent=None):
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
        super().__init__(parent=parent)
        ok_butt = QPushButton("Ok", self)
        cancel_butt = QPushButton("Cancel", self)
        self.object_dict = {}
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
        res = {}
        for name, (type_of, item) in self.object_dict.items():
            if type_of == str:
                val = str(item.text())
                if not val.strip():
                    QMessageBox.warning(self, "Not all fields filled", "")
                    return
                else:
                    res[name] = val
            else:
                val = type_of(item.value())
                res[name] = val
        self.result = res
        self.accept()

    @property
    def get_response(self):
        return self.result
