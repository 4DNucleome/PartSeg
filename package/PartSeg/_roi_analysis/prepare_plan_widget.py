import json
import logging
import os
import typing
from copy import copy, deepcopy
from enum import Enum
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PartSeg.common_gui.universal_gui_part import EnumComboBox
from PartSegCore.algorithm_describe_base import AlgorithmProperty, ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.analysis.calculation_plan import (
    CalculationPlan,
    MaskBase,
    MaskCreate,
    MaskFile,
    MaskIntersection,
    MaskMapper,
    MaskSub,
    MaskSuffix,
    MaskSum,
    MeasurementCalculate,
    NodeType,
    Operations,
    PlanChanges,
    RootType,
    Save,
)
from PartSegCore.analysis.save_functions import save_dict
from PartSegCore.io_utils import SaveBase
from PartSegCore.universal_const import Units

from ..common_gui.custom_save_dialog import FormDialog
from ..common_gui.mask_widget import MaskWidget
from ..common_gui.searchable_list_widget import SearchableListWidget
from ..common_gui.universal_gui_part import right_label
from .partseg_settings import PartSettings
from .profile_export import ExportDialog, ImportDialog

group_sheet = (
    "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em;} "
    "QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px;}"
)

MAX_CHANNEL_NUM = 10


class MaskDialog(QDialog):
    def __init__(self, mask_names):
        super().__init__()
        self.mask_names = mask_names
        completer = QCompleter(list(mask_names))
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.setWindowTitle("Masks name choose")
        self.mask1_name = QLineEdit()
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn = QPushButton("Ok")

        self.mask1_name.setCompleter(completer)
        self.mask1_name.textChanged.connect(self.text_changed)
        self.cancel_btn.clicked.connect(self.close)
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDisabled(True)

        layout = QGridLayout()
        layout.addWidget(right_label("Mask 1 name:"), 0, 0)
        layout.addWidget(self.mask1_name, 0, 1)
        layout.addWidget(self.cancel_btn, 2, 0)
        layout.addWidget(self.ok_btn, 2, 1)
        self.setLayout(layout)

    def text_changed(self):
        text1 = self.get_result()[0]
        if text1 == "" or text1 not in self.mask_names:
            self.ok_btn.setDisabled(True)
        else:
            self.ok_btn.setDisabled(False)

    def get_result(self):
        text1 = str(self.mask1_name.text()).strip()
        return (text1,)


class TwoMaskDialog(QDialog):
    def __init__(self, mask_names):
        """
        :type mask_names: set
        :param mask_names: iterable collection of all available mask names
        """
        super().__init__()
        self.mask_names = mask_names
        completer = QCompleter(list(mask_names))
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.setWindowTitle("Masks name choose")
        self.mask1_name = QLineEdit()
        self.mask2_name = QLineEdit()
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn = QPushButton("Ok")

        self.mask1_name.setCompleter(completer)
        self.mask1_name.textChanged.connect(self.text_changed)
        self.mask2_name.setCompleter(completer)
        self.mask2_name.textChanged.connect(self.text_changed)
        self.cancel_btn.clicked.connect(self.close)
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDisabled(True)

        layout = QGridLayout()
        layout.addWidget(right_label("Mask 1 name:"), 0, 0)
        layout.addWidget(self.mask1_name, 0, 1)
        layout.addWidget(right_label("Mask 2 name:"), 1, 0)
        layout.addWidget(self.mask2_name, 1, 1)
        layout.addWidget(self.cancel_btn, 2, 0)
        layout.addWidget(self.ok_btn, 2, 1)
        self.setLayout(layout)

    def text_changed(self):
        text1, text2 = self.get_result()
        if text1 == "" or text2 == "" or text1 not in self.mask_names or text2 not in self.mask_names:
            self.ok_btn.setDisabled(True)
        else:
            self.ok_btn.setDisabled(text1 == text2)

    def get_result(self):
        text1 = str(self.mask1_name.text()).strip()
        text2 = str(self.mask2_name.text()).strip()
        return text1, text2


def stretch_widget(widget):
    base_widget = QWidget()
    base_widget.setContentsMargins(0, 0, 0, 0)
    lay = QVBoxLayout()
    lay.addWidget(widget)
    lay.addStretch(3)
    base_widget.setLayout(lay)
    return base_widget


class FileMask(QWidget):
    value_changed = Signal()

    def __init__(self):
        super().__init__()
        self.select_type = QComboBox()
        self.select_type.addItems(["Suffix", "Replace", "Mapping file"])
        self.values = ["_mask", ("", ""), ""]
        self.first_text = QLineEdit(self.values[0])
        self.second_text = QLineEdit()
        self.first_label = QLabel("Use suffix:")
        self.second_label = QLabel("Replace:")
        self.select_file_btn = QPushButton("Select file")
        self.state = 0

        layout = QGridLayout()
        layout.addWidget(self.select_type, 0, 0, 1, 2)
        layout.addWidget(self.first_label, 1, 0)
        layout.addWidget(self.second_label, 1, 1)
        layout.addWidget(self.first_text, 2, 0)
        layout.addWidget(self.second_text, 2, 1)
        layout.addWidget(self.select_file_btn, 3, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        self.setLayout(layout)

        self.second_text.setHidden(True)
        self.second_label.setHidden(True)
        self.select_file_btn.setHidden(True)

        self.first_text.textChanged.connect(self.value_changed.emit)
        self.second_text.textChanged.connect(self.value_changed.emit)
        self.select_type.currentIndexChanged.connect(self.value_changed.emit)
        self.select_type.currentIndexChanged.connect(self.change_type)
        self.select_file_btn.clicked.connect(self.select_file)

    def change_type(self, index):
        if self.state == index:
            return
        if self.state == 1:
            self.values[1] = self.first_text.text(), self.second_text.text()
        else:
            self.values[self.state] = self.first_text.text()
        if index == 1:
            self.second_text.setHidden(False)
            self.second_label.setHidden(False)
            self.layout().setColumnStretch(1, 1)
            self.first_text.setText(self.values[1][0])
            self.second_text.setText(self.values[1][1])
            self.first_label.setText("Base")
            self.select_file_btn.setHidden(True)
        else:
            self.second_text.setHidden(True)
            self.second_label.setHidden(True)
            self.layout().setColumnStretch(1, 0)
            self.first_label.setText(["Use suffix:", "", "Path:"][index])
            self.first_text.setText(self.values[index])
            self.select_file_btn.setHidden(index == 0)
        self.state = index

    def select_file(self):
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)

        if dial.exec():
            self.first_text.setText(dial.selectedFiles()[0])

    def is_valid(self):
        if self.select_type.currentIndex() == 0:
            return self.first_text.text().strip() != ""
        if self.select_type.currentIndex() == 1:
            return self.first_text.text().strip() != "" and self.second_text.text().strip() != ""

        text = self.first_text.text().strip()
        return text != "" and os.path.exists(text) and os.path.isfile(text)

    def get_value(self, name=""):
        mask_type = self.select_type.currentIndex()
        if mask_type == 0:
            return MaskSuffix(name, self.first_text.text().strip())
        if mask_type == 1:
            return MaskSub(name, self.first_text.text().strip(), self.second_text.text().strip())

        return MaskFile(name, self.first_text.text().strip())


class MaskOperation(Enum):
    mask_intersection = 0
    mask_sum = 1

    def __str__(self):
        return self.name.replace("_", " ").capitalize()


def _check_widget(tab_widget, type_):
    return isinstance(tab_widget.currentWidget(), type_) or tab_widget.currentWidget().findChildren(type_)


class CreatePlan(QWidget):

    plan_created = Signal()
    plan_node_changed = Signal()

    def __init__(self, settings: PartSettings):
        super().__init__()
        self.settings = settings
        self.save_translate_dict: typing.Dict[str, SaveBase] = {x.get_short_name(): x for x in save_dict.values()}
        self.plan = PlanPreview(self)
        self.save_plan_btn = QPushButton("Save")
        self.clean_plan_btn = QPushButton("Remove all")
        self.remove_btn = QPushButton("Remove")
        self.update_element_chk = QCheckBox("Update element")
        self.change_root = EnumComboBox(RootType)
        self.save_choose = QComboBox()
        self.save_choose.addItem("<none>")
        self.save_choose.addItems(list(self.save_translate_dict.keys()))
        self.save_btn = QPushButton("Save")
        self.segment_profile = SearchableListWidget()
        self.pipeline_profile = SearchableListWidget()
        self.segment_stack = QTabWidget()
        self.segment_stack.addTab(self.segment_profile, "Profile")
        self.segment_stack.addTab(self.pipeline_profile, "Pipeline")
        self.generate_mask_btn = QPushButton("Add mask")
        self.generate_mask_btn.setToolTip("Mask need to have unique name")
        self.mask_name = QLineEdit()
        self.mask_operation = EnumComboBox(MaskOperation)

        self.chanel_num = QSpinBox()
        self.choose_channel_for_measurements = QComboBox()
        self.choose_channel_for_measurements.addItems(
            ["Same as segmentation"] + [str(x + 1) for x in range(MAX_CHANNEL_NUM)]
        )
        self.units_choose = EnumComboBox(Units)
        self.units_choose.set_value(self.settings.get("units_value", Units.nm))
        self.chanel_num.setRange(0, 10)
        self.expected_node_type = None
        self.save_constructor = None

        self.chose_profile_btn = QPushButton("Add Profile")
        self.get_big_btn = QPushButton("Leave the biggest")
        self.get_big_btn.hide()
        self.get_big_btn.setDisabled(True)
        self.measurements_list = SearchableListWidget(self)
        self.measurement_name_prefix = QLineEdit(self)
        self.add_calculation_btn = QPushButton("Add measurement calculation")
        self.information = QTextEdit()
        self.information.setReadOnly(True)

        self.protect = False
        self.mask_set = set()
        self.calculation_plan = CalculationPlan()
        self.plan.set_plan(self.calculation_plan)
        self.segmentation_mask = MaskWidget(settings)
        self.file_mask = FileMask()

        self.change_root.currentIndexChanged.connect(self.change_root_type)
        self.save_choose.currentTextChanged.connect(self.save_changed)
        self.measurements_list.currentTextChanged.connect(self.show_measurement)
        self.segment_profile.currentTextChanged.connect(self.show_segment)
        self.measurements_list.currentTextChanged.connect(self.show_measurement_info)
        self.segment_profile.currentTextChanged.connect(self.show_segment_info)
        self.pipeline_profile.currentTextChanged.connect(self.show_segment_info)
        self.pipeline_profile.currentTextChanged.connect(self.show_segment)
        self.mask_name.textChanged.connect(self.mask_name_changed)
        self.generate_mask_btn.clicked.connect(self.create_mask)
        self.clean_plan_btn.clicked.connect(self.clean_plan)
        self.remove_btn.clicked.connect(self.remove_element)
        self.mask_name.textChanged.connect(self.mask_text_changed)
        self.chose_profile_btn.clicked.connect(self.add_segmentation)
        self.get_big_btn.clicked.connect(self.add_leave_biggest)
        self.add_calculation_btn.clicked.connect(self.add_measurement)
        self.save_plan_btn.clicked.connect(self.add_calculation_plan)
        # self.forgot_mask_btn.clicked.connect(self.forgot_mask)
        # self.cmap_save_btn.clicked.connect(self.save_to_cmap)
        self.save_btn.clicked.connect(self.add_save_to_project)
        self.update_element_chk.stateChanged.connect(self.mask_text_changed)
        self.update_element_chk.stateChanged.connect(self.show_measurement)
        self.update_element_chk.stateChanged.connect(self.show_segment)
        self.update_element_chk.stateChanged.connect(self.update_names)
        self.segment_stack.currentChanged.connect(self.change_segmentation_table)

        plan_box = QGroupBox("Prepare workflow:")
        lay = QVBoxLayout()
        lay.addWidget(self.plan)
        bt_lay = QGridLayout()
        bt_lay.setSpacing(1)
        bt_lay.addWidget(self.save_plan_btn, 0, 0)
        bt_lay.addWidget(self.clean_plan_btn, 0, 1)
        bt_lay.addWidget(self.remove_btn, 1, 0)
        bt_lay.addWidget(self.update_element_chk, 1, 1)
        lay.addLayout(bt_lay)
        plan_box.setLayout(lay)
        plan_box.setStyleSheet(group_sheet)

        other_box = QGroupBox("Other operations:")
        other_box.setContentsMargins(0, 0, 0, 0)
        bt_lay = QVBoxLayout()
        bt_lay.setSpacing(0)
        bt_lay.addWidget(QLabel("Root type:"))
        bt_lay.addWidget(self.change_root)
        bt_lay.addStretch(1)
        bt_lay.addWidget(QLabel("Saving:"))
        bt_lay.addWidget(self.save_choose)
        bt_lay.addWidget(self.save_btn)
        other_box.setLayout(bt_lay)
        other_box.setStyleSheet(group_sheet)

        mask_box = QGroupBox("Use mask from:")
        mask_box.setStyleSheet(group_sheet)
        self.mask_stack = QTabWidget()

        self.mask_stack.addTab(stretch_widget(self.file_mask), "File")
        self.mask_stack.addTab(stretch_widget(self.segmentation_mask), "Current ROI")
        self.mask_stack.addTab(stretch_widget(self.mask_operation), "Operations on masks")
        self.mask_stack.setTabToolTip(2, "Allows to create mask which is based on masks previously added to plan.")

        lay = QGridLayout()
        lay.setSpacing(0)
        lay.addWidget(self.mask_stack, 0, 0, 1, 2)
        label = QLabel("Mask name:")
        label.setToolTip("Needed if you would like to reuse this mask in tab 'Operations on masks'")
        self.mask_name.setToolTip("Needed if you would like to reuse this mask in tab 'Operations on masks'")
        lay.addWidget(label, 1, 0)
        lay.addWidget(self.mask_name, 1, 1)
        lay.addWidget(self.generate_mask_btn, 2, 0, 1, 2)
        mask_box.setLayout(lay)

        segment_box = QGroupBox("ROI extraction:")
        segment_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.setSpacing(0)
        lay.addWidget(self.segment_stack)
        lay.addWidget(self.chose_profile_btn)
        lay.addWidget(self.get_big_btn)
        segment_box.setLayout(lay)

        measurement_box = QGroupBox("Set of measurements:")
        measurement_box.setStyleSheet(group_sheet)
        lay = QGridLayout()
        lay.setSpacing(0)
        lay.addWidget(self.measurements_list, 0, 0, 1, 2)
        lab = QLabel("Name prefix:")
        lab.setToolTip("Prefix added before each column name")
        lay.addWidget(lab, 1, 0)
        lay.addWidget(self.measurement_name_prefix, 1, 1)
        lay.addWidget(QLabel("Channel:"), 2, 0)
        lay.addWidget(self.choose_channel_for_measurements, 2, 1)
        lay.addWidget(QLabel("Units:"), 3, 0)
        lay.addWidget(self.units_choose, 3, 1)
        lay.addWidget(self.add_calculation_btn, 4, 0, 1, 2)
        measurement_box.setLayout(lay)

        info_box = QGroupBox("Information")
        info_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.information)
        info_box.setLayout(lay)

        layout = QGridLayout()
        fst_col = QVBoxLayout()
        fst_col.addWidget(plan_box, 1)
        fst_col.addWidget(mask_box)
        layout.addWidget(plan_box, 0, 0, 5, 1)
        # layout.addWidget(plan_box, 0, 0, 3, 1)
        # layout.addWidget(mask_box, 3, 0, 2, 1)
        # layout.addWidget(segmentation_mask_box, 1, 1)
        layout.addWidget(mask_box, 0, 2, 1, 2)
        layout.addWidget(other_box, 0, 1)
        layout.addWidget(segment_box, 1, 1, 1, 2)
        layout.addWidget(measurement_box, 1, 3)
        layout.addWidget(info_box, 3, 1, 1, 3)
        self.setLayout(layout)

        self.generate_mask_btn.setDisabled(True)
        self.chose_profile_btn.setDisabled(True)
        self.add_calculation_btn.setDisabled(True)

        self.mask_allow = False
        self.segment_allow = False
        self.file_mask_allow = False
        self.node_type = NodeType.root
        self.node_name = ""
        self.plan_node_changed.connect(self.mask_text_changed)
        self.plan.changed_node.connect(self.node_type_changed)
        self.plan_node_changed.connect(self.show_segment)
        self.plan_node_changed.connect(self.show_measurement)
        self.plan_node_changed.connect(self.mask_stack_change)
        self.mask_stack.currentChanged.connect(self.mask_stack_change)
        self.file_mask.value_changed.connect(self.mask_stack_change)
        self.mask_name.textChanged.connect(self.mask_stack_change)
        self.node_type_changed()

    def change_root_type(self):
        value: RootType = self.change_root.get_value()
        self.calculation_plan.set_root_type(value)
        self.plan.update_view()

    def change_segmentation_table(self):
        index = self.segment_stack.currentIndex()
        text = self.segment_stack.tabText(index)
        if self.update_element_chk.isChecked():
            self.chose_profile_btn.setText("Replace " + text)
        else:
            self.chose_profile_btn.setText("Add " + text)
        self.segment_profile.setCurrentItem(None)
        self.pipeline_profile.setCurrentItem(None)

    def save_changed(self, text):
        text = str(text)
        if text == "<none>":
            self.save_btn.setText("Save")
            self.save_btn.setToolTip("Choose file type")
            self.expected_node_type = None
            self.save_constructor = None
        else:
            save_class = self.save_translate_dict.get(text, None)
            if save_class is None:
                self.save_choose.setCurrentText("<none>")
                return
            self.save_btn.setText(f"Save to {save_class.get_short_name()}")
            self.save_btn.setToolTip("Choose mask create in plan view")
            if save_class.need_mask():
                self.expected_node_type = NodeType.mask
            elif save_class.need_segmentation():
                self.expected_node_type = NodeType.segment
            else:
                self.expected_node_type = NodeType.root
            self.save_constructor = Save
        self.save_activate()

    def save_activate(self):
        self.save_btn.setDisabled(True)
        if self.node_type == self.expected_node_type:
            self.save_btn.setEnabled(True)
            return

    def segmentation_from_project(self):
        self.calculation_plan.add_step(Operations.reset_to_base)
        self.plan.update_view()

    def update_names(self):
        if self.update_element_chk.isChecked():
            self.chose_profile_btn.setText("Replace Profile")
            self.add_calculation_btn.setText("Replace set of measurements")
            self.generate_mask_btn.setText("Replace mask")
        else:
            self.chose_profile_btn.setText("Add Profile")
            self.add_calculation_btn.setText("Add set of measurements")
            self.generate_mask_btn.setText("Generate mask")

    def node_type_changed(self):
        # self.cmap_save_btn.setDisabled(True)
        self.save_btn.setDisabled(True)
        self.node_name = ""
        if self.plan.currentItem() is None:
            self.mask_allow = False
            self.file_mask_allow = False
            self.segment_allow = False
            self.remove_btn.setDisabled(True)
            self.plan_node_changed.emit()
            logging.debug("[node_type_changed] return")
            return
        node_type = self.calculation_plan.get_node_type()
        self.node_type = node_type
        if node_type in [NodeType.file_mask, NodeType.mask, NodeType.segment, NodeType.measurement, NodeType.save]:
            self.remove_btn.setEnabled(True)
        else:
            self.remove_btn.setEnabled(False)
        if node_type in (NodeType.mask, NodeType.file_mask):
            self.mask_allow = False
            self.segment_allow = True
            self.file_mask_allow = False
            self.node_name = self.calculation_plan.get_node().operation.name
        elif node_type == NodeType.segment:
            self.mask_allow = True
            self.segment_allow = False
            self.file_mask_allow = False
            self.save_btn.setEnabled(True)
            # self.cmap_save_btn.setEnabled(True)
        elif node_type == NodeType.root:
            self.mask_allow = False
            self.segment_allow = True
            self.file_mask_allow = True
        elif node_type in (NodeType.none, NodeType.measurement, NodeType.save):
            self.mask_allow = False
            self.segment_allow = False
            self.file_mask_allow = False
        self.save_activate()
        self.plan_node_changed.emit()

    def add_save_to_project(self):
        save_class = self.save_translate_dict.get(self.save_choose.currentText(), None)
        if save_class is None:
            QMessageBox.warning(self, "Save problem", "Not found save class")
        dial = FormDialog(
            [AlgorithmProperty("suffix", "File suffix", ""), AlgorithmProperty("directory", "Sub directory", "")]
            + save_class.get_fields()
        )
        if dial.exec():
            values = dial.get_values()
            suffix = values["suffix"]
            directory = values["directory"]
            del values["suffix"]
            del values["directory"]
            save_elem = Save(suffix, directory, save_class.get_name(), save_class.get_short_name(), values)
            if self.update_element_chk.isChecked():
                self.calculation_plan.replace_step(save_elem)
            else:
                self.calculation_plan.add_step(save_elem)
            self.plan.update_view()

    def create_mask(self):
        text = str(self.mask_name.text()).strip()

        if text != "" and text in self.mask_set:
            QMessageBox.warning(self, "Already exists", "Mask with this name already exists", QMessageBox.Ok)
            return
        if _check_widget(self.mask_stack, EnumComboBox):  # existing mask
            mask_dialog = TwoMaskDialog
            if self.mask_operation.get_value() == MaskOperation.mask_intersection:  # Mask intersection
                MaskConstruct = MaskIntersection
            else:
                MaskConstruct = MaskSum
            dial = mask_dialog(self.mask_set)
            if not dial.exec():
                return
            names = dial.get_result()

            mask_ob = MaskConstruct(text, *names)
        elif _check_widget(self.mask_stack, MaskWidget):
            mask_ob = MaskCreate(text, self.segmentation_mask.get_mask_property())
        elif _check_widget(self.mask_stack, FileMask):
            mask_ob = self.file_mask.get_value(text)
        else:
            raise ValueError("Unknowsn widget")

        if self.update_element_chk.isChecked():
            node = self.calculation_plan.get_node()
            name = node.operation.name
            if name in self.calculation_plan.get_reused_mask() and name != text:
                QMessageBox.warning(
                    self, "Cannot remove", f"Cannot remove mask '{name}' from plan because it is used in other elements"
                )
                return

            self.mask_set.remove(name)
            self.mask_set.add(mask_ob.name)
            self.calculation_plan.replace_step(mask_ob)
        else:
            self.mask_set.add(mask_ob.name)
            self.calculation_plan.add_step(mask_ob)
        self.plan.update_view()
        self.mask_text_changed()

    def mask_stack_change(self):
        node_type = self.calculation_plan.get_node_type()
        if self.update_element_chk.isChecked() and node_type not in [NodeType.mask, NodeType.file_mask]:
            self.generate_mask_btn.setDisabled(True)
        text = self.mask_name.text()
        update = self.update_element_chk.isChecked()
        if self.node_type == NodeType.none:
            self.generate_mask_btn.setDisabled(True)
            return
        operation = self.calculation_plan.get_node().operation
        if (
            not update
            and isinstance(operation, (MaskMapper, MaskBase))
            and self.calculation_plan.get_node().operation.name == text
        ):
            self.generate_mask_btn.setDisabled(True)
            return
        if _check_widget(self.mask_stack, EnumComboBox):  # reuse mask
            if len(self.mask_set) > 1 and (
                (not update and node_type == NodeType.root) or (update and node_type == NodeType.file_mask)
            ):
                self.generate_mask_btn.setEnabled(True)
            else:
                self.generate_mask_btn.setEnabled(False)
            self.generate_mask_btn.setToolTip("Need at least two named mask and root selected")
        elif _check_widget(self.mask_stack, MaskWidget):  # mask from segmentation
            if (not update and node_type == NodeType.segment) or (update and node_type == NodeType.mask):
                self.generate_mask_btn.setEnabled(True)
            else:
                self.generate_mask_btn.setEnabled(False)
            self.generate_mask_btn.setToolTip("Select segmentation")
        else:
            if (not update and node_type == NodeType.root) or (update and node_type == NodeType.file_mask):
                self.generate_mask_btn.setEnabled(self.file_mask.is_valid())
            else:
                self.generate_mask_btn.setEnabled(False)
            self.generate_mask_btn.setToolTip("Need root selected")

    def mask_name_changed(self, text):
        if str(text) in self.mask_set:
            self.generate_mask_btn.setDisabled(True)
        else:
            self.generate_mask_btn.setDisabled(False)

    def add_leave_biggest(self):
        profile = self.calculation_plan.get_node().operation
        profile.leave_biggest_swap()
        self.calculation_plan.replace_step(profile)
        self.plan.update_view()

    def add_segmentation(self):
        if self.segment_stack.currentIndex() == 0:
            text = str(self.segment_profile.currentItem().text())
            if text not in self.settings.segmentation_profiles:
                self.refresh_all_profiles()
                return
            profile = self.settings.segmentation_profiles[text]
            if self.update_element_chk.isChecked():
                self.calculation_plan.replace_step(profile)
            else:
                self.calculation_plan.add_step(profile)
        else:  # self.segment_stack.currentIndex() == 1
            text = self.pipeline_profile.currentItem().text()
            segmentation_pipeline = self.settings.segmentation_pipelines[text]
            pos = self.calculation_plan.current_pos[:]
            old_pos = self.calculation_plan.current_pos[:]
            for el in segmentation_pipeline.mask_history:
                self.calculation_plan.add_step(el.segmentation)
                self.plan.update_view()
                node = self.calculation_plan.get_node(pos)
                pos.append(len(node.children) - 1)
                self.calculation_plan.set_position(pos)
                self.calculation_plan.add_step(MaskCreate("", el.mask_property))
                self.plan.update_view()
                pos.append(0)
                self.calculation_plan.set_position(pos)
            self.calculation_plan.add_step(segmentation_pipeline.segmentation)
            self.calculation_plan.set_position(old_pos)

        self.plan.update_view()

    def add_measurement(self):
        text = str(self.measurements_list.currentItem().text())
        measurement_copy = deepcopy(self.settings.measurement_profiles[text])
        prefix = str(self.measurement_name_prefix.text()).strip()
        channel = self.choose_channel_for_measurements.currentIndex() - 1
        measurement_copy.name_prefix = prefix
        # noinspection PyTypeChecker
        measurement_calculate = MeasurementCalculate(
            channel=channel,
            measurement_profile=measurement_copy,
            name_prefix=prefix,
            units=self.units_choose.get_value(),
        )
        if self.update_element_chk.isChecked():
            self.calculation_plan.replace_step(measurement_calculate)
        else:
            self.calculation_plan.add_step(measurement_calculate)
        self.plan.update_view()

    def remove_element(self):
        conflict_mask, used_mask = self.calculation_plan.get_file_mask_names()
        if len(conflict_mask) > 0:
            logging.info("Mask in use")
            QMessageBox.warning(self, "In use", "Masks {} are used in other places".format(", ".join(conflict_mask)))
            return
        self.mask_set -= used_mask
        self.calculation_plan.remove_step()
        self.plan.update_view()

    def clean_plan(self):
        self.calculation_plan = CalculationPlan()
        self.plan.set_plan(self.calculation_plan)
        self.node_type_changed()
        self.mask_set = set()

    def mask_text_changed(self):
        name = str(self.mask_name.text()).strip()
        self.generate_mask_btn.setDisabled(True)
        # load mask from file
        if not self.update_element_chk.isChecked():
            # generate mask from segmentation
            if self.mask_allow and (name == "" or name not in self.mask_set):
                self.generate_mask_btn.setEnabled(True)
        else:
            if self.node_type not in [NodeType.file_mask, NodeType.mask]:
                return
            # generate mask from segmentation
            if self.node_type == NodeType.mask and (name == "" or name == self.node_name or name not in self.mask_set):
                self.generate_mask_btn.setEnabled(True)

    def add_calculation_plan(self, text=None):
        if text is None or isinstance(text, bool):
            text, ok = QInputDialog.getText(self, "Plan title", "Set plan title")
        else:
            text, ok = QInputDialog.getText(
                self, "Plan title", f"Set plan title. Previous ({text}) is already in use", text=text
            )
        text = text.strip()
        if ok:
            if text == "":
                QMessageBox.information(
                    self, "Name cannot be empty", "Name cannot be empty, Please set correct name", QMessageBox.Ok
                )
                self.add_calculation_plan()
                return
            if text in self.settings.batch_plans:
                res = QMessageBox.information(
                    self,
                    "Name already in use",
                    "Name already in use. Would like to overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if res == QMessageBox.No:
                    self.add_calculation_plan(text)
                    return
            plan = copy(self.calculation_plan)
            plan.set_name(text)
            self.settings.batch_plans[text] = plan
            self.settings.dump()
            self.plan_created.emit()

    @staticmethod
    def get_index(item: QListWidgetItem, new_values: typing.List[str]) -> int:
        if item is None:
            return -1
        text = item.text()
        try:
            return new_values.index(text)
        except ValueError:
            return -1

    @staticmethod
    def refresh_profiles(list_widget: QListWidget, new_values: typing.List[str], index: int):
        list_widget.clear()
        list_widget.addItems(new_values)
        if index != -1:
            list_widget.setCurrentRow(index)

    def showEvent(self, _event):
        self.refresh_all_profiles()

    def refresh_all_profiles(self):
        new_measurements = list(sorted(self.settings.measurement_profiles.keys()))
        new_segment = list(sorted(self.settings.segmentation_profiles.keys()))
        new_pipelines = list(sorted(self.settings.segmentation_pipelines.keys()))
        measurement_index = self.get_index(self.measurements_list.currentItem(), new_measurements)
        segment_index = self.get_index(self.segment_profile.currentItem(), new_segment)
        pipeline_index = self.get_index(self.pipeline_profile.currentItem(), new_pipelines)
        self.protect = True
        self.refresh_profiles(self.measurements_list, new_measurements, measurement_index)
        self.refresh_profiles(self.segment_profile, new_segment, segment_index)
        self.refresh_profiles(self.pipeline_profile, new_pipelines, pipeline_index)
        self.protect = False

    def show_measurement_info(self, text=None):
        if self.protect:
            return
        if text is None:
            if self.measurements_list.currentItem() is not None:
                text = str(self.measurements_list.currentItem().text())
            else:
                return
        profile = self.settings.measurement_profiles[text]
        self.information.setText(str(profile))

    def show_measurement(self):
        if self.update_element_chk.isChecked():
            if self.node_type == NodeType.measurement:
                self.add_calculation_btn.setEnabled(True)
            else:
                self.add_calculation_btn.setDisabled(True)
        else:
            if self.measurements_list.currentItem() is not None:
                self.add_calculation_btn.setEnabled(self.mask_allow)
            else:
                self.add_calculation_btn.setDisabled(True)

    def show_segment_info(self, text=None):
        if self.protect:
            return
        if text == "":
            return
        if self.segment_stack.currentIndex() == 0:
            if text is None:
                if self.segment_profile.currentItem() is not None:
                    text = str(self.segment_profile.currentItem().text())
                else:
                    return
            profile = self.settings.segmentation_profiles[text]
        else:
            if text is None:
                if self.pipeline_profile.currentItem() is not None:
                    text = str(self.pipeline_profile.currentItem().text())
                else:
                    return
            profile = self.settings.segmentation_pipelines[text]
        self.information.setText(profile.pretty_print(analysis_algorithm_dict))

    def show_segment(self):
        if self.update_element_chk.isChecked() and self.segment_stack.currentIndex() == 0:
            self.get_big_btn.setDisabled(True)
            if self.node_type == NodeType.segment:
                self.chose_profile_btn.setEnabled(True)
            else:
                self.chose_profile_btn.setDisabled(True)
        else:
            if self.node_type == NodeType.segment:
                self.get_big_btn.setEnabled(True)
            else:
                self.get_big_btn.setDisabled(True)
            if (
                self.segment_stack.currentIndex() == 0 and self.segment_profile.currentItem()
            ) or self.pipeline_profile.currentItem() is not None:
                self.chose_profile_btn.setEnabled(self.segment_allow)
            else:
                self.chose_profile_btn.setDisabled(True)

    def edit_plan(self):
        plan = self.sender().plan_to_edit  # type: CalculationPlan
        self.calculation_plan = copy(plan)
        self.plan.set_plan(self.calculation_plan)
        self.mask_set.clear()
        self.calculation_plan.set_position([])
        self.mask_set.update(self.calculation_plan.get_mask_names())


class PlanPreview(QTreeWidget):
    """
    :type calculation_plan: CalculationPlan
    """

    changed_node = Signal()

    def __init__(self, parent=None, calculation_plan=None):
        super().__init__(parent)
        self.calculation_plan = calculation_plan
        self.header().close()
        self.itemSelectionChanged.connect(self.set_path)

    def restore_path(self, widget, path):
        """
        :type widget: QTreeWidgetItem
        :type path: list[int]
        :param widget:
        :param path:
        :return:
        """
        if widget is None:
            return list(reversed(path))
        parent = widget.parent()
        if parent is None:
            return list(reversed(path))
        index = parent.indexOfChild(widget)
        if str(parent.child(0).text(0)) == "Description":
            index -= 1
        if index == -1:
            return None
        path.append(index)
        return self.restore_path(parent, path)

    def set_path(self):
        current_item = self.currentItem()  # type : QTreeWidgetItem
        if current_item is None:
            return
        self.calculation_plan.set_position(self.restore_path(current_item, []))
        self.changed_node.emit()

    def preview_object(self, calculation_plan):
        self.set_plan(calculation_plan)

    def set_plan(self, calculation_plan):
        self.calculation_plan = calculation_plan
        self.setCurrentItem(self.topLevelItem(0))
        self.update_view(True)

    def explore_tree(self, up_widget, node_plan, deep=True):
        """
        :type up_widget: QTreeWidgetItem
        :type node_plan: CalculationTree
        :type deep: bool
        :param up_widget: List widget item
        :param node_plan: node from calculation plan
        :return:
        """
        widget = QTreeWidgetItem(up_widget)
        widget.setText(0, CalculationPlan.get_el_name(node_plan.operation))
        self.setCurrentItem(widget)
        if isinstance(node_plan.operation, (MeasurementCalculate, ROIExtractionProfile, MaskCreate)):
            desc = QTreeWidgetItem(widget)
            desc.setText(0, "Description")
            if isinstance(node_plan.operation, ROIExtractionProfile):
                txt = node_plan.operation.pretty_print(analysis_algorithm_dict)
            else:
                txt = str(node_plan.operation)
            for line in txt.split("\n")[1:]:
                QTreeWidgetItem(desc, [line])
        if deep:
            for el in node_plan.children:
                self.explore_tree(widget, el)
        up_widget.setExpanded(True)

    def get_node(self, path):
        """
        :type path: list[int]
        :param path:
        :return: QTreeWidgetItem
        """
        widget = self.topLevelItem(0)  # type : QTreeWidgetItem
        for index in path:
            if str(widget.child(0).text(0)) == "Description":
                index += 1
            widget = widget.child(index)
        return widget

    def update_view(self, reset=False):
        if reset:
            self.clear()
            root = QTreeWidgetItem(self)
            root.setText(0, f"Root {self.calculation_plan.get_root_type()}")
            self.setCurrentItem(root)
            for el in self.calculation_plan.execution_tree.children:
                self.explore_tree(root, el, True)
            return
        self.blockSignals(True)
        root = self.get_node([])
        root.setText(0, f"Root {self.calculation_plan.get_root_type()}")
        for path, el, op_type in self.calculation_plan.get_changes():
            if op_type == PlanChanges.add_node:
                node = self.get_node(path)
                self.explore_tree(node, el, False)
            elif op_type == PlanChanges.remove_node:
                node = self.get_node(path[:-1])
                index = path[-1]
                if str(node.child(0).text(0)) == "Description":
                    index += 1
                node.removeChild(node.child(index))
            elif op_type == PlanChanges.replace_node:
                node = self.get_node(path)
                node.setText(0, CalculationPlan.get_el_name(el.operation))
                if isinstance(el.operation, (MeasurementCalculate, ROIExtractionProfile, MaskCreate)):
                    child = node.child(0)
                    child.takeChildren()
                    if isinstance(el.operation, ROIExtractionProfile):
                        txt = el.operation.pretty_print(analysis_algorithm_dict)
                    else:
                        txt = str(el.operation)
                    for line in txt.split("\n")[1:]:
                        QTreeWidgetItem(child, [line])

            else:
                logging.error(f"Unknown operation {op_type}")
        self.blockSignals(False)
        self.set_path()
        self.changed_node.emit()


class CalculateInfo(QWidget):
    """
    "widget to show information about plans and allow to se plan details
    :type settings: Settings
    """

    plan_to_edit_signal = Signal()

    def __init__(self, settings: PartSettings):
        super().__init__()
        self.settings = settings
        self.calculate_plans = SearchableListWidget(self)
        self.plan_view = PlanPreview(self)
        self.delete_plan_btn = QPushButton("Delete")
        self.edit_plan_btn = QPushButton("Edit")
        self.export_plans_btn = QPushButton("Export")
        self.import_plans_btn = QPushButton("Import")
        info_layout = QVBoxLayout()
        info_butt_layout = QGridLayout()
        info_butt_layout.setSpacing(1)
        info_butt_layout.addWidget(self.delete_plan_btn, 1, 1)
        info_butt_layout.addWidget(self.edit_plan_btn, 0, 1)
        info_butt_layout.addWidget(self.export_plans_btn, 1, 0)
        info_butt_layout.addWidget(self.import_plans_btn, 0, 0)
        info_layout.addLayout(info_butt_layout)
        info_chose_layout = QVBoxLayout()
        info_chose_layout.setSpacing(2)
        info_chose_layout.addWidget(QLabel("List of workflows:"))
        info_chose_layout.addWidget(self.calculate_plans)
        info_chose_layout.addWidget(QLabel("Preview:"))
        info_chose_layout.addWidget(self.plan_view)
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
        choose = ExportDialog(self.settings.batch_plans, PlanPreview)
        if not choose.exec_():
            return
        dial = QFileDialog(self, "Export calculation plans")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setDirectory(dial.setDirectory(self.settings.get("io.batch_plan_directory", str(Path.home()))))
        dial.setNameFilter("Calculation plans (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("calculation_plans.json")
        dial.setHistory(dial.history() + self.settings.get_path_history())
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.set("io.batch_plan_directory", os.path.dirname(file_path))
            self.settings.add_path_history(os.path.dirname(file_path))
            data = {x: self.settings.batch_plans[x] for x in choose.get_export_list()}
            with open(file_path, "w") as ff:
                json.dump(data, ff, cls=self.settings.json_encoder_class, indent=2)

    def import_plans(self):
        dial = QFileDialog(self, "Import calculation plans")
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        dial.setDirectory(self.settings.get("io.open_directory", str(Path.home())))
        dial.setNameFilter("Calculation plans (*.json)")
        dial.setDefaultSuffix("json")
        dial.setHistory(dial.history() + self.settings.get_path_history())
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            plans, err = self.settings.load_part(file_path)
            self.settings.set("io.batch_plan_directory", os.path.dirname(file_path))
            self.settings.add_path_history(os.path.dirname(file_path))
            if err:
                QMessageBox.warning(self, "Import error", "error during importing, part of data were filtered.")
            choose = ImportDialog(plans, self.settings.batch_plans, PlanPreview)
            if choose.exec_():
                for original_name, final_name in choose.get_import_list():
                    self.settings.batch_plans[final_name] = plans[original_name]
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
        self.plan_view.set_plan(plan)


class CalculatePlaner(QSplitter):
    """
    :type settings: Settings
    """

    def __init__(self, settings, parent):
        super().__init__(parent)
        self.settings = settings
        self.info_widget = CalculateInfo(settings)
        self.addWidget(self.info_widget)
        self.create_plan = CreatePlan(settings)
        self.create_plan.plan_created.connect(self.info_widget.update_plan_list)
        self.info_widget.plan_to_edit_signal.connect(self.create_plan.edit_plan)
        self.addWidget(self.create_plan)
