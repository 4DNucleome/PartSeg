import json
import logging
import os
import typing
from contextlib import contextmanager
from copy import copy, deepcopy
from enum import Enum

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAction,
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
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt import QEnumComboBox

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg._roi_analysis.profile_export import ExportDialog, ImportDialog
from PartSeg.common_backend.except_hook import show_warning
from PartSeg.common_gui.custom_load_dialog import PLoadDialog
from PartSeg.common_gui.custom_save_dialog import FormDialog, PSaveDialog
from PartSeg.common_gui.mask_widget import MaskWidget
from PartSeg.common_gui.searchable_list_widget import SearchableListWidget
from PartSeg.common_gui.universal_gui_part import right_label
from PartSegCore.algorithm_describe_base import AlgorithmProperty, ROIExtractionProfile
from PartSegCore.analysis import SegmentationPipeline
from PartSegCore.analysis.algorithm_description import AnalysisAlgorithmSelection
from PartSegCore.analysis.calculation_plan import (
    CalculationPlan,
    MaskBase,
    MaskCreate,
    MaskFile,
    MaskIntersection,
    MaskSub,
    MaskSuffix,
    MaskSum,
    MeasurementCalculate,
    NodeType,
    PlanChanges,
    RootType,
    Save,
)
from PartSegCore.analysis.measurement_calculation import MeasurementProfile
from PartSegCore.analysis.save_functions import save_dict
from PartSegCore.io_utils import LoadPlanExcel, LoadPlanJson, SaveBase
from PartSegCore.universal_const import Units
from PartSegData import icons_dir

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
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
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
        if not text1 or text1 not in self.mask_names:
            self.ok_btn.setDisabled(True)
        else:
            self.ok_btn.setDisabled(False)

    def get_result(self):
        text1 = str(self.mask1_name.text()).strip()
        return (text1,)


class TwoMaskDialog(QDialog):
    def __init__(self, mask_names: typing.Iterable[str]):
        """
        :param mask_names: iterable collection of all available mask names
        """
        super().__init__()
        self.mask_names = mask_names
        completer = QCompleter(list(mask_names))
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
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
        if "" in {text1, text2} or text1 not in self.mask_names or text2 not in self.mask_names:
            self.ok_btn.setDisabled(True)
        else:
            self.ok_btn.setDisabled(text1 == text2)

    def get_result(self):
        text1 = str(self.mask1_name.text()).strip()
        text2 = str(self.mask2_name.text()).strip()
        return text1, text2


class FileMaskType(Enum):
    """
    Enum for file mask types
    """

    Suffix = 0
    Replace = 1
    Mapping_file = 2


class FileMask(QWidget):
    value_changed = Signal()

    def __init__(self):
        super().__init__()
        self.select_type = QEnumComboBox(enum_class=FileMaskType)
        self.values = ["_mask", ("", ""), ""]
        self.first_text = QLineEdit(self.values[0])
        self.second_text = QLineEdit()
        self.first_label = QLabel("Use suffix:")
        self.second_label = QLabel("Replace:")
        self.select_file_btn = QPushButton("Select file")
        self.state = FileMaskType.Suffix

        layout = QGridLayout()
        layout.addWidget(self.select_type, 0, 0, 1, 2)
        layout.addWidget(self.first_label, 1, 0)
        layout.addWidget(self.second_label, 1, 1)
        layout.addWidget(self.first_text, 2, 0)
        layout.addWidget(self.second_text, 2, 1)
        layout.addWidget(self.select_file_btn, 3, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        layout.setRowStretch(4, 1)
        self.setLayout(layout)

        self.second_text.setHidden(True)
        self.second_label.setHidden(True)
        self.select_file_btn.setHidden(True)

        self.first_text.textChanged.connect(self._value_change_wrap)
        self.second_text.textChanged.connect(self._value_change_wrap)
        self.select_type.currentIndexChanged.connect(self._value_change_wrap)
        self.select_type.currentEnumChanged.connect(self.change_type)
        self.select_file_btn.clicked.connect(self.select_file)

    def _value_change_wrap(self, _val=None):
        """Pyside bug workaround"""
        self.value_changed.emit()

    def change_type(self, index: FileMaskType):
        if self.state == index:
            return
        if self.state == FileMaskType.Replace:
            self.values[1] = self.first_text.text(), self.second_text.text()
        else:
            self.values[self.state.value] = self.first_text.text()
        if index == FileMaskType.Replace:
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
            self.first_label.setText(["Use suffix:", "", "Path:"][index.value])
            self.first_text.setText(self.values[index.value])
            self.select_file_btn.setHidden(index == 0)
        self.state = index

    def select_file(self):
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)

        if dial.exec_():
            self.first_text.setText(dial.selectedFiles()[0])

    def is_valid(self) -> bool:
        if self.select_type.currentEnum() == FileMaskType.Suffix:
            return bool(self.first_text.text().strip())
        if self.select_type.currentEnum() == FileMaskType.Replace:
            return "" not in {self.first_text.text().strip(), self.second_text.text().strip()}

        text = self.first_text.text().strip()
        return text and os.path.exists(text) and os.path.isfile(text)

    def get_value(self, name=""):
        mask_type = self.select_type.currentEnum()
        if mask_type == FileMaskType.Suffix:
            return MaskSuffix(name=name, suffix=self.first_text.text().strip())
        if mask_type == FileMaskType.Replace:
            return MaskSub(name=name, base=self.first_text.text().strip(), rep=self.second_text.text().strip())
        return MaskFile(name=name, path_to_file=self.first_text.text().strip())


class MaskOperation(Enum):
    mask_intersection = 0
    mask_sum = 1

    def __str__(self):
        return self.name.replace("_", " ").capitalize()


class ProtectedGroupBox(QGroupBox):
    def __init__(self, text: str, parent: typing.Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setStyleSheet(group_sheet)
        self.protect = False
        self._node_type = None
        self._parent_node_type = None
        self._replace = False

    def set_current_node(self, node: typing.Optional[NodeType], parent_node: typing.Optional[NodeType] = None):
        self._node_type = node
        self._parent_node_type = parent_node
        self._activate_button()

    def set_replace(self, replace: bool):
        self._replace = replace
        self._activate_button()

    def _activate_button(self, _value=None):
        raise NotImplementedError

    @contextmanager
    def enable_protect(self):
        previous = self.protect
        self.protect = True
        try:
            yield
        finally:
            self.protect = previous

    @classmethod
    def refresh_profiles(
        cls, list_widget: typing.Union[QListWidget, SearchableListWidget], new_values: typing.List[str]
    ):
        index = cls.get_index(list_widget.currentItem(), new_values)
        list_widget.clear()
        list_widget.addItems(new_values)
        if index != -1:
            list_widget.setCurrentRow(index)

    @staticmethod
    def get_index(item: QListWidgetItem, new_values: typing.List[str]) -> int:
        if item is None:
            return -1
        text = item.text()
        try:
            return new_values.index(text)
        except ValueError:  # pragma: no cover
            return -1


class OtherOperations(ProtectedGroupBox):
    save_operation = Signal(object)

    def __init__(self, parent=None):
        super().__init__("Other operations:", parent)
        self.save_translate_dict: typing.Dict[str, SaveBase] = {x.get_short_name(): x for x in save_dict.values()}
        self.save_constructor = None

        self.change_root = QEnumComboBox(self, enum_class=RootType)
        self.choose_save_method = QComboBox()
        self.choose_save_method.addItem("<none>")
        self.choose_save_method.addItems(list(self.save_translate_dict.keys()))
        self.save_btn = QPushButton("Save")

        self.choose_save_method.currentTextChanged.connect(self.save_changed)
        self.save_btn.clicked.connect(self.save_action)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(QLabel("Root type:"))
        layout.addWidget(self.change_root)
        layout.addStretch(1)
        layout.addWidget(QLabel("Saving:"))
        layout.addWidget(self.choose_save_method)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    @property
    def root_type_changed(self):
        return self.change_root.currentEnumChanged

    def save_changed(self, text):
        text = str(text)
        save_class = self.save_translate_dict.get(text, None)
        if save_class is None:
            self.choose_save_method.setCurrentText("<none>")
            self.save_btn.setText("Save")
            self.save_btn.setToolTip("Choose file type")
        else:
            self.save_btn.setText(f"Save to {save_class.get_short_name()}")
            self.save_btn.setToolTip("Choose mask create in plan view")
        self._activate_button()

    @property
    def expected_node_type(self) -> typing.Optional[NodeType]:
        save_class = self.save_translate_dict.get(self.choose_save_method.currentText(), None)
        if save_class is None:
            return None
        if save_class.need_mask():
            return NodeType.mask
        return NodeType.segment if save_class.need_segmentation() else NodeType.root

    def _activate_button(self, _value=None):
        if self._replace:
            self.save_btn.setEnabled(self._parent_node_type == self.expected_node_type and self._node_type is not None)
        else:
            self.save_btn.setEnabled(self._node_type == self.expected_node_type and self._node_type is not None)

    def save_action(self):
        save_class = self.save_translate_dict.get(self.choose_save_method.currentText(), None)
        if save_class is None:  # pragma: no cover
            show_warning(self, "Save problem", "Not found save class")
            return
        dial = FormDialog(
            [
                AlgorithmProperty("suffix", "File suffix", ""),
                AlgorithmProperty("directory", "Sub directory", ""),
                *save_class.get_fields(),
            ]
        )
        if not dial.exec_():
            return
        values = dial.get_values()
        suffix = values["suffix"]
        directory = values["directory"]
        del values["suffix"]
        del values["directory"]
        save_elem = Save(
            suffix=suffix,
            directory=directory,
            algorithm=save_class.get_name(),
            short_name=save_class.get_short_name(),
            values=values,
        )
        self.save_operation.emit(save_elem)


class ROIExtractionOp(ProtectedGroupBox):
    roi_extraction_profile_selected = Signal(object)
    roi_extraction_pipeline_selected = Signal(object)
    roi_extraction_profile_add = Signal(object)
    roi_extraction_pipeline_add = Signal(object)

    def __init__(self, settings: PartSettings, parent: typing.Optional[QWidget] = None):
        super().__init__("ROI extraction", parent)
        self.settings = settings

        self.roi_profile = SearchableListWidget()
        self.roi_pipeline = SearchableListWidget()
        self.roi_extraction_tab = QTabWidget()
        self.roi_extraction_tab.addTab(self.roi_profile, "Profile")
        self.roi_extraction_tab.addTab(self.roi_pipeline, "Pipeline")

        self.choose_profile_btn = QPushButton("Add Profile")
        self.choose_profile_btn.setDisabled(True)

        self.settings.roi_profiles_changed.connect(self._refresh_profiles)
        self.settings.roi_pipelines_changed.connect(self._refresh_pipelines)

        self.roi_profile.currentTextChanged.connect(self._roi_extraction_profile_selected)
        self.roi_pipeline.currentTextChanged.connect(self._roi_extraction_pipeline_selected)
        self.choose_profile_btn.clicked.connect(self._add_profile)
        self.roi_extraction_tab.currentChanged.connect(self._on_change_tab)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.roi_extraction_tab)
        layout.addWidget(self.choose_profile_btn)

        self.setLayout(layout)

        self._refresh_profiles()
        self._refresh_pipelines()
        self._update_btn_text()
        self.settings.roi_profiles_changed.connect(self._refresh_profiles)
        self.settings.roi_pipelines_changed.connect(self._refresh_pipelines)

    def set_replace(self, replace: bool):
        super().set_replace(replace)
        self._on_change_tab()

    def _activate_button(self, _value=None):
        if self._replace:
            self.choose_profile_btn.setEnabled(
                self._node_type == NodeType.segment
                and self.roi_extraction_tab.currentWidget() == self.roi_profile
                and self.roi_profile.currentItem() is not None
            )
            return
        self.choose_profile_btn.setEnabled(
            self._node_type in {NodeType.root, NodeType.mask, NodeType.file_mask}
            and self.roi_extraction_tab.currentWidget().currentRow() >= 0
        )

    def _update_btn_text(self):
        index = self.roi_extraction_tab.currentIndex()
        text = self.roi_extraction_tab.tabText(index)
        if self._replace:
            self.choose_profile_btn.setText(f"Replace {text}")
        else:
            self.choose_profile_btn.setText(f"Add {text}")

    def _on_change_tab(self, _val=None):
        self._update_btn_text()
        with self.enable_protect():
            self.roi_profile.setCurrentItem(None)
            self.roi_pipeline.setCurrentItem(None)
        self._activate_button()

    def _refresh_profiles(self):
        new_profiles = sorted(self.settings.roi_profiles.keys(), key=str.lower)
        with self.enable_protect():
            self.refresh_profiles(self.roi_profile, new_profiles)

    def _refresh_pipelines(self):
        new_pipelines = sorted(self.settings.roi_pipelines.keys(), key=str.lower)
        with self.enable_protect():
            self.refresh_profiles(self.roi_pipeline, new_pipelines)

    def _roi_extraction_profile_selected(self, name: str):
        if self.protect:
            return
        self._activate_button()
        self.roi_extraction_profile_selected.emit(self.settings.roi_profiles[name])

    def _roi_extraction_pipeline_selected(self, name: str):
        if self.protect:
            return
        self._activate_button()
        self.roi_extraction_pipeline_selected.emit(self.settings.roi_pipelines[name])

    def _add_profile(self):
        if self.roi_extraction_tab.currentWidget() == self.roi_profile:
            item = self.roi_profile.currentItem()
            if item is None:
                return
            self.roi_extraction_profile_add.emit(deepcopy(self.settings.roi_profiles[item.text()]))
        else:
            item = self.roi_pipeline.currentItem()
            if item is None:
                return
            self.roi_extraction_pipeline_add.emit(deepcopy(self.settings.roi_pipelines[item.text()]))


class SelectMeasurementOp(ProtectedGroupBox):
    set_of_measurement_add = Signal(object)
    set_of_measurement_selected = Signal(object)

    def __init__(self, settings: PartSettings, parent: typing.Optional[QWidget] = None):
        super().__init__("Set of measurements:", parent)
        self.settings = settings

        self.measurements_list = SearchableListWidget(self)
        self.measurement_name_prefix = QLineEdit(self)
        self.choose_channel_for_measurements = QComboBox()
        self.choose_channel_for_measurements.addItems(
            ["Same as segmentation"] + [str(x + 1) for x in range(MAX_CHANNEL_NUM)]
        )
        self.units_choose = QEnumComboBox(enum_class=Units)
        self.units_choose.setCurrentEnum(self.settings.get("units_value", Units.nm))
        self.add_measurement_btn = QPushButton("Add measurement calculation")
        self.add_measurement_btn.clicked.connect(self._measurement_add)
        self.measurements_list.currentTextChanged.connect(self._measurement_selected)

        layout = QGridLayout()
        layout.setSpacing(0)
        layout.addWidget(self.measurements_list, 0, 0, 1, 2)
        lab = QLabel("Name prefix:")
        lab.setToolTip("Prefix added before each column name")
        layout.addWidget(lab, 1, 0)
        layout.addWidget(self.measurement_name_prefix, 1, 1)
        layout.addWidget(QLabel("Channel:"), 2, 0)
        layout.addWidget(self.choose_channel_for_measurements, 2, 1)
        layout.addWidget(QLabel("Units:"), 3, 0)
        layout.addWidget(self.units_choose, 3, 1)
        layout.addWidget(self.add_measurement_btn, 4, 0, 1, 2)
        self.setLayout(layout)

        self.add_measurement_btn.setDisabled(True)
        self._refresh_measurement()
        self.settings.measurement_profiles_changed.connect(self._refresh_measurement)

    def set_replace(self, replace: bool):
        super().set_replace(replace)
        self.add_measurement_btn.setText("Replace set of measurements" if self._replace else "Add set of measurements")

    def _activate_button(self, _value=None):
        if self._replace:
            self.add_measurement_btn.setEnabled(
                self._node_type == NodeType.measurement and self.measurements_list.currentItem() is not None
            )
        else:
            self.add_measurement_btn.setEnabled(
                self._node_type == NodeType.segment and self.measurements_list.currentItem() is not None
            )

    def _refresh_measurement(self):
        new_measurements = sorted(self.settings.measurement_profiles.keys(), key=str.lower)
        with self.enable_protect():
            self.refresh_profiles(self.measurements_list, new_measurements)

    def _measurement_add(self):
        item = self.measurements_list.currentItem()
        if item is None:
            return
        measurement_copy = deepcopy(self.settings.measurement_profiles[item.text()])
        prefix = str(self.measurement_name_prefix.text()).strip()
        channel = self.choose_channel_for_measurements.currentIndex() - 1
        measurement_copy.name_prefix = prefix
        self.set_of_measurement_add.emit(
            MeasurementCalculate(
                channel=channel,
                measurement_profile=measurement_copy,
                name_prefix=prefix,
                units=self.units_choose.currentEnum(),
            )
        )

    def _measurement_selected(self, name: str):
        if self.protect:
            return
        self._activate_button()
        self.set_of_measurement_selected.emit(self.settings.measurement_profiles[name])


class StretchWrap(QWidget):
    def __init__(self, widget: QWidget, parent: typing.Optional[QWidget] = None):
        super().__init__(parent)
        self.widget = widget
        lay = QVBoxLayout()
        lay.setSpacing(0)
        lay.addWidget(widget)
        lay.addStretch(1)
        self.setLayout(lay)

    def __getattr__(self, item):
        return getattr(self.widget, item)


class SelectMaskOp(ProtectedGroupBox):
    mask_step_add = Signal(object)

    def __init__(self, settings: PartSettings, parent: typing.Optional[QWidget] = None):
        super().__init__("Use mask from:", parent)
        self.settings = settings
        self.mask_set = {}

        self.file_mask = FileMask()
        self.mask_from_segmentation = MaskWidget(settings)
        self.mask_operation = StretchWrap(QEnumComboBox(enum_class=MaskOperation))
        self.add_mask_btn = QPushButton("Add mask")
        self.add_mask_btn.setToolTip("Mask need to have unique name")
        self.add_mask_btn.clicked.connect(self._add_mask)
        self.mask_name = QLineEdit()

        self.mask_tab_select = QTabWidget()

        self.mask_tab_select.addTab(self.file_mask, "File")
        self.mask_tab_select.addTab(self.mask_from_segmentation, "Current ROI")
        self.mask_tab_select.addTab(self.mask_operation, "Operations on masks")
        self.mask_tab_select.setTabToolTip(2, "Allows to create mask which is based on masks previously added to plan.")
        self.mask_tab_select.currentChanged.connect(self._activate_button)
        self.mask_name.textChanged.connect(self._activate_button)

        layout = QGridLayout()
        layout.setSpacing(0)
        layout.addWidget(self.mask_tab_select, 0, 0, 1, 2)
        label = QLabel("Mask name:")
        label.setToolTip("Needed if you would like to reuse this mask in tab 'Operations on masks'")
        self.mask_name.setToolTip("Needed if you would like to reuse this mask in tab 'Operations on masks'")
        layout.addWidget(label, 1, 0)
        layout.addWidget(self.mask_name, 1, 1)
        layout.addWidget(self.add_mask_btn, 2, 0, 1, 2)
        self.setLayout(layout)

        self.add_mask_btn.setDisabled(True)

    def update_mask_set(self, mask_set: typing.Set[str]):
        self.mask_set = mask_set

    def set_replace(self, replace: bool):
        super().set_replace(replace)
        self.add_mask_btn.setText("Replace mask" if self._replace else "Add mask")

    def _activate_button(self, _value=None):
        name = self.mask_name.text().strip()
        name_ok = not name or name not in self.mask_set
        if self._replace:
            name_ok = name_ok and self._node_type == NodeType.mask
            node_type = self._parent_node_type
        else:
            node_type = self._node_type
        if self.mask_tab_select.currentWidget() == self.mask_from_segmentation:
            self.add_mask_btn.setEnabled(node_type == NodeType.segment and name_ok)
            return
        self.add_mask_btn.setEnabled(node_type == NodeType.root and name_ok)

    def _add_mask(self):
        widget = self.mask_tab_select.currentWidget()
        name = self.mask_name.text().strip()
        if widget == self.file_mask:
            mask_ob = self.file_mask.get_value(name)
        elif widget == self.mask_from_segmentation:
            mask_ob = MaskCreate(name=name, mask_property=self.mask_from_segmentation.get_mask_property())
        elif widget == self.mask_operation:
            dial = TwoMaskDialog(self.mask_set)
            if not dial.exec_():
                return  # pragma: no cover
            names = dial.get_result()

            if self.mask_operation.currentEnum() == MaskOperation.mask_intersection:  # Mask intersection
                mask_construct = MaskIntersection
            else:
                mask_construct = MaskSum
            mask_ob = mask_construct(name=name, mask1=names[0], mask2=names[1])
        else:
            raise ValueError("Unknown widget")  # pragma: no cover

        self.mask_step_add.emit(mask_ob)


class CreatePlan(QWidget):
    plan_node_changed = Signal()

    def __init__(self, settings: PartSettings):
        super().__init__()
        self.settings = settings
        self.save_translate_dict: typing.Dict[str, SaveBase] = {x.get_short_name(): x for x in save_dict.values()}
        self._mask_set = set()
        self.plan = PlanPreview(self)
        self.save_plan_btn = QPushButton("Save")
        self.clean_plan_btn = QPushButton("Remove all")
        self.remove_btn = QPushButton("Remove")
        self.update_element_chk = QCheckBox("Update element")
        self.other_operations = OtherOperations(self)
        self.roi_extraction = ROIExtractionOp(settings=settings, parent=self)
        self.select_measurement = SelectMeasurementOp(settings=settings, parent=self)
        self.select_mask = SelectMaskOp(settings=settings, parent=self)
        self.mask_set = set()

        self.expected_node_type = None
        self.save_constructor = None

        self.information = QTextEdit()
        self.information.setReadOnly(True)

        # FIXME: fix in better way
        self.calculation_plan = CalculationPlan()
        self.plan.set_plan(self.calculation_plan)
        self.segmentation_mask = MaskWidget(settings)
        self.file_mask = FileMask()

        self.other_operations.root_type_changed.connect(self.change_root_type)
        self.other_operations.save_operation.connect(self.add_save_operation)
        self.roi_extraction.roi_extraction_pipeline_selected.connect(self.show_info)
        self.roi_extraction.roi_extraction_profile_selected.connect(self.show_info)
        self.roi_extraction.roi_extraction_profile_add.connect(self.add_roi_extraction)
        self.roi_extraction.roi_extraction_pipeline_add.connect(self.add_roi_extraction_pipeline)
        self.select_measurement.set_of_measurement_add.connect(self.add_set_of_measurement)
        self.select_measurement.set_of_measurement_selected.connect(self.show_info)
        self.select_mask.mask_step_add.connect(self.create_mask)

        self.clean_plan_btn.clicked.connect(self.clean_plan)
        self.remove_btn.clicked.connect(self.remove_element)
        self.save_plan_btn.clicked.connect(self.add_calculation_plan)
        self.update_element_chk.stateChanged.connect(self.select_mask.set_replace)
        self.update_element_chk.stateChanged.connect(self.roi_extraction.set_replace)
        self.update_element_chk.stateChanged.connect(self.select_measurement.set_replace)

        self.setup_ui()

        self.node_type = NodeType.root
        self.node_name = ""
        self.plan.changed_node.connect(self.node_type_changed)
        self.node_type_changed()

    def setup_ui(self):
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

        info_box = QGroupBox("Information")
        info_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.information)
        info_box.setLayout(lay)

        layout = QGridLayout()
        layout.addWidget(plan_box, 0, 0, 5, 1)
        layout.addWidget(self.select_mask, 0, 2, 1, 2)
        layout.addWidget(self.other_operations, 0, 1)
        layout.addWidget(self.roi_extraction, 1, 1, 1, 2)
        layout.addWidget(self.select_measurement, 1, 3)
        layout.addWidget(info_box, 3, 1, 1, 3)
        self.setLayout(layout)

    @property
    def mask_set(self):
        return self._mask_set

    @mask_set.setter
    def mask_set(self, value):
        self._mask_set = value
        self.select_mask.update_mask_set(value)

    def change_root_type(self, root_type: RootType):
        self.calculation_plan.set_root_type(root_type)
        self.plan.update_view()

    def add_save_operation(self, save_info: Save):
        if self.update_element_chk.isChecked():
            self.calculation_plan.replace_step(save_info)
        else:
            self.calculation_plan.add_step(save_info)
        self.plan.update_view()

    def add_set_of_measurement(self, set_of_measurement: MeasurementCalculate):
        if self.update_element_chk.isChecked():
            self.calculation_plan.replace_step(set_of_measurement)
        else:
            self.calculation_plan.add_step(set_of_measurement)
        self.plan.update_view()

    def node_type_changed(self):
        self.node_name = ""
        if self.plan.currentItem() is None:
            self.remove_btn.setDisabled(True)
            self.plan_node_changed.emit()
            logging.debug("[node_type_changed] return")
            return
        node_type = self.calculation_plan.get_node_type()

        node_type_for_ob = self.calculation_plan.get_node_type(parent=self.update_element_chk.isChecked())

        self.other_operations.set_current_node(node_type, node_type_for_ob)
        self.roi_extraction.set_current_node(node_type, node_type_for_ob)
        self.select_measurement.set_current_node(node_type, node_type_for_ob)
        self.select_mask.set_current_node(node_type, node_type_for_ob)

        self.node_type = node_type
        self.plan_node_changed.emit()

    def create_mask(self, mask_ob: MaskBase):
        if mask_ob.name and mask_ob.name in self.mask_set:
            show_warning("Already exists", "Mask with this name already exists")
            return

        if self.update_element_chk.isChecked():
            node = self.calculation_plan.get_node()
            name = node.operation.name
            if name in self.calculation_plan.get_reused_mask() and name != mask_ob.name:
                show_warning(
                    "Cannot remove", f"Cannot remove mask '{name}' from plan because it is used in other elements"
                )
                return

            self.mask_set.remove(name)
            self.mask_set.add(mask_ob.name)
            self.calculation_plan.replace_step(mask_ob)
        else:
            self.mask_set.add(mask_ob.name)
            self.calculation_plan.add_step(mask_ob)
        self.plan.update_view()

    def add_roi_extraction(self, roi_extraction: ROIExtractionOp):
        if self.update_element_chk.isChecked():
            self.calculation_plan.replace_step(roi_extraction)
        else:
            self.calculation_plan.add_step(roi_extraction)
        self.plan.update_view()

    def add_roi_extraction_pipeline(self, roi_extraction_pipeline: SegmentationPipeline):
        if self.update_element_chk.isChecked():
            show_warning("Cannot update pipeline", "Cannot update pipeline")
            return
        pos = self.calculation_plan.current_pos[:]
        old_pos = pos[:]
        for el in roi_extraction_pipeline.mask_history:
            self.calculation_plan.add_step(el.segmentation)
            self.plan.update_view()
            node = self.calculation_plan.get_node(pos)
            pos.append(len(node.children) - 1)
            self.calculation_plan.set_position(pos)
            self.calculation_plan.add_step(MaskCreate(name="", mask_property=el.mask_property))
            self.plan.update_view()
            pos.append(0)
            self.calculation_plan.set_position(pos)
        self.calculation_plan.add_step(roi_extraction_pipeline.segmentation)
        self.calculation_plan.set_position(old_pos)
        self.plan.update_view()

    def remove_element(self):
        conflict_mask, used_mask = self.calculation_plan.get_file_mask_names()
        if len(conflict_mask) > 0:
            logging.info("Mask in use")
            show_warning("In use", f'Masks {", ".join(conflict_mask)} are used in other places')

            return
        self.mask_set -= used_mask
        self.calculation_plan.remove_step()
        self.plan.update_view()

    def clean_plan(self):
        self.calculation_plan = CalculationPlan()
        self.plan.set_plan(self.calculation_plan)
        self.node_type_changed()
        self.mask_set = set()

    def add_calculation_plan(self, text=None):
        if text is None or isinstance(text, bool):
            text, ok = QInputDialog.getText(self, "Plan title", "Set plan title")
        else:
            text, ok = QInputDialog.getText(
                self, "Plan title", f"Set plan title. Previous ({text}) is already in use", text=text
            )
        text = text.strip()
        if ok:
            if not text:
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

    def show_info(self, item: typing.Union[ROIExtractionOp, SegmentationPipeline, MeasurementProfile]):
        if isinstance(item, (ROIExtractionOp, MeasurementProfile)):
            self.information.setText(str(item))
        else:
            self.information.setText(item.pretty_print(AnalysisAlgorithmSelection))

    def edit_plan(self, plan: CalculationPlan):
        if plan.is_bad():
            QMessageBox().warning(
                self, "Cannot edit broken plan", f"Cannot edit broken plan. {plan.get_error_source()}"
            )
            return
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
        self.calculation_plan = None
        self.header().close()
        self.itemSelectionChanged.connect(self.set_path)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        if calculation_plan is not None:
            self.set_plan(calculation_plan)

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
        if calculation_plan is not None and calculation_plan.is_bad():
            QMessageBox().warning(
                self, "Cannot preview broken plan", f"Cannot preview broken plan. {calculation_plan.get_error_source()}"
            )
            return
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
        if isinstance(node_plan.operation, (MeasurementCalculate, ROIExtractionProfile)):
            widget.setData(0, Qt.ItemDataRole.UserRole, node_plan.operation)
        if isinstance(node_plan.operation, (MeasurementCalculate, ROIExtractionProfile, MaskCreate)):
            desc = QTreeWidgetItem(widget)
            desc.setText(0, "Description")
            if isinstance(node_plan.operation, ROIExtractionProfile):
                txt = node_plan.operation.pretty_print(AnalysisAlgorithmSelection)
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
                widget = widget.child(index + 1)
            else:
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
                        txt = el.operation.pretty_print(AnalysisAlgorithmSelection)
                    else:
                        txt = str(el.operation)
                    for line in txt.split("\n")[1:]:
                        QTreeWidgetItem(child, [line])

            else:
                logging.error("Unknown operation %s", op_type)  # pragma: no cover
        self.blockSignals(False)
        self.set_path()
        self.changed_node.emit()


class CalculateInfo(QWidget):
    """
    "widget to show information about plans and allow to se plan details
    :type settings: Settings
    """

    plan_to_edit_signal = Signal(object)

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
        self.protect = False
        self.plan_to_edit = None

        self.plan_view.header().close()
        self.calculate_plans.currentTextChanged.connect(self.plan_preview)
        self.delete_plan_btn.clicked.connect(self.delete_plan)
        self.edit_plan_btn.clicked.connect(self.edit_plan)
        self.export_plans_btn.clicked.connect(self.export_plans)
        self.import_plans_btn.clicked.connect(self.import_plans)
        self.settings.batch_plans_changed.connect(self.update_plan_list)
        self.plan_view.customContextMenuRequested.connect(self._context_menu)
        self.update_plan_list()

    def _context_menu(self, point):
        item = self.plan_view.itemAt(point)
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return

        menu = QMenu(self)
        if isinstance(data, ROIExtractionProfile):
            action = QAction("Save ROI extraction Profile")
            action.triggered.connect(lambda _: self._save_roi_profile(data))
        elif isinstance(data, MeasurementCalculate):
            action = QAction("Save Measurement profile")
            action.triggered.connect(lambda _: self._save_measurement_profile(data.measurement_profile))
        else:
            raise ValueError(f"Not supported data type {type(data)} for {data}")
        menu.addAction(action)
        menu.exec_(self.plan_view.mapToGlobal(point))

    def _save_roi_profile(self, data: ROIExtractionProfile):
        if data.name in self.settings.roi_profiles:
            text, ok = QInputDialog.getText(
                self, "Name collision", "Profile with this name exists, please provide a new name.", text=data.name
            )
            if not ok:
                return None
            return self._save_roi_profile(typing.cast(ROIExtractionProfile, data.copy(update={"name": text})))
        self.settings.roi_profiles[data.name] = data
        return None

    def _save_measurement_profile(self, data: MeasurementProfile):
        if data.name in self.settings.measurement_profiles:
            text, ok = QInputDialog.getText(
                self, "Name collision", "Profile with this name exists, please provide a new name.", text=data.name
            )
            if not ok:
                return None
            return self._save_measurement_profile(typing.cast(MeasurementProfile, data.copy(update={"name": text})))
        self.settings.measurement_profiles[data.name] = data
        return None

    def update_plan_list(self):
        new_plan_list = sorted(self.settings.batch_plans.items(), key=lambda x: x[0])
        if self.calculate_plans.currentItem() is not None:
            text = str(self.calculate_plans.currentItem().text())
            try:
                index = [x[0] for x in new_plan_list].index(text)
            except ValueError:
                index = -1
        else:
            index = -1
        self.protect = True
        self.calculate_plans.clear()

        for name, plan in new_plan_list:
            item = QListWidgetItem(name)
            if plan.is_bad():
                item.setIcon(QIcon(os.path.join(icons_dir, "task-reject.png")))
                item.setToolTip(plan.get_error_source())
            self.calculate_plans.addItem(item)
        if index != -1:
            self.calculate_plans.setCurrentRow(index)
        self.protect = False

    def export_plans(self):
        choose = ExportDialog(self.settings.batch_plans, PlanPreview)
        if not choose.exec_():
            return
        dial = PSaveDialog(
            "Calculation plans (*.json)",
            caption="Export calculation plans",
            settings=self.settings,
            path="io.batch_plan_directory",
        )
        dial.selectFile("calculation_plans.json")
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            data = {x: self.settings.batch_plans[x] for x in choose.get_export_list()}
            with open(file_path, "w", encoding="utf-8") as ff:
                json.dump(data, ff, cls=self.settings.json_encoder_class, indent=2)

    def import_plans(self):
        dial = PLoadDialog(
            [LoadPlanJson, LoadPlanExcel],
            settings=self.settings,
            path="io.batch_plan_directory",
            caption="Import calculation plans",
        )
        if dial.exec_():
            res = dial.get_result()
            plans, err = res.load_class.load(res.load_location)
            if err:
                error_str = "\n".join(err)
                show_warning("Import error", f"error during importing, part of data were filtered. {error_str}")
            if not plans:
                show_warning("Import error", "No plans were imported")
                return
            choose = ImportDialog(plans, self.settings.batch_plans, PlanPreview, CalculationPlan)
            if choose.exec_():
                for original_name, final_name in choose.get_import_list():
                    self.settings.batch_plans[final_name] = plans[original_name]

    def delete_plan(self):
        if self.calculate_plans.currentItem() is None:
            return
        text = str(self.calculate_plans.currentItem().text())
        if not text:
            return  # pragma: no cover
        if text in self.settings.batch_plans:
            del self.settings.batch_plans[text]
        self.plan_view.clear()

    def edit_plan(self):
        if self.calculate_plans.currentItem() is None:
            return
        text = str(self.calculate_plans.currentItem().text())
        if not text:
            return  # pragma: no cover
        if text in self.settings.batch_plans:
            self.plan_to_edit = self.settings.batch_plans[text]
            self.plan_to_edit_signal.emit(self.plan_to_edit)

    def plan_preview(self, text):
        if self.protect:
            return
        text = str(text)
        if not text.strip():
            return
        plan = self.settings.batch_plans[text]
        if not plan.is_bad():
            self.plan_view.set_plan(plan)


class CalculatePlaner(QSplitter):
    """
    :type settings: Settings
    """

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.info_widget = CalculateInfo(settings)
        self.addWidget(self.info_widget)
        self.create_plan = CreatePlan(settings)
        self.info_widget.plan_to_edit_signal.connect(self.create_plan.edit_plan)
        self.addWidget(self.create_plan)
