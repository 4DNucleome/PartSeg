import typing
from contextlib import suppress

import numpy as np
import pandas as pd
from napari import Viewer
from napari.layers import Layer
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PartSeg import plugins
from PartSeg._roi_analysis.profile_export import ExportDialog, ImportDialog, ProfileDictViewer
from PartSeg.common_backend.base_settings import IO_SAVE_DIRECTORY, BaseSettings
from PartSeg.common_backend.except_hook import show_warning
from PartSeg.common_gui.algorithms_description import (
    AlgorithmChooseBase,
    FormWidget,
    InteractiveAlgorithmSettingsWidget,
)
from PartSeg.common_gui.custom_load_dialog import PLoadDialog
from PartSeg.common_gui.custom_save_dialog import PSaveDialog
from PartSeg.common_gui.searchable_combo_box import SearchComboBox
from PartSeg.common_gui.searchable_list_widget import SearchableListWidget
from PartSeg.common_gui.universal_gui_part import TextShow
from PartSeg.plugins import register as register_plugins
from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSeg.plugins.napari_widgets.utils import NapariFormWidgetWithMask, generate_image
from PartSegCore import UNIT_SCALE, Units
from PartSegCore.algorithm_describe_base import AlgorithmSelection, ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import AnalysisAlgorithmSelection
from PartSegCore.analysis.load_functions import LoadProfileFromJSON
from PartSegCore.analysis.save_functions import SaveProfilesToJSON
from PartSegCore.mask.algorithm_description import MaskAlgorithmSelection
from PartSegCore.segmentation import ROIExtractionResult

if typing.TYPE_CHECKING:
    from qtpy.QtGui import QHideEvent, QShowEvent  # pragma: no cover
SELECT_TEXT = "<select>"


class NapariInteractiveAlgorithmSettingsWidget(InteractiveAlgorithmSettingsWidget):
    form_widget: NapariFormWidgetWithMask

    def _form_widget(self, algorithm, start_values) -> FormWidget:
        return NapariFormWidgetWithMask(
            algorithm.__argument_class__ if algorithm.__new_style__ else algorithm.get_fields(),
            start_values=start_values,
            parent=self,
        )

    def reset_choices(self, event=None):
        self.form_widget.reset_choices(event)

    def get_layer_list(self) -> typing.List[str]:
        return [x.name for x in self.get_layers().values() if x.name != "mask"]

    def get_layers(self) -> typing.Dict[str, Layer]:
        values = self.form_widget.get_layers()
        return {k: v for k, v in values.items() if isinstance(v, Layer)}


class NapariAlgorithmChoose(AlgorithmChooseBase):
    def _algorithm_widget(self, settings, val) -> InteractiveAlgorithmSettingsWidget:
        return NapariInteractiveAlgorithmSettingsWidget(settings, val, [], parent=self)

    def reset_choices(self, event=None):
        for widget in self.algorithm_dict.values():
            widget.reset_choices(event)


class ROIExtractionAlgorithms(QWidget):
    @staticmethod
    def get_method_dict() -> typing.Type[AlgorithmSelection]:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def prefix() -> str:  # pragma: no cover
        raise NotImplementedError

    def __init__(self, napari_viewer: Viewer):
        plugins.register_if_need()
        super().__init__()
        self._scale = np.array((1, 1, 1))
        self.channel_names = []
        self.mask_name = ""

        self.viewer = napari_viewer
        self.settings = get_settings()
        self.algorithm_chose = NapariAlgorithmChoose(self.settings, self.get_method_dict())
        self.calculate_btn = QPushButton("Run")
        self.calculate_btn.clicked.connect(self._run_calculation)
        self.info_text = TextShow()

        self.profile_combo_box = SearchComboBox()
        self.profile_combo_box.addItem(SELECT_TEXT)
        self.profile_combo_box.addItems(list(self.profile_dict.keys()))
        self.save_btn = QPushButton("Save parameters")
        self.manage_btn = QPushButton("Manage parameters")
        self.target_layer_name = QLineEdit()
        self.target_layer_name.setText(self.settings.get(f"{self.prefix()}.target_layer_name", "ROI"))

        layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.manage_btn)
        layout.addLayout(btn_layout)
        target_layer_layout = QHBoxLayout()
        target_layer_layout.addWidget(QLabel("Target layer name:"))
        target_layer_layout.addWidget(self.target_layer_name)
        layout.addLayout(target_layer_layout)
        layout.addWidget(self.profile_combo_box)
        layout.addWidget(self.calculate_btn)
        layout.addWidget(self.algorithm_chose, 1)
        layout.addWidget(self.info_text)

        self.setLayout(layout)

        self.algorithm_chose.result.connect(self.set_result)
        self.algorithm_chose.finished.connect(self._enable_calculation_btn)
        self.algorithm_chose.algorithm_changed.connect(self.algorithm_changed)
        self.save_btn.clicked.connect(self.save_action)
        self.manage_btn.clicked.connect(self.manage_action)
        self.profile_combo_box.textActivated.connect(self.select_profile)

        self.update_tooltips()
        register_plugins()

    def _enable_calculation_btn(self):
        self.calculate_btn.setEnabled(True)

    def manage_action(self):
        dialog = ProfilePreviewDialog(self.profile_dict, self.get_method_dict(), self.settings, parent=self)
        dialog.exec_()
        self.refresh_profiles()

    def select_profile(self, text):
        if text in [SELECT_TEXT, ""]:
            return  # pragma: no cover
        profile = self.profile_dict[text]
        self.algorithm_chose.change_algorithm(profile.algorithm, profile.values)
        self.profile_combo_box.setCurrentIndex(0)

    @property
    def profile_dict(self) -> typing.Dict[str, ROIExtractionProfile]:
        return self.settings.get_from_profile(f"{self.prefix()}.profiles", {})

    def save_action(self):
        widget = typing.cast(NapariInteractiveAlgorithmSettingsWidget, self.algorithm_chose.current_widget())
        profiles = self.profile_dict
        while True:
            text, ok = QInputDialog.getText(self, "Profile Name", "Input profile name here")
            if not ok:
                return  # pragma: no cover
            if text not in profiles or QMessageBox.Yes == QMessageBox.warning(
                self,
                "Already exists",
                "Profile with this name already exist. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ):
                break  # pragma: no cover
        resp = ROIExtractionProfile(name=text, algorithm=widget.name, values=widget.get_values())
        profiles[text] = resp
        self.settings.dump()
        self.profile_combo_box.addItem(text)
        self.update_tooltips()

    def update_tooltips(self):
        for i in range(1, self.profile_combo_box.count()):
            if self.profile_combo_box.itemData(i, Qt.ToolTipRole) is not None:
                continue
            text = self.profile_combo_box.itemText(i)
            profile: ROIExtractionProfile = self.profile_dict[text]
            tool_tip_text = str(profile)
            self.profile_combo_box.setItemData(i, tool_tip_text, Qt.ToolTipRole)

    def algorithm_changed(self):
        self._scale = np.array((1, 1, 1))
        self.channel_names = []
        self.mask_name = ""

    def update_mask(self):
        widget = typing.cast(NapariInteractiveAlgorithmSettingsWidget, self.algorithm_chose.current_widget())
        mask = widget.get_layers().get("mask", None)
        if getattr(mask, "name", "") != self.mask_name or (widget.mask() is None and mask is not None):
            widget.set_mask(getattr(mask, "data", None))
            self.mask_name = getattr(mask, "name", "")

    def update_image(self):
        widget = typing.cast(NapariInteractiveAlgorithmSettingsWidget, self.algorithm_chose.current_widget())
        self.settings.last_executed_algorithm = widget.name
        layer_names: typing.List[str] = widget.get_layer_list()
        if layer_names == self.channel_names:
            return
        image = generate_image(self.viewer, *layer_names)

        self._scale = np.array(image.spacing)
        self.channel_names = image.channel_names
        widget.image_changed(image)
        self.mask_name = ""

    def _run_calculation(self):
        widget = typing.cast(NapariInteractiveAlgorithmSettingsWidget, self.algorithm_chose.current_widget())
        self.settings.last_executed_algorithm = widget.name
        self.update_image()
        self.update_mask()
        widget.execute()
        self.calculate_btn.setDisabled(True)

    def showEvent(self, event: "QShowEvent") -> None:
        self.reset_choices(None)
        super().showEvent(event)

    def hideEvent(self, event: "QHideEvent") -> None:
        self.settings.dump()
        super().hideEvent(event)

    def reset_choices(self, event=None):
        self.algorithm_chose.reset_choices(event)

    def set_result(self, result: ROIExtractionResult):
        if result.info_text:
            show_info(result.info_text)
        if len(result.roi_info.bound_info) == 0:
            if not result.info_text:
                show_info("There is no ROI in result. Please check algorithm parameters.")
            return
        roi = result.roi
        if self.sender() is not None:
            self.info_text.setPlainText(self.sender().get_info_text())
            with suppress(Exception):
                roi = self.sender().current_widget().algorithm_thread.algorithm.image.fit_array_to_image(result.roi)

        layer_name = self.target_layer_name.text()
        self.settings.set(f"{self.prefix()}.target_layer_name", layer_name)
        column_list = []
        column_set = set()
        for value in result.roi_annotation.values():
            for column_name in value.items():
                if column_name not in column_set:
                    column_list.append(column_name)
                    column_set.add(column_name)
        properties = pd.DataFrame.from_dict(result.roi_annotation, orient="index")
        properties["index"] = list(result.roi_annotation.keys())
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = result.roi
            self.viewer.layers[layer_name].metadata = {"parameters": result.parameters}
            self.viewer.layers[layer_name].properties = properties
        else:
            self.viewer.add_labels(
                roi,
                scale=np.array(self._scale)[-result.roi.ndim :] * UNIT_SCALE[Units.nm.value],
                name=layer_name,
                metadata={"parameters": result.parameters},
                properties=properties,
            )

    def refresh_profiles(self):
        self.profile_combo_box.clear()
        self.profile_combo_box.addItem(SELECT_TEXT)
        self.profile_combo_box.addItems(list(self.profile_dict.keys()))


class ROIAnalysisExtraction(ROIExtractionAlgorithms):
    @staticmethod
    def get_method_dict():
        return AnalysisAlgorithmSelection

    @staticmethod
    def prefix() -> str:
        return "roi_analysis_extraction"


class ROIMaskExtraction(ROIExtractionAlgorithms):
    @staticmethod
    def get_method_dict():
        return MaskAlgorithmSelection

    @staticmethod
    def prefix() -> str:
        return "roi_mask_extraction"


class ProfilePreviewDialog(QDialog):
    def __init__(
        self,
        profile_dict: typing.Dict[str, ROIExtractionProfile],
        algorithm_selection: typing.Type[AlgorithmSelection],
        settings: BaseSettings,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.profile_dict = profile_dict
        self.algorithm_selection = algorithm_selection
        self.settings = settings

        self.profile_list = SearchableListWidget()
        self.profile_list.addItems(list(self.profile_dict.keys()))
        self.profile_list.currentTextChanged.connect(self.profile_selected)
        self.profile_view = QPlainTextEdit()
        self.profile_view.setReadOnly(True)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_action)
        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self.rename_action)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_action)
        self.import_btn = QPushButton("Import")
        self.import_btn.clicked.connect(self.import_action)
        layout = QGridLayout()
        layout.addWidget(self.profile_list, 0, 0)
        layout.addWidget(self.profile_view, 0, 1)
        layout.addWidget(self.delete_btn, 1, 0)
        layout.addWidget(self.rename_btn, 1, 1)
        layout.addWidget(self.export_btn, 2, 0)
        layout.addWidget(self.import_btn, 2, 1)

        self.setLayout(layout)

    def profile_selected(self, text):
        if text not in self.profile_dict:
            return
        profile = self.profile_dict[text]
        self.profile_view.setPlainText(str(profile))

    def delete_action(self):
        if self.profile_list.currentItem() is None:
            return  # pragma: no cover
        if self.profile_list.currentItem().text() in self.profile_dict:
            del self.profile_dict[self.profile_list.currentItem().text()]
        self.profile_list.clear()
        self.profile_list.addItems(list(self.profile_dict.keys()))

    def rename_action(self):
        if self.profile_list.currentItem() is None:
            return  # pragma: no cover
        old_name = self.profile_list.currentItem().text()
        if old_name not in self.profile_dict:
            return  # pragma: no cover

        text, ok = QInputDialog.getText(self, "Profile Name", "Input profile name here")
        if not ok:
            return  # pragma: no cover

        if text in self.profile_dict:  # pragma: no cover
            QMessageBox.warning(
                self,
                "Already exists",
                "Profile with this name already exist.",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
            self.rename_action()
            return
        profile = self.profile_dict[old_name]
        del self.profile_dict[old_name]
        profile.name = text
        self.profile_dict[text] = profile
        self.profile_list.clear()
        self.profile_list.addItems(list(self.profile_dict.keys()))

    def export_action(self):
        exp = ExportDialog(self.profile_dict, ProfileDictViewer, parent=self)
        if not exp.exec_():
            return  # pragma: no cover
        dial = PSaveDialog(SaveProfilesToJSON, settings=self.settings, parent=self, path=IO_SAVE_DIRECTORY)
        if dial.exec_():
            save_location, _selected_filter, save_class, values = dial.get_result()
            data = {x: self.profile_dict[x] for x in exp.get_export_list()}
            save_class.save(save_location, data, values)

    def import_action(self):
        dial = PLoadDialog(LoadProfileFromJSON, settings=self.settings, parent=self, path=IO_SAVE_DIRECTORY)
        if not dial.exec_():
            return  # pragma: no cover
        file_list, _, load_class = dial.get_result()
        profs, err = load_class.load(file_list)
        if err:
            show_warning("Import error", "error during importing, part of data were filtered.")  # pragma: no cover
        imp = ImportDialog(profs, self.profile_dict, ProfileDictViewer, parent=self)
        if not imp.exec_():
            return  # pragma: no cover
        for original_name, final_name in imp.get_import_list():
            self.profile_dict[final_name] = profs[original_name]
        self.settings.dump()
        self.profile_list.clear()
        self.profile_list.addItems(list(self.profile_dict.keys()))
