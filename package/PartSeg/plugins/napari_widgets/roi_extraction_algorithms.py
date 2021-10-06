import inspect
import itertools
import json
import os
import typing
from pathlib import Path

import numpy as np
from magicgui.widgets import Widget, create_widget
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels, Layer
from PyQt5.QtWidgets import QFileDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PartSegCore import UNIT_SCALE, Units
from PartSegCore.algorithm_describe_base import AlgorithmProperty, Register, ROIExtractionProfile
from PartSegCore.analysis import PartEncoder
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.channel_class import Channel
from PartSegCore.mask.algorithm_description import mask_algorithm_dict
from PartSegCore.segmentation import ROIExtractionResult
from PartSegImage import Image

from ..._roi_analysis.profile_export import ExportDialog, ProfileDictViewer
from ...common_backend.base_settings import BaseSettings
from ...common_gui.algorithms_description import (
    AlgorithmChooseBase,
    FormWidget,
    InteractiveAlgorithmSettingsWidget,
    QtAlgorithmProperty,
)
from ...common_gui.searchable_list_widget import SearchableListWidget
from ._settings import get_settings

if typing.TYPE_CHECKING:
    from qtpy.QtGui import QHideEvent, QShowEvent


SELECT_TEXT = "<select>"


class QtNapariAlgorithmProperty(QtAlgorithmProperty):
    def _get_field(self) -> typing.Union[QWidget, Widget]:
        if inspect.isclass(self.value_type) and issubclass(self.value_type, Channel):
            return create_widget(annotation=NapariImage, label="Image", options={})
        return super()._get_field()


class NapariFormWidget(FormWidget):
    @staticmethod
    def _element_list(fields) -> typing.Iterable[QtAlgorithmProperty]:
        mask = AlgorithmProperty("mask", "Mask", None, value_type=typing.Optional[Labels])
        return map(QtNapariAlgorithmProperty.from_algorithm_property, itertools.chain([mask], fields))

    def reset_choices(self, event=None):
        for widget in self.widgets_dict.values():
            # print("eeee", widget)
            if hasattr(widget.get_field(), "reset_choices"):
                widget.get_field().reset_choices(event)


class NapariInteractiveAlgorithmSettingsWidget(InteractiveAlgorithmSettingsWidget):
    @staticmethod
    def _form_widget(algorithm, start_values) -> FormWidget:
        return NapariFormWidget(algorithm.get_fields(), start_values=start_values)

    def reset_choices(self, event=None):
        self.form_widget.reset_choices(event)

    def get_order_mapping(self):
        layers = self.get_layers()
        layer_order_dict = {}
        for name, layer in layers.items():
            if isinstance(layer, NapariImage) and layer not in layer_order_dict:
                layer_order_dict[layer] = len(layer_order_dict)
        return layer_order_dict

    def get_values(self):
        layer_order_dict = self.get_order_mapping()
        return {
            k: layer_order_dict.get(v) if isinstance(v, NapariImage) else v
            for k, v in self.form_widget.get_values().items()
            if not isinstance(v, Labels) and k != "mask"
        }

    def get_layers(self):
        return {k: v for k, v in self.form_widget.get_values().items() if isinstance(v, Layer)}


class NapariAlgorithmChoose(AlgorithmChooseBase):
    @staticmethod
    def _algorithm_widget(settings, name, val) -> InteractiveAlgorithmSettingsWidget:
        return NapariInteractiveAlgorithmSettingsWidget(settings, name, val, [])

    def reset_choices(self, event=None):
        for widget in self.algorithm_dict.values():
            widget.reset_choices(event)


class ROIExtractionAlgorithms(QWidget):
    @staticmethod
    def get_method_dict():
        raise NotImplementedError

    @staticmethod
    def prefix() -> str:
        raise NotImplementedError

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self._scale = 1, 1, 1
        self.channel_names = []
        self.mask_name = ""

        self.viewer = napari_viewer
        self.settings = get_settings()
        self.algorithm_chose = NapariAlgorithmChoose(self.settings, self.get_method_dict())
        self.calculate_btn = QPushButton("Run")
        self.calculate_btn.clicked.connect(self._run_calculation)

        self.profile_combo_box = QComboBox()
        self.profile_combo_box.addItem(SELECT_TEXT)
        self.profile_combo_box.addItems(self.profile_dict.keys())
        self.save_btn = QPushButton("Save parameters")
        self.manage_btn = QPushButton("Manage parameters")

        layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.manage_btn)
        layout.addLayout(btn_layout)
        layout.addWidget(self.profile_combo_box)
        layout.addWidget(self.calculate_btn)
        layout.addWidget(self.algorithm_chose)

        self.setLayout(layout)

        self.algorithm_chose.result.connect(self.set_result)
        self.algorithm_chose.algorithm_changed.connect(self.algorithm_changed)
        self.save_btn.clicked.connect(self.save_action)
        self.manage_btn.clicked.connect(self.manage_action)
        self.profile_combo_box.currentTextChanged.connect(self.select_profile)

        self.update_tooltips()

    def manage_action(self):
        dialog = ProfilePreviewDialog(self.profile_dict, self.get_method_dict(), self.settings)
        dialog.exec_()
        self.refresh_profiles()

    def select_profile(self, text):
        if text in [SELECT_TEXT, ""]:
            return
        profile = self.profile_dict[text]
        self.algorithm_chose.change_algorithm(profile.algorithm, profile.values)
        self.profile_combo_box.setCurrentIndex(0)

    @property
    def profile_dict(self) -> typing.Dict[str, ROIExtractionProfile]:
        return self.settings.get_from_profile(f"{self.prefix()}.profiles", {})

    def save_action(self):
        widget: NapariInteractiveAlgorithmSettingsWidget = self.algorithm_chose.current_widget()
        profiles = self.profile_dict
        while True:
            text, ok = QInputDialog.getText(self, "Profile Name", "Input profile name here")
            if not ok:
                return
            if text not in profiles or QMessageBox.Yes == QMessageBox.warning(
                self,
                "Already exists",
                "Profile with this name already exist. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ):
                break
        resp = ROIExtractionProfile(text, widget.name, widget.get_values())
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
        self._scale = 1, 1, 1
        self.channel_names = []
        self.mask_name = ""

    def update_mask(self):
        widget: NapariInteractiveAlgorithmSettingsWidget = self.algorithm_chose.current_widget()
        mask = widget.get_layers().get("mask", None)
        if getattr(mask, "name", "") != self.mask_name:
            widget.set_mask(getattr(mask, "data", None))
            self.mask_name = getattr(mask, "name", "")

    def update_image(self):
        widget: NapariInteractiveAlgorithmSettingsWidget = self.algorithm_chose.current_widget()
        self.settings.last_executed_algorithm = widget.name
        layers = widget.get_order_mapping()
        axis_order = Image.axis_order.replace("C", "")
        channel_names = []
        for image_layer in layers:
            if image_layer.name not in channel_names:
                channel_names.append(image_layer.name)
        if self.channel_names == channel_names:
            return

        image_list = []
        for image_layer in layers:
            data_scale = image_layer.scale[-3:] / UNIT_SCALE[Units.nm.value]
            image_list.append(
                Image(
                    image_layer.data,
                    data_scale,
                    axes_order=axis_order[-image_layer.data.ndim :],
                    channel_names=[image_layer.name],
                )
            )
        res_image = image_list[0]
        for image in image_list[1:]:
            res_image = res_image.merge(image, "C")

        self._scale = np.array(res_image.spacing)
        self.channel_names = res_image.channel_names
        widget.image_changed(res_image)

    def _run_calculation(self):
        widget: NapariInteractiveAlgorithmSettingsWidget = self.algorithm_chose.current_widget()
        self.settings.last_executed_algorithm = widget.name
        self.update_image()
        self.update_mask()
        widget.execute()

    def showEvent(self, event: "QShowEvent") -> None:
        self.reset_choices(None)
        super().showEvent(event)

    def hideEvent(self, event: "QHideEvent") -> None:
        self.settings.dump()
        super().hideEvent(event)

    def reset_choices(self, event=None):
        self.algorithm_chose.reset_choices(event)

    def set_result(self, result: ROIExtractionResult):
        if "ROI" in self.viewer.layers:
            self.viewer.layers["ROI"].data = result.roi
        else:
            self.viewer.add_labels(
                result.roi, scale=self._scale[-result.roi.ndim :] * UNIT_SCALE[Units.nm.value], name="ROI"
            )

    def refresh_profiles(self):
        self.profile_combo_box.clear()
        self.profile_combo_box.addItem(SELECT_TEXT)
        self.profile_combo_box.addItems(self.profile_dict.keys())


class ROIAnalysisExtraction(ROIExtractionAlgorithms):
    @staticmethod
    def get_method_dict():
        return analysis_algorithm_dict

    @staticmethod
    def prefix() -> str:
        return "roi_analysis_extraction"


class ROIMaskExtraction(ROIExtractionAlgorithms):
    @staticmethod
    def get_method_dict():
        return mask_algorithm_dict

    @staticmethod
    def prefix() -> str:
        return "roi_mask_extraction"


class ProfilePreviewDialog(QDialog):
    def __init__(
        self, profile_dict: typing.Dict[str, ROIExtractionProfile], algorithm_dict: Register, settings: BaseSettings
    ):
        super().__init__()
        self.profile_dict = profile_dict
        self.algorithm_dict = algorithm_dict
        self.settings = settings

        self.profile_list = SearchableListWidget()
        self.profile_list.addItems(self.profile_dict.keys())
        self.profile_list.currentTextChanged.connect(self.profile_selected)
        self.profile_view = QPlainTextEdit()
        self.profile_view.setReadOnly(True)

        self.delete_btn = QPushButton("Delete")
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.export_action)
        self.import_btn = QPushButton("Import")
        layout = QGridLayout()
        layout.addWidget(self.profile_list, 0, 0)
        layout.addWidget(self.profile_view, 0, 1)
        layout.addWidget(self.delete_btn, 1, 0, 1, 2)
        layout.addWidget(self.export_btn, 2, 0)
        layout.addWidget(self.import_btn, 2, 1)

        self.setLayout(layout)

    def profile_selected(self, text):
        profile = self.profile_dict[text]
        self.profile_view.setPlainText(str(profile))

    def delete_action(self):
        if self.profile_list.text() in self.profile_dict:
            del self.profile_dict[self.profile_list.text()]
        self.profile_list.clear()
        self.profile_list.addItems(self.profile_dict.keys())

    def export_action(self):
        exp = ExportDialog(self.profile_dict, ProfileDictViewer)
        if not exp.exec_():
            return
        dial = QFileDialog(self, "Export profile segment")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setDirectory(self.settings.get("io.save_directory", str(Path.home())))
        dial.setNameFilter("Segment profile (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("segment_profile.json")
        dial.setHistory(dial.history() + self.settings.get_path_history())
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self.settings.set("io.save_directory", os.path.dirname(file_path))
            self.settings.add_path_history(os.path.dirname(file_path))
            data = {x: self.profile_dict[x] for x in exp.get_export_list()}
            with open(file_path, "w") as ff:
                json.dump(data, ff, cls=PartEncoder, indent=2)
