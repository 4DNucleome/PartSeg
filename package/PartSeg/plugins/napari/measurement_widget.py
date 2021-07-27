from typing import Optional

from magicgui.widgets import Table, create_widget
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels as NapariLabels
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from PartSeg.common_gui.universal_gui_part import EnumComboBox
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore import UNIT_SCALE, Units
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, PerComponent
from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT, MeasurementProfile, MeasurementResult
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


class Measurement(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_choice = create_widget(annotation=Optional[NapariImage], label="Image")
        self.labels_choice = create_widget(annotation=NapariLabels, label="Labels")
        self.scale_units_select = EnumComboBox(Units)
        self.units_select = EnumComboBox(Units)
        self.table = Table()
        self.calculate_btn = QPushButton("Calculate")
        layout = QVBoxLayout()
        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel(self.image_choice.label + ":"))
        options_layout.addWidget(self.image_choice.native)
        options_layout.addWidget(QLabel(self.labels_choice.label + ":"))
        options_layout.addWidget(self.labels_choice.native)
        options_layout.addWidget(QLabel("Data units:"))
        options_layout.addWidget(self.scale_units_select)
        options_layout.addWidget(QLabel("Units:"))
        options_layout.addWidget(self.units_select)
        options_layout.addWidget(self.calculate_btn)
        layout.addLayout(options_layout)

        bottom_layout = QHBoxLayout()
        self.measurement_layout = QVBoxLayout()
        bottom_layout.addLayout(self.measurement_layout)
        bottom_layout.addWidget(self.table.native)
        layout.addLayout(bottom_layout)

        self.setLayout(layout)
        self.image_choice.native.currentIndexChanged.connect(self.refresh_measurements)
        self.labels_choice.native.currentIndexChanged.connect(self.refresh_measurements)
        self.calculate_btn.clicked.connect(self.calculate)

    def calculate(self):
        to_calculate = []
        for i in range(self.measurement_layout.count()):
            # noinspection PyTypeChecker
            chk: QCheckBox = self.measurement_layout.itemAt(i).widget()
            if chk.isChecked():
                leaf: Leaf = MEASUREMENT_DICT[chk.text()].get_starting_leaf()
                to_calculate.append(leaf.replace_(per_component=PerComponent.Yes, area=AreaType.ROI))
        if not to_calculate:
            QMessageBox.warning(self, "No measurement", "Select at least one measurement")
            return

        profile = MeasurementProfile("", [MeasurementEntry(x.name, x) for x in to_calculate])

        data_ndim = self.image_choice.value.data.ndim
        if data_ndim > 4:
            QMessageBox.warning(
                self, "Not Supported", "Currently measurement engine does not support data over 4 dim (TZYX)"
            )
            return
        data_scale = self.image_choice.value.scale[-3:] / UNIT_SCALE[self.scale_units_select.get_value().value]
        image = Image(self.image_choice.value.data, data_scale, axes_order="TZYX"[-data_ndim:])
        roi_info = ROIInfo(self.labels_choice.value.data).fit_to_image(image)

        dial = ExecuteFunctionDialog(
            profile.calculate,
            kwargs={
                "image": image,
                "channel_num": 0,
                "roi": roi_info,
                "result_units": self.units_select.get_value(),
            },
        )
        dial.exec()
        result: MeasurementResult = dial.get_result()
        self.table.value = result.to_dataframe()

    def _clean_measurements(self):
        selected = set()
        for _ in range(self.measurement_layout.count()):
            # noinspection PyTypeChecker
            chk = self.measurement_layout.takeAt(0).widget()
            if chk.isChecked():
                selected.add(chk.text())
            chk.deleteLater()
        return selected

    def refresh_measurements(self, event=None):
        has_channel = self.image_choice.value is not None
        selected = self._clean_measurements()
        for val in MEASUREMENT_DICT.values():
            area = val.get_starting_leaf().area
            pc = val.get_starting_leaf().per_component
            if (
                val.get_fields()
                or (area is not None and area != AreaType.ROI)
                or (pc is not None and pc != PerComponent.Yes)
            ):
                continue
            text = val.get_name()
            chk = QCheckBox(text)
            if val.need_channel() and not has_channel:
                chk.setDisabled(True)
                chk.setToolTip("Need selected image")
            elif text in selected:
                chk.setChecked(True)
            self.measurement_layout.addWidget(chk)

    def reset_choices(self, event=None):
        self.image_choice.reset_choices(event)
        self.labels_choice.reset_choices(event)

    def showEvent(self, event):
        self.reset_choices()
        self.refresh_measurements()


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return Measurement
