import warnings
from typing import Optional

import numpy as np
from magicgui.widgets import create_widget
from napari import Viewer
from napari.layers import Labels
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from PartSeg.common_gui.mask_widget import MaskWidget
from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSegCore.mask_create import calculate_mask
from PartSegCore.segmentation.algorithm_base import calculate_operation_radius


class NapariMaskWidget(MaskWidget):
    def __init__(self, settings, label_widget):
        super().__init__(settings)
        self.label_widget = label_widget

    def get_dilate_radius(self):
        if self.label_widget.value is None:
            return self.dilate_radius.value()
        spacing = self.label_widget.value.scale[-3:]
        radius = calculate_operation_radius(self.dilate_radius.value(), spacing, self.dilate_dim.currentEnum())
        if isinstance(radius, (list, tuple)):
            return [int(x + 0.5) for x in radius]
        return int(radius)


class MaskCreate(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = get_settings()
        self.roi_select = create_widget(annotation=Labels, label="ROI", options={})
        self.mask_select = create_widget(annotation=Optional[Labels], label="Base mask", options={})
        self.mask_widget = NapariMaskWidget(self.settings, self.roi_select)
        self.create = QPushButton("Create")
        self.name = QLineEdit()
        self.name.setText(self.settings.get("mask_create_name", "Mask"))
        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("ROI"))
        layout2.addWidget(self.roi_select.native)
        layout.addLayout(layout2)
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Base mask"))
        layout2.addWidget(self.mask_select.native)
        layout.addLayout(layout2)
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("New layer name:"))
        layout2.addWidget(self.name)
        layout.addLayout(layout2)
        layout.addWidget(self.mask_widget)
        layout.addWidget(self.mask_widget.radius_information)
        layout.addWidget(self.create)
        self.setLayout(layout)

        self.mask_select.native.setDisabled(True)
        self.mask_widget.clip_to_mask.stateChanged.connect(self.mask_select.native.setEnabled)
        self.create.clicked.connect(self.create_mask)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.roi_select.reset_choices(event)
        self.mask_select.reset_choices(event)

    def create_mask(self):
        if self.roi_select.value is None:
            return
        layer_name = self.name.text()
        self.settings.set("mask_create_name", layer_name)
        mask_property = self.mask_widget.get_mask_property()
        if mask_property.clip_to_mask and self.mask_select.value is None:
            warnings.warn("Select base mask", RuntimeWarning, stacklevel=1)
            return
        base_mask = None if self.mask_select.value is None else self.mask_select.value.data
        scale = np.array(self.roi_select.value.scale)
        mask = np.array(calculate_mask(mask_property, self.roi_select.value.data, base_mask, scale[-3:]))
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = mask
        else:
            self.viewer.add_labels(
                mask,
                scale=scale[-int(mask.ndim) :],
                name=layer_name,
            )

        self.settings.dump()
