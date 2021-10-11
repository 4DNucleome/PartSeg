from magicgui.widgets import create_widget
from napari import Viewer
from napari.layers import Labels
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from PartSeg.common_gui.mask_widget import MaskWidget
from PartSeg.plugins.napari_widgets._settings import get_settings


class MaskCreateNapari(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = get_settings()
        self.roi_select = create_widget(annotation=Labels, label="Image", options={})
        self.mask_widget = MaskWidget(self.settings)
        self.create = QPushButton("Create")
        self.name = QLineEdit()
        self.name.setText(self.settings.get("mask_create_name", "Mask"))
        layout = QVBoxLayout()
        layout.addWidget(self.roi_select.native)
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("New layer name:"))
        layout2.addWidget(self.name)
        layout.addLayout(layout2)
        layout.addWidget(self.mask_widget)
        layout.addWidget(self.create)
        self.setLayout(layout)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.roi_select.reset_choices(event)

    def create_mask(self):
        self.settings.set("mask_create_name", self.name.text())
        self.mask_widget.get_mask_property()

        self.settings.dump()
