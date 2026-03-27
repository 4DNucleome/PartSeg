import napari
from magicgui.widgets import create_widget
from napari.layers import Image as NapariImage
from qtpy.QtWidgets import QVBoxLayout, QWidget

from PartSeg.common_gui.dict_viewer import DictViewer


class LayerMetadata(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer
        self.layer_selector = create_widget(annotation=NapariImage, label="Layer", options={})
        self._dict_viewer = DictViewer()
        layout = QVBoxLayout()
        layout.addWidget(self.layer_selector.native)
        layout.addWidget(self._dict_viewer)

        self.setLayout(layout)
        self.update_metadata()
        self.layer_selector.changed.connect(self.update_metadata)

    def reset_choices(self):
        self.layer_selector.reset_choices()

    def update_metadata(self):
        if self.layer_selector.value is None:
            return
        self._dict_viewer.set_data(self.layer_selector.value.metadata)
