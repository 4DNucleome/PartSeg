from tempfile import TemporaryDirectory
from typing import Type

from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels as NapariLabels
from napari.layers import Layer as NapariLayer
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel

from PartSeg._roi_mask.simple_measurements import SimpleMeasurements
from PartSeg._roi_mask.stack_settings import StackSettings
from PartSegImage import Image


class SimpleMeasurementWidget(SimpleMeasurements):
    def __init__(self, napari_viewer: Viewer):
        settings = StackSettings(TemporaryDirectory())
        super().__init__(settings)
        self._settings = settings
        self.viewer = napari_viewer
        self.channel_select.setVisible(False)
        self.image_select = QComboBox()
        self.roi_select = QComboBox()
        self._shift = 3

        lay = self.measurement_layout.takeAt(1).layout()
        text = lay.takeAt(0).widget()
        text.deleteLater()

        l1 = QHBoxLayout()
        l1.addWidget(QLabel("Image"))
        l1.addWidget(self.image_select)
        self.measurement_layout.insertLayout(1, l1)

        l1 = QHBoxLayout()
        l1.addWidget(QLabel("ROI"))
        l1.addWidget(self.roi_select)
        self.measurement_layout.insertLayout(2, l1)

        self.viewer.layers.events.connect(self.image_list_update)
        self.viewer.layers.events.connect(self.roi_list_update)
        self.refresh_measurements()
        self.image_list_update()
        self.roi_list_update()

    def update_elements(self, klass: Type[NapariLayer], select: QComboBox):
        current = select.currentText()
        select.clear()
        res = []
        for layer in self.viewer.layers:
            if layer.__class__ == klass:
                res.append(layer.name)
        try:
            index = res.index(current)
        except ValueError:
            index = 0
        select.addItems(res)
        select.setCurrentIndex(index)
        self.calculate_btn.setDisabled(len(res) == 0)

    def image_list_update(self, event=None):
        self.update_elements(NapariImage, self.image_select)

    def roi_list_update(self, event=None):
        self.update_elements(NapariLabels, self.roi_select)

    def calculate(self):
        name = self.image_select.currentText()
        for layer in self.viewer.layers:
            if layer.name == name:
                channel = layer
                break
        else:
            raise ValueError("LAyer not found")
        name = self.roi_select.currentText()
        for layer in self.viewer.layers:
            if layer.name == name:
                roi = layer
                break
        else:
            raise ValueError("LAyer not found")
        # TODO fix scale
        self.settings.image = Image(channel.data, channel.scale, axes_order="TZXY"[-channel.data.ndim :])
        self.settings.roi = roi.data
        super().calculate()


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return SimpleMeasurementWidget
