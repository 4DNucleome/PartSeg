import itertools
from typing import Optional, List

import numpy as np
from napari._qt.qt_viewer import QtViewer
from napari.components import ViewerModel as Viewer
from napari.layers.image import Image as NapariImage
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from vispy.color import Colormap, ColorArray

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.channel_control import ChannelProperty, ColorComboBoxGroup
from PartSeg.common_gui.stack_image_view import ImageShowState
from PartSegCore.color_image import create_color_map, ColorMap
from PartSegImage import Image


class ImageView(QWidget):
    position_changed = Signal([int, int, int], [int, int])
    component_clicked = Signal(int)
    text_info_change = Signal(str)
    hide_signal = Signal(bool)
    view_changed = Signal()

    def __init__(
        self, settings: BaseSettings, channel_property: ChannelProperty, name: str, parent: Optional[QWidget] = None,
    ):
        super(ImageView, self).__init__(parent=parent)

        self.settings = settings
        self.channel_property = channel_property
        self.name = name
        self.image_layers: List[List[NapariImage]] = []

        self.viewer = Viewer()
        self.viewer_widget = QtViewer(self.viewer)
        self.image_state = ImageShowState(settings, name)
        self.channel_control = ColorComboBoxGroup(settings, name, channel_property, height=30)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.channel_control)
        self.btn_layout2 = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addLayout(self.btn_layout)
        layout.addLayout(self.btn_layout2)
        layout.addWidget(self.viewer_widget)

        self.setLayout(layout)

        self.channel_control.change_channel.connect(self.change_visibility)
        self.viewer.events.status.connect(self.print_info)
        self.viewer.grid_view()

    def print_info(self, value):
        if self.viewer.active_layer:
            cords = np.array([int(x) for x in self.viewer.active_layer.coordinates])
            bright_array = []
            for layer_group in self.image_layers:
                for layer in layer_group:
                    moved_coords = (cords - layer.translate_grid).astype(np.int)
                    if np.all(moved_coords >= 0) and np.all(moved_coords < layer.shape):
                        bright_array.append(layer.data[tuple(moved_coords)])

            if not bright_array:
                self.text_info_change.emit("")
                return
            self.text_info_change.emit(f"{cords}: {bright_array}")

    @staticmethod
    def convert_to_vispy_colormap(colormap: ColorMap):
        return Colormap(ColorArray(create_color_map(colormap) / 255))

    def set_image(self, image: Optional[Image] = None):
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.image_layers = []
        image = self.add_image(image)
        self.viewer.stack_view()
        self.viewer.dims.set_point(image.time_pos, image.times // 2)
        self.viewer.dims.set_point(image.stack_pos, image.layers // 2)

    def add_image(self, image: Optional[Image]):
        if image is None:
            image = self.settings.image

        channels = max(image.channels, *[len(x) for x in self.image_layers]) if self.image_layers else image.channels
        self.channel_control.set_channels(channels)
        visibility = self.channel_control.channel_visibility
        self.image_layers.append([])
        for i in range(image.channels):
            self.image_layers[-1].append(
                self.viewer.add_image(
                    image.get_channel(i),
                    colormap=self.convert_to_vispy_colormap(self.channel_control.selected_colormaps[i]),
                    visible=visibility[i],
                    blending="additive",
                )
            )

        return image

    def grid_view(self):
        n_row = np.ceil(np.sqrt(len(self.image_layers))).astype(int)
        n_row = max(1, n_row)
        for layers_group, (x, y) in zip(self.image_layers, itertools.product(range(n_row), repeat=2)):
            for layer in layers_group:
                pass  # layer.translate_grid =

    def change_visibility(self, name: str, index: int):
        for group in self.image_layers:
            if len(group) > index:
                group[index].colormap = self.convert_to_vispy_colormap(self.channel_control.selected_colormaps[index])
                group[index].visible = self.channel_control.channel_visibility[index]
