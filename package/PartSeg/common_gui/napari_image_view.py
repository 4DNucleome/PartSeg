from dataclasses import dataclass
import itertools
from typing import Optional, List, Dict, Tuple

import numpy as np
from napari._qt.qt_viewer import QtViewer
from napari._qt.qt_viewer_buttons import QtNDisplayButton, QtViewerPushButton
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


@dataclass
class ImageInfo:
    image: Image
    layers: List[NapariImage]
    mask: Optional[NapariImage] = None


class ImageView(QWidget):
    position_changed = Signal([int, int, int], [int, int])
    component_clicked = Signal(int)
    text_info_change = Signal(str)
    hide_signal = Signal(bool)
    view_changed = Signal()

    def __init__(
        self,
        settings: BaseSettings,
        channel_property: ChannelProperty,
        name: str,
        parent: Optional[QWidget] = None,
        ndisplay=2,
    ):
        super(ImageView, self).__init__(parent=parent)

        self.settings = settings
        self.channel_property = channel_property
        self.name = name
        self.image_layers: Dict[str, ImageInfo] = {}
        self.current_image = ""

        self.viewer = Viewer(ndisplay=ndisplay)
        self.viewer.theme = self.settings.theme_name
        self.viewer_widget = QtViewer(self.viewer)
        self.image_state = ImageShowState(settings, name)
        self.channel_control = ColorComboBoxGroup(settings, name, channel_property, height=30)
        self.ndim_btn = QtNDisplayButton(self.viewer)
        self.reset_view_button = QtViewerPushButton(self.viewer, "home", "Reset view", self.viewer.reset_view)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.reset_view_button)
        self.btn_layout.addWidget(self.ndim_btn)
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
            for image_info in self.image_layers.values():
                for layer in image_info.layers:
                    moved_coords = (cords - layer.translate_grid).astype(np.int)
                    if np.all(moved_coords >= 0) and np.all(moved_coords < layer.data.shape):
                        bright_array.append(layer.data[tuple(moved_coords)])

            if not bright_array:
                self.text_info_change.emit("")
                return
            self.text_info_change.emit(f"{cords}: {bright_array}")

    def toggle_dims(self):
        if self.viewer.dims.ndim == 2:
            self.viewer.dims.ndim = 3
        else:
            self.viewer.dims.ndim = 2

    @staticmethod
    def convert_to_vispy_colormap(colormap: ColorMap):
        return Colormap(ColorArray(create_color_map(colormap) / 255))

    def set_image(self, image: Optional[Image] = None):
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.image_layers = {}
        image = self.add_image(image)
        self.viewer.stack_view()
        self.viewer.dims.set_point(image.time_pos, image.times // 2)
        self.viewer.dims.set_point(image.stack_pos, image.layers // 2)

    def has_image(self, image: Image):
        return image.file_path in self.image_layers

    def add_image(self, image: Optional[Image]):
        if image is None:
            image = self.settings.image

        if not image.channels:
            raise ValueError("Need non empty image")

        if image.file_path in self.image_layers:
            raise ValueError("Image already added")

        channels = image.channels
        if self.image_layers:
            channels = max(channels, *[x.image.channels for x in self.image_layers.values()])

        self.channel_control.set_channels(channels)
        visibility = self.channel_control.channel_visibility
        image_layers = []
        min_scale = min(image.spacing)
        scaling = (1,) + tuple([x / min_scale for x in image.spacing])
        for i in range(image.channels):
            image_layers.append(
                self.viewer.add_image(
                    image.get_channel(i),
                    colormap=self.convert_to_vispy_colormap(self.channel_control.selected_colormaps[i]),
                    visible=visibility[i],
                    blending="additive",
                    scale=scaling,
                )
            )
        self.image_layers[image.file_path] = ImageInfo(image, image_layers)
        self.current_image = image.file_path
        return image

    def images_bounds(self) -> Tuple[List[int], List[int], Tuple[int, int]]:
        ranges = []
        for image_info in self.image_layers.values():
            if not image_info.layers:
                continue
            ranges = [
                (min(a, b), max(c, d), min(e, f))
                for (a, c, e), (b, d, f) in itertools.zip_longest(
                    image_info.layers[0].dims.range, ranges, fillvalue=(np.inf, -np.inf, np.inf)
                )
            ]

        visible = [ranges[i] for i in self.viewer.dims.displayed]
        min_shape, max_shape, _ = zip(*visible)
        size = np.subtract(max_shape, min_shape)
        return size, min_shape, (min_shape[1], max_shape[1])

    def grid_view(self):
        n_row = np.ceil(np.sqrt(len(self.image_layers))).astype(int)
        n_row = max(1, n_row)
        scene_size, corner, layers = self.images_bounds()
        layers_max = layers[1] - layers[0]
        for image_info, pos in zip(self.image_layers.values(), itertools.product(range(n_row), repeat=2)):
            translate_2d = np.multiply(scene_size[-2:], pos)
            for layer in image_info.layers:
                translate = [0] * layer.ndim
                translate[-2:] = translate_2d
                layers_shift = layers[0] + (layers_max - layer.shape[1]) // 2
                translate[-3] = layers_shift
                layer.translate_grid = translate

    def change_visibility(self, name: str, index: int):
        for image_info in self.image_layers.values():
            if len(image_info.layers) > index:
                image_info.layers[index].colormap = self.convert_to_vispy_colormap(
                    self.channel_control.selected_colormaps[index]
                )
                image_info.layers[index].visible = self.channel_control.channel_visibility[index]

    def reset_image_size(self):
        self.viewer.reset_view()

    def set_theme(self, theme: str):
        self.viewer.theme = theme
