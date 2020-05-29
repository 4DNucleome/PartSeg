from dataclasses import dataclass
import itertools
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
from napari._qt.qt_viewer import QtViewer
from napari._qt.qt_viewer_buttons import QtNDisplayButton, QtViewerPushButton
from napari.components import ViewerModel as Viewer
from napari.layers import Layer
from napari.layers.image import Image as NapariImage
from napari.layers.labels import Labels
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from vispy.color import Colormap, ColorArray, Color

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.channel_control import ChannelProperty, ColorComboBoxGroup
from PartSeg.common_gui.stack_image_view import ImageShowState, LabelEnum
from PartSegCore.color_image import create_color_map, ColorMap, calculate_borders
from PartSegCore.segmentation.segmentation_info import SegmentationInfo
from PartSegImage import Image


@dataclass
class ImageInfo:
    image: Image
    layers: List[NapariImage]
    mask: Optional[Labels] = None
    mask_array: Optional[np.ndarray] = None
    segmentation: Optional[Labels] = None
    segmentation_info: SegmentationInfo = SegmentationInfo(None)
    segmentation_count: int = 0

    def coords_in(self, coords: Union[List[int], np.ndarray]) -> bool:
        fst_layer = self.layers[0]
        moved_coords = self.translated_coords(coords)
        return np.all(moved_coords >= 0) and np.all(moved_coords < fst_layer.data.shape)

    def translated_coords(self, coords: Union[List[int], np.ndarray]) -> np.ndarray:
        fst_layer = self.layers[0]
        return np.subtract(coords, fst_layer.translate_grid).astype(np.int)


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
        self.image_info: Dict[str, ImageInfo] = {}
        self.current_image = ""
        self.components = None

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

        settings.mask_changed.connect(self.set_mask)
        settings.segmentation_changed.connect(self.set_segmentation)
        settings.segmentation_clean.connect(self.set_segmentation)
        settings.image_changed.connect(self.set_image)
        settings.image_spacing_changed.connect(self.update_spacing_info)
        # settings.labels_changed.connect(self.paint_layer)

        self.image_state.coloring_changed.connect(self.update_segmentation_coloring)
        self.image_state.borders_changed.connect(self.update_segmentation_representation)

    def update_spacing_info(self, image: Optional[Image] = None) -> None:
        """
        Update spacing of image if not provide, then use image pointed by settings.

        :param Optional[Image] image: image which spacing should be updated.
        :return: None
        """
        if image is None:
            image = self.settings.image

        if image.file_path not in self.image_info:
            raise ValueError("Image not registered")

        image_info = self.image_info[image.file_path]

        for layer in image_info.layers:
            layer.scale = image.normalized_scaling()

        if image_info.segmentation is not None:
            image_info.segmentation.scale = image.normalized_scaling()

        if image_info.mask is not None:
            image_info.mask.scale = image.normalized_scaling()

    def print_info(self, value):
        if not self.viewer.active_layer:
            return
        cords = np.array([int(x) for x in self.viewer.active_layer.coordinates])
        bright_array = []
        components = []
        for image_info in self.image_info.values():
            if not image_info.coords_in(cords):
                continue
            moved_coords = image_info.translated_coords(cords)
            for layer in image_info.layers:
                if layer.visible:
                    bright_array.append(layer.data[tuple(moved_coords)])
            if image_info.segmentation_info.segmentation is not None and image_info.segmentation is not None:
                val = image_info.segmentation_info.segmentation[tuple(moved_coords)]
                if val:
                    components.append(val)

        if not bright_array and not components:
            self.text_info_change.emit("")
            return
        text = f"{cords}: "
        if bright_array:
            if len(bright_array) == 1:
                text += str(bright_array[0])
            else:
                text += str(bright_array)
        self.components = components
        if components:
            if len(components) == 1:
                text += f" component: {components[0]}"
            else:
                text += f" components: {components}"
        self.text_info_change.emit(text)

    def get_control_view(self) -> ImageShowState:
        return self.image_state

    @staticmethod
    def convert_to_vispy_colormap(colormap: ColorMap):
        return Colormap(ColorArray(create_color_map(colormap) / 255))

    def mask_opacity(self) -> float:
        """Get mask opacity"""
        return self.settings.get_from_profile("mask_presentation   _opacity", 1)

    def mask_color(self) -> Colormap:
        """Get mask marking color"""
        color = Color(np.divide(self.settings.get_from_profile("mask_presentation_color", [255, 255, 255]), 255))
        return Colormap(ColorArray(["black", color.rgba]))

    def get_image(self, image: Optional[Image]) -> Image:
        if image is not None:
            return image
        if self.current_image not in self.image_info:
            return self.settings.image
        return self.image_info[self.current_image].image

    def set_segmentation(
        self, segmentation_info: Optional[SegmentationInfo] = None, image: Optional[Image] = None
    ) -> None:
        image = self.get_image(image)
        if segmentation_info is None:
            segmentation_info = self.settings.segmentation_info
        image_info = self.image_info[image.file_path]
        if image_info.segmentation is not None:
            self.viewer.layers.unselect_all()
            image_info.segmentation.selected = True
            self.viewer.layers.remove_selected()
            image_info.segmentation = None

        segmentation = segmentation_info.segmentation
        if segmentation is None:
            return

        image_info.segmentation_info = segmentation_info
        image_info.segmentation_count = np.max(segmentation)
        self.add_segmentation_layer(image_info)
        image_info.segmentation.colormap = self.get_segmentation_view_parameters(image_info)
        image_info.segmentation.opacity = self.image_state.opacity

    def get_segmentation_view_parameters(self, image_info: ImageInfo) -> Colormap:
        colors = self.settings.label_colors / 255
        if self.image_state.show_label == LabelEnum.Not_show or image_info.segmentation_count == 0 or colors.size == 0:
            colors = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        else:
            repeat = int(np.ceil(image_info.segmentation_count / colors.shape[0]))
            colors = np.concatenate([colors] * repeat)
            colors = np.concatenate([colors, np.ones(colors.shape[0]).reshape(colors.shape[0], 1)], axis=1)
            colors = np.concatenate([[[0, 0, 0, 0]], colors[: image_info.segmentation_count]])
            if self.image_state.show_label == LabelEnum.Show_selected:
                try:
                    colors *= self.settings.components_mask().reshape((colors.shape[0], 1))
                except ValueError:
                    pass
        control_points = np.linspace(0, 1, endpoint=True, num=colors.shape[0] + 1)
        return Colormap(colors, controls=control_points, interpolation="zero")

    def update_segmentation_coloring(self):
        for image_info in self.image_info.values():
            if image_info.segmentation is None:
                continue
            image_info.segmentation.colormap = self.get_segmentation_view_parameters(image_info)
            image_info.segmentation.opacity = self.image_state.opacity

    def remove_all_segmentation(self):
        self.viewer.layers.unselect_all()
        for image_info in self.image_info.values():
            if image_info.segmentation is None:
                continue
            image_info.segmentation.selected = True
            image_info.segmentation = None

        self.viewer.layers.remove_selected()

    def add_segmentation_layer(self, image_info: ImageInfo):
        if image_info.segmentation_info.segmentation is None:
            return
        if self.image_state.only_borders:
            data = calculate_borders(
                image_info.segmentation_info.segmentation,
                self.image_state.borders_thick // 2,
                self.viewer.dims.ndisplay == 2,
            )
            image_info.segmentation = self.viewer.add_image(data, scale=image_info.image.normalized_scaling())
        else:
            image_info.segmentation = self.viewer.add_image(
                image_info.segmentation_info.segmentation, scale=image_info.image.normalized_scaling()
            )

    def update_segmentation_representation(self):
        self.remove_all_segmentation()

        for image_info in self.image_info.values():
            self.add_segmentation_layer(image_info)

        self.update_segmentation_coloring()

    def set_mask(self, mask: Optional[np.ndarray] = None, image: Optional[Image] = None) -> None:
        image = self.get_image(image)
        if image.file_path not in self.image_info:
            raise ValueError("Image not added to viewer")
        if mask is None:
            mask = image.mask

        image_info = self.image_info[image.file_path]
        if image_info.mask is not None:
            self.viewer.layers.unselect_all()
            image_info.mask.selected = True
            self.viewer.layers.remove_selected()
            image_info.mask = None

        if mask is None:
            return

        mask_marker = mask == 0

        layer = self.viewer.add_image(mask_marker, scale=image.normalized_scaling(), blending="additive")
        layer.colormap = self.mask_color()
        layer.opacity = self.mask_opacity()
        image_info.mask = layer

    def update_mask_parameters(self):
        opacity = self.mask_opacity()
        colormap = self.mask_color()
        for image_info in self.image_info.values():
            if image_info.mask is not None:
                image_info.mask.opacity = opacity
                image_info.mask.colormap = colormap

    def set_image(self, image: Optional[Image] = None):
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.image_info = {}
        image = self.add_image(image)
        self.viewer.stack_view()
        self.viewer.dims.set_point(image.time_pos, image.times * image.normalized_scaling()[image.time_pos] // 2)
        self.viewer.dims.set_point(image.stack_pos, image.layers * image.normalized_scaling()[image.stack_pos] // 2)

    def has_image(self, image: Image):
        return image.file_path in self.image_info

    def add_image(self, image: Optional[Image]):
        if image is None:
            image = self.settings.image

        if not image.channels:
            raise ValueError("Need non empty image")

        if image.file_path in self.image_info:
            raise ValueError("Image already added")

        channels = image.channels
        if self.image_info:
            channels = max(channels, *[x.image.channels for x in self.image_info.values()])

        self.channel_control.set_channels(channels)
        visibility = self.channel_control.channel_visibility
        image_layers = []
        for i in range(image.channels):
            image_layers.append(
                self.viewer.add_image(
                    image.get_channel(i),
                    colormap=self.convert_to_vispy_colormap(self.channel_control.selected_colormaps[i]),
                    visible=visibility[i],
                    blending="additive",
                    scale=image.normalized_scaling(),
                )
            )
        if not self.image_info:
            image_layers[0].blending = "translucent"
        self.image_info[image.file_path] = ImageInfo(image, image_layers)
        self.current_image = image.file_path
        if image.mask is not None:
            self.set_mask()
        return image

    def images_bounds(self) -> Tuple[List[int], List[int]]:
        ranges = []
        for image_info in self.image_info.values():
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
        return size, min_shape

    def _shift_layer(self, layer: Layer, translate_2d):
        translate = [0] * layer.ndim
        translate[-2:] = translate_2d
        layer.translate_grid = translate

    def grid_view(self):
        """Present multiple images in grid view"""
        n_row = np.ceil(np.sqrt(len(self.image_info))).astype(int)
        n_row = max(1, n_row)
        scene_size, corner = self.images_bounds()
        for image_info, pos in zip(self.image_info.values(), itertools.product(range(n_row), repeat=2)):
            translate_2d = np.multiply(scene_size[-2:], pos)
            for layer in image_info.layers:
                self._shift_layer(layer, translate_2d)

            if image_info.mask is not None:
                self._shift_layer(image_info.mask, translate_2d)

            if image_info.segmentation is not None:
                self._shift_layer(image_info.segmentation, translate_2d)
        self.viewer.reset_view()

    def change_visibility(self, name: str, index: int):
        for image_info in self.image_info.values():
            if len(image_info.layers) > index:
                image_info.layers[index].colormap = self.convert_to_vispy_colormap(
                    self.channel_control.selected_colormaps[index]
                )
                image_info.layers[index].visible = self.channel_control.channel_visibility[index]

    def reset_image_size(self):
        self.viewer.reset_view()

    def set_theme(self, theme: str):
        self.viewer.theme = theme
