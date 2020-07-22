import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from napari._qt.qt_viewer_buttons import QtViewerPushButton
from napari.components import ViewerModel as Viewer
from napari.layers import Layer
from napari.layers.image import Image as NapariImage
from napari.layers.image._image_constants import Interpolation3D
from napari.layers.labels import Labels
from napari.qt import QtNDisplayButton, QtViewer
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from vispy.color import Color, ColorArray, Colormap
from vispy.scene import BaseCamera

from PartSegCore.color_image import ColorMap, calculate_borders, create_color_map
from PartSegCore.image_operations import NoiseFilterType, gaussian, median
from PartSegCore.segmentation_info import SegmentationInfo
from PartSegImage import Image

from ..common_backend.base_settings import BaseSettings, ViewSettings
from .channel_control import ChannelProperty, ColorComboBoxGroup


@dataclass
class ImageInfo:
    image: Image
    layers: List[NapariImage]
    filter_info: List[Tuple[NoiseFilterType, float]] = field(default_factory=list)
    mask: Optional[Labels] = None
    mask_array: Optional[np.ndarray] = None
    segmentation: Optional[Labels] = None
    segmentation_info: SegmentationInfo = field(default_factory=lambda: SegmentationInfo(None))
    segmentation_count: int = 0

    def coords_in(self, coords: Union[List[int], np.ndarray]) -> bool:
        fst_layer = self.layers[0]
        moved_coords = self.translated_coords(coords)
        return np.all(moved_coords >= 0) and np.all(moved_coords < fst_layer.data.shape)

    def translated_coords(self, coords: Union[List[int], np.ndarray]) -> np.ndarray:
        fst_layer = self.layers[0]
        return np.subtract(coords, fst_layer.translate_grid).astype(np.int)


class LabelEnum(Enum):
    Not_show = 0
    Show_results = 1
    Show_selected = 2

    def __str__(self):
        if self.value == 0:
            return "Don't show"
        return self.name.replace("_", " ")


class ImageShowState(QObject):
    """Object for storing state used when presenting it in :class:`.ImageView`"""

    parameter_changed = Signal()  # signal informing that some of image presenting parameters
    coloring_changed = Signal()
    borders_changed = Signal()
    # changed and image need to be refreshed

    def __init__(self, settings: ViewSettings, name: str):
        if len(name) == 0:
            raise ValueError("Name string should be not empty")
        super().__init__()
        self.name = name
        self.settings = settings
        self.zoom = False
        self.move = False
        self.opacity = settings.get_from_profile(f"{name}.image_state.opacity", 1.0)
        self.show_label = settings.get_from_profile(f"{name}.image_state.show_label", LabelEnum.Show_results)
        self.only_borders = settings.get_from_profile(f"{name}.image_state.only_border", True)
        self.borders_thick = settings.get_from_profile(f"{name}.image_state.border_thick", 1)

    def set_zoom(self, val):
        self.zoom = val

    def set_borders(self, val: bool):
        """decide if draw only component 2D borders, or whole area"""
        if self.only_borders != val:
            self.settings.set_in_profile(f"{self.name}.image_state.only_border", val)
            self.only_borders = val
            self.parameter_changed.emit()
            self.borders_changed.emit()

    def set_borders_thick(self, val: int):
        """If draw only 2D borders of component then set thickness of line used for it"""
        if val != self.borders_thick:
            self.settings.set_in_profile(f"{self.name}.image_state.border_thick", val)
            self.borders_thick = val
            self.parameter_changed.emit()
            self.borders_changed.emit()

    def set_opacity(self, val: float):
        """Set opacity of component labels"""
        if self.opacity != val:
            self.settings.set_in_profile(f"{self.name}.image_state.opacity", val)
            self.opacity = val
            self.parameter_changed.emit()
            self.coloring_changed.emit()

    def components_change(self):
        if self.show_label == LabelEnum.Show_selected:
            self.parameter_changed.emit()
            self.coloring_changed.emit()

    def set_show_label(self, val: LabelEnum):
        if self.show_label != val:
            self.settings.set_in_profile(f"{self.name}.image_state.show_label", val)
            self.show_label = val
            self.parameter_changed.emit()
            self.coloring_changed.emit()


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
        self.viewer_widget = NapariQtViewer(self.viewer)
        self.image_state = ImageShowState(settings, name)
        self.channel_control = ColorComboBoxGroup(settings, name, channel_property, height=30)
        self.ndim_btn = QtNDisplayButton(self.viewer)
        self.reset_view_button = QtViewerPushButton(self.viewer, "home", "Reset view", self.viewer.reset_view)
        self.mask_chk = QCheckBox()
        self.mask_label = QLabel("Mask:")

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.reset_view_button)
        self.btn_layout.addWidget(self.ndim_btn)
        self.btn_layout.addWidget(self.channel_control, 1)
        self.btn_layout.addWidget(self.mask_label)
        self.btn_layout.addWidget(self.mask_chk)
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
        self.old_scene: BaseCamera = self.viewer_widget.view.scene

        self.image_state.coloring_changed.connect(self.update_segmentation_coloring)
        self.image_state.borders_changed.connect(self.update_segmentation_representation)
        self.mask_chk.stateChanged.connect(self.change_mask_visibility)
        self.viewer_widget.view.scene.transform.changed.connect(self._view_changed, position="last")
        self.viewer.dims.events.axis.connect(self._view_changed, position="last")
        self.viewer.dims.events.ndisplay.connect(self._view_changed, position="last")
        self.viewer.dims.events.camera.connect(self._view_changed, position="last")
        self.viewer.dims.events.camera.connect(self.camera_change, position="last")
        self.viewer.events.reset_view.connect(self._view_changed, position="last")

    def camera_change(self, _args):
        self.old_scene.transform.changed.disconnect(self._view_changed)
        self.old_scene: BaseCamera = self.viewer_widget.view.camera
        self.old_scene.transform.changed.connect(self._view_changed, position="last")

    def _view_changed(self, _args):
        self.view_changed.emit()

    def get_state(self):
        return {
            "ndisplay": self.viewer.dims.ndisplay,
            "point": self.viewer.dims.point,
            "camera": self.viewer_widget.view.camera.get_state(),
        }

    def set_state(self, dkt):
        if "ndisplay" in dkt and self.viewer.dims.ndisplay != dkt["ndisplay"]:
            self.viewer.dims.ndisplay = dkt["ndisplay"]
            return
        if "point" in dkt:
            for i, val in enumerate(dkt["point"]):
                self.viewer.dims.set_point(i, val)
        if "camera" in dkt:
            try:
                self.viewer_widget.view.camera.set_state(dkt["camera"])
            except KeyError:
                pass

    def change_mask_visibility(self):
        for image_info in self.image_info.values():
            if image_info.mask is not None:
                image_info.mask.visible = self.mask_chk.isChecked()

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
        control_points = [0] + list(np.linspace(1 / (2 * colors.shape[0]), 1, endpoint=True, num=colors.shape[0]))
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
        try:
            max_num = max(1, image_info.segmentation_count)
        except ValueError:
            max_num = 1
        if self.image_state.only_borders:
            data = calculate_borders(
                image_info.segmentation_info.segmentation,
                self.image_state.borders_thick // 2,
                self.viewer.dims.ndisplay == 2,
            )
            image_info.segmentation = self.viewer.add_image(
                data, scale=image_info.image.normalized_scaling(), contrast_limits=[0, max_num],
            )
        else:
            image_info.segmentation = self.viewer.add_image(
                image_info.segmentation_info.segmentation,
                scale=image_info.image.normalized_scaling(),
                contrast_limits=[0, max_num],
                name="segmentation",
                blending="translucent",
            )
        image_info.segmentation._interpolation[3] = Interpolation3D.NEAREST

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
        layer.visible = self.mask_chk.isChecked()
        image_info.mask = layer

    def update_mask_parameters(self):
        opacity = self.mask_opacity()
        colormap = self.mask_color()
        for image_info in self.image_info.values():
            if image_info.mask is not None:
                image_info.mask.opacity = opacity
                image_info.mask.colormap = colormap

    def set_image(self, image: Optional[Image] = None):
        # self.viewer.layers.select_all()
        # self.viewer.layers.remove_selected()
        layer_list = list(self.viewer.layers)
        if image is None:
            image = self.settings.image
        self.image_info = {}
        self.viewer.dims.set_point(
            image.time_pos,
            min(self.viewer.dims.point[image.time_pos], image.times * image.normalized_scaling()[image.time_pos] // 2),
        )
        self.viewer.dims.set_point(
            image.stack_pos,
            min(
                self.viewer.dims.point[image.stack_pos], image.layers * image.normalized_scaling()[image.stack_pos] // 2
            ),
        )

        try:
            image = self.add_image(image)
        except IndexError:
            for el in layer_list:
                el.selected = True
            self.viewer.layers.remove_selected()

        # self.viewer.stack_view()
        self.viewer.layers.unselect_all()
        for el in layer_list:
            index = self.viewer.layers.index(el)
            self.viewer.layers.move_selected(index, len(self.viewer.layers))

        self.viewer.layers.unselect_all()
        for el in layer_list:
            el.selected = True

        self.viewer.layers.remove_selected()
        self.viewer.reset_view()
        if len(self.viewer.layers):
            self.viewer.layers[-1].selected = True
        self.viewer.dims.set_point(image.time_pos, image.times * image.normalized_scaling()[image.time_pos] // 2)
        self.viewer.dims.set_point(image.stack_pos, image.layers * image.normalized_scaling()[image.stack_pos] // 2)

    def has_image(self, image: Image):
        return image.file_path in self.image_info

    @staticmethod
    def calculate_filter(array: np.ndarray, parameters: Tuple[NoiseFilterType, float]) -> np.ndarray:
        if parameters[0] == NoiseFilterType.No or parameters[1] == 0:
            return array
        if parameters[0] == NoiseFilterType.Gauss:
            return gaussian(array, parameters[1])
        else:
            return median(array, int(parameters[1]))

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
        limits = self.channel_control.get_limits()
        limits = [image.get_ranges()[i] if x is None else x for i, x in enumerate(limits)]
        gamma = self.channel_control.get_gamma()
        filters = self.channel_control.get_filter()
        image_layers = []

        for i in range(image.channels):
            lim = list(limits[i])
            if lim[1] == lim[0]:
                lim[1] += 1
            blending = "additive" if self.image_info or i != 0 else "translucent"
            # FIXME detect layer order impact on representation.
            image_layers.append(
                NapariImage(
                    self.calculate_filter(image.get_channel(i), filters[i]),
                    colormap=self.convert_to_vispy_colormap(self.channel_control.selected_colormaps[i]),
                    visible=visibility[i],
                    blending=blending,
                    scale=image.normalized_scaling(),
                    contrast_limits=lim,
                    gamma=gamma[i],
                    name=f"channel {i}; {len(self.viewer.layers) + i}",
                )
            )
        for el in image_layers:
            self.viewer.add_layer(el)
        self.image_info[image.file_path] = ImageInfo(image, image_layers, filters)
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

    @staticmethod
    def _shift_layer(layer: Layer, translate_2d):
        translate = [0] * layer.ndim
        translate[-2:] = translate_2d
        layer.translate_grid = translate

    def grid_view(self):
        """Present multiple images in grid view"""
        n_row = np.ceil(np.sqrt(len(self.image_info))).astype(int)
        n_row = max(1, n_row)
        scene_size, _ = self.images_bounds()
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
                image_info.layers[index].visible = self.channel_control.channel_visibility[index]
                if self.channel_control.channel_visibility[index]:
                    image_info.layers[index].colormap = self.convert_to_vispy_colormap(
                        self.channel_control.selected_colormaps[index]
                    )
                    limits = self.channel_control.get_limits()[index]
                    limits = image_info.image.get_ranges()[index] if limits is None else limits
                    image_info.layers[index].contrast_limits = limits
                    image_info.layers[index].gamma = self.channel_control.get_gamma()[index]
                    filter_type = self.channel_control.get_filter()[index]
                    if filter_type != image_info.filter_info[index]:
                        image_info.layers[index].data = self.calculate_filter(
                            image_info.image.get_channel(index), filter_type
                        )
                        image_info.filter_info[index] = filter_type

    def reset_image_size(self):
        self.viewer.reset_view()

    def set_theme(self, theme: str):
        self.viewer.theme = theme


class NapariQtViewer(QtViewer):
    def dragEnterEvent(self, event):  # pylint: disable=R0201

        """
        ignore napari reading mechanism
        """
        event.ignore()
