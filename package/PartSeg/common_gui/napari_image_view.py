import itertools
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from PartSegCore.class_generator import enum_register

try:
    from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton
except ImportError:
    from napari._qt.qt_viewer_buttons import QtViewerPushButton

from napari.components import ViewerModel as Viewer
from napari.layers import Layer, Points
from napari.layers.image import Image as NapariImage
from napari.layers.image._image_constants import Interpolation3D
from napari.layers.labels import Labels
from napari.qt import QtNDisplayButton, QtViewer
from napari.qt.threading import thread_worker
from qtpy.QtCore import QEvent, QObject, QPoint, Qt, Signal
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QMenu, QToolTip, QVBoxLayout, QWidget
from vispy.color import Color, ColorArray, Colormap
from vispy.scene import BaseCamera

from PartSegCore.color_image import calculate_borders
from PartSegCore.image_operations import NoiseFilterType, gaussian, median
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image

from ..common_backend.base_settings import BaseSettings, ViewSettings
from .channel_control import ChannelProperty, ColorComboBoxGroup

ORDER_DICT = {"xy": [0, 1, 2, 3], "zy": [0, 2, 1, 3], "zx": [0, 3, 1, 2]}
NEXT_ORDER = {"xy": "zy", "zy": "zx", "zx": "xy"}


@dataclass
class ImageInfo:
    image: Image
    layers: List[NapariImage]
    filter_info: List[Tuple[NoiseFilterType, float]] = field(default_factory=list)
    mask: Optional[Labels] = None
    mask_array: Optional[np.ndarray] = None
    roi: Optional[Labels] = None
    roi_info: ROIInfo = field(default_factory=lambda: ROIInfo(None))
    roi_count: int = 0

    def coords_in(self, coords: Union[List[int], np.ndarray]) -> bool:
        if not self.layers:
            return False
        fst_layer = self.layers[0]
        moved_coords = self.translated_coords(coords)
        return np.all(moved_coords >= 0) and np.all(moved_coords < fst_layer.data.shape)

    def translated_coords(self, coords: Union[List[int], np.ndarray]) -> np.ndarray:
        if not self.layers:
            return np.array(coords)
        fst_layer = self.layers[0]
        return np.subtract(coords, fst_layer.translate_grid).astype(int)


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
    roi_presented_changed = Signal()
    # changed and image need to be refreshed

    def __init__(self, settings: ViewSettings, name: str):
        if len(name) == 0:
            raise ValueError("Name string should be not empty")
        super().__init__()
        self.name = name
        self.settings = settings
        self.opacity = settings.get_from_profile(f"{name}.image_state.opacity", 1.0)
        self.show_label = settings.get_from_profile(f"{name}.image_state.show_label", LabelEnum.Show_results)
        self.only_borders = settings.get_from_profile(f"{name}.image_state.only_border", True)
        self.borders_thick = settings.get_from_profile(f"{name}.image_state.border_thick", 1)
        self.roi_presented = "ROI"

    def set_roi_presented(self, val):
        self.roi_presented = val
        self.roi_presented_changed.emit()

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
    image_added = Signal()

    def __init__(
        self,
        settings: BaseSettings,
        channel_property: ChannelProperty,
        name: str,
        parent: Optional[QWidget] = None,
        ndisplay=2,
    ):
        super().__init__(parent=parent)

        self.settings = settings
        self.channel_property = channel_property
        self.name = name
        self.image_info: Dict[str, ImageInfo] = {}
        self.current_image = ""
        self._current_order = "xy"
        self.components = None
        self.worker_list = []
        self.points_layer = None

        self.viewer = Viewer(ndisplay=ndisplay)
        self.viewer.theme = self.settings.theme_name
        self.viewer_widget = NapariQtViewer(self.viewer)
        self.image_state = ImageShowState(settings, name)
        self.channel_control = ColorComboBoxGroup(settings, name, channel_property, height=30)
        self.ndim_btn = QtNDisplayButton(self.viewer)
        self.reset_view_button = QtViewerPushButton(self.viewer, "home", "Reset view", self._reset_view)
        self.points_view_button = QtViewerPushButton(
            self.viewer, "new_points", "Show points", self.toggle_points_visibility
        )
        self.roll_dim_button = QtViewerPushButton(self.viewer, "roll", "Roll dimension", self._rotate_dim)
        self.roll_dim_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roll_dim_button.customContextMenuRequested.connect(self._dim_order_menu)
        self.mask_chk = QCheckBox()
        self.mask_label = QLabel("Mask:")

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.reset_view_button)
        self.btn_layout.addWidget(self.ndim_btn)
        self.btn_layout.addWidget(self.roll_dim_button)
        self.btn_layout.addWidget(self.points_view_button)
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
        settings.mask_representation_changed.connect(self.update_mask_parameters)
        settings.roi_changed.connect(self.set_roi)
        settings.roi_clean.connect(self.set_roi)
        settings.image_changed.connect(self.set_image)
        settings.image_spacing_changed.connect(self.update_spacing_info)
        settings.points_changed.connect(self.update_points)
        # settings.labels_changed.connect(self.paint_layer)
        self.old_scene: BaseCamera = self.viewer_widget.view.scene

        self.image_state.coloring_changed.connect(self.update_roi_coloring)
        self.image_state.roi_presented_changed.connect(self.update_roi_representation)
        self.image_state.borders_changed.connect(self.update_roi_representation)
        self.mask_chk.stateChanged.connect(self.change_mask_visibility)
        self.viewer_widget.view.scene.transform.changed.connect(self._view_changed, position="last")
        try:
            self.viewer.dims.events.current_step.connect(self._view_changed, position="last")
        except AttributeError:
            self.viewer.dims.events.axis.connect(self._view_changed, position="last")
        self.viewer.dims.events.ndisplay.connect(self._view_changed, position="last")
        self.viewer.dims.events.ndisplay.connect(self._view_changed, position="last")
        self.viewer.dims.events.ndisplay.connect(self.camera_change, position="last")
        self.viewer.events.reset_view.connect(self._view_changed, position="last")

    def toggle_points_visibility(self):
        if self.points_layer is not None:
            self.points_layer.visible = not self.points_layer.visible

    def _dim_order_menu(self, point: QPoint):
        menu = QMenu()
        for key in ORDER_DICT:
            action = menu.addAction(key)
            action.triggered.connect(partial(self._set_new_order, key))
            if key == self._current_order:
                font = action.font()
                font.setBold(True)
                action.setFont(font)

        menu.exec_(self.roll_dim_button.mapToGlobal(point))

    def _set_new_order(self, text: str):
        self._current_order = text
        self.viewer.dims.order = ORDER_DICT[text]
        self.update_roi_representation()

    def _reset_view(self):
        self._set_new_order("xy")
        self.viewer.dims.order = ORDER_DICT[self._current_order]
        self.viewer.reset_view()

    def _rotate_dim(self):
        self._set_new_order(NEXT_ORDER[self._current_order])

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

        if image_info.roi is not None:
            image_info.roi.scale = image.normalized_scaling()

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
            if image_info.roi_info.roi is not None and image_info.roi is not None:
                val = image_info.roi_info.roi[tuple(moved_coords)]
                if val:
                    components.append(val)

        if not bright_array and not components:
            self.text_info_change.emit("")
            return
        text = f"{cords}: "
        if bright_array:
            text += str(bright_array[0]) if len(bright_array) == 1 else str(bright_array)
        self.components = components
        if components:
            if len(components) == 1:
                text += f" component: {components[0]}"
            else:
                text += f" components: {components}"
        self.text_info_change.emit(text)

    def get_control_view(self) -> ImageShowState:
        return self.image_state

    def mask_opacity(self) -> float:
        """Get mask opacity"""
        return self.settings.get_from_profile("mask_presentation_opacity", 1)

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

    def update_points(self):
        if self.settings.points is not None:
            self.points_view_button.setVisible(True)
            if self.points_layer is None or self.points_layer not in self.viewer.layers:
                self.points_layer = Points(self.settings.points, scale=self.settings.image.normalized_scaling())
                self.viewer.add_layer(self.points_layer)
            else:
                self.points_layer.data = self.settings.points
                self.points_layer.scale = self.settings.image.normalized_scaling()
        else:
            if self.points_layer is not None and self.points_layer in self.viewer.layers:
                self.points_view_button.setVisible(False)
                self.points_layer.data = np.empty((0, 4))

    def set_roi(self, roi_info: Optional[ROIInfo] = None, image: Optional[Image] = None) -> None:
        image = self.get_image(image)
        if roi_info is None:
            roi_info = self.settings.roi_info
        image_info = self.image_info[image.file_path]
        if image_info.roi is not None:
            self.viewer.layers.unselect_all()
            image_info.roi.selected = True
            self.viewer.layers.remove_selected()
            image_info.roi = None

        if roi_info.roi is None:
            return

        image_info.roi_info = roi_info
        image_info.roi_count = max(roi_info.bound_info) if roi_info.bound_info else 0
        self.add_roi_layer(image_info)
        image_info.roi.colormap = self.get_roi_view_parameters(image_info)
        image_info.roi.opacity = self.image_state.opacity

    def get_roi_view_parameters(self, image_info: ImageInfo) -> Colormap:
        colors = self.settings.label_colors / 255
        if self.image_state.show_label == LabelEnum.Not_show or image_info.roi_count == 0 or colors.size == 0:
            colors = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        else:
            repeat = int(np.ceil(image_info.roi_count / colors.shape[0]))
            colors = np.concatenate([colors] * repeat)
            colors = np.concatenate([colors, np.ones(colors.shape[0]).reshape(colors.shape[0], 1)], axis=1)
            colors = np.concatenate([[[0, 0, 0, 0]], colors[: image_info.roi_count]])
            if self.image_state.show_label == LabelEnum.Show_selected:
                try:
                    colors *= self.settings.components_mask().reshape((colors.shape[0], 1))
                except ValueError:
                    pass
        control_points = [0] + list(np.linspace(1 / (2 * colors.shape[0]), 1, endpoint=True, num=colors.shape[0]))
        return Colormap(colors, controls=control_points, interpolation="zero")

    def update_roi_coloring(self):
        for image_info in self.image_info.values():
            if image_info.roi is None:
                continue
            image_info.roi.colormap = self.get_roi_view_parameters(image_info)
            image_info.roi.opacity = self.image_state.opacity

    def remove_all_roi(self):
        self.viewer.layers.unselect_all()
        for image_info in self.image_info.values():
            if image_info.roi is None:
                continue
            image_info.roi.selected = True
            image_info.roi = None

        self.viewer.layers.remove_selected()

    def add_roi_layer(self, image_info: ImageInfo):
        if image_info.roi_info.roi is None:
            return
        try:
            max_num = max(1, image_info.roi_count)
        except ValueError:
            max_num = 1
        roi = image_info.roi_info.alternative.get(self.image_state.roi_presented, image_info.roi_info.roi)
        if self.image_state.only_borders:

            data = calculate_borders(
                roi.transpose(ORDER_DICT[self._current_order]),
                self.image_state.borders_thick // 2,
                self.viewer.dims.ndisplay == 2,
            ).transpose(np.argsort(ORDER_DICT[self._current_order]))
            image_info.roi = self.viewer.add_image(
                data,
                scale=image_info.image.normalized_scaling(),
                contrast_limits=[0, max_num],
            )
        else:
            image_info.roi = self.viewer.add_image(
                roi,
                scale=image_info.image.normalized_scaling(),
                contrast_limits=[0, max_num],
                name="ROI",
                blending="translucent",
            )
        image_info.roi._interpolation[3] = Interpolation3D.NEAREST

    def update_roi_representation(self):
        self.remove_all_roi()

        for image_info in self.image_info.values():
            self.add_roi_layer(image_info)

        self.update_roi_coloring()

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
        self.image_info = {}
        self.add_image(image, True)

    def has_image(self, image: Image):
        return image.file_path in self.image_info

    @staticmethod
    def calculate_filter(array: np.ndarray, parameters: Tuple[NoiseFilterType, float]) -> Optional[np.ndarray]:
        if parameters[0] == NoiseFilterType.No or parameters[1] == 0:
            return array
        if parameters[0] == NoiseFilterType.Gauss:
            return gaussian(array, parameters[1])
        return median(array, int(parameters[1]))

    def _remove_worker(self, sender):
        for worker in self.worker_list:
            if sender is worker.signals:
                self.worker_list.remove(worker)
                break
        else:
            print("[_remove_worker]", sender)

    def _add_layer_util(self, index, layer, filters):
        self.viewer.add_layer(layer)

        def set_data(val):
            self._remove_worker(self.sender())
            data_, layer_ = val
            if data_ is None:
                return
            if layer_ not in self.viewer.layers:
                return
            layer_.data = data_

        @thread_worker(connect={"returned": set_data})
        def calc_filter(j, layer_):
            if filters[j][0] == NoiseFilterType.No or filters[j][1] == 0:
                return None, layer_
            return self.calculate_filter(layer_.data, parameters=filters[j]), layer_

        worker = calc_filter(index, layer)
        self.worker_list.append(worker)

    def _add_image(self, image_data: Tuple[ImageInfo, bool]):
        self._remove_worker(self.sender())

        image_info, replace = image_data
        image = image_info.image
        if replace:
            self.viewer.layers.select_all()
            self.viewer.layers.remove_selected()

        filters = self.channel_control.get_filter()
        for i, layer in enumerate(image_info.layers):
            try:
                self._add_layer_util(i, layer, filters)
            except AssertionError:
                layer.colormap = "gray"
                self._add_layer_util(i, layer, filters)

        self.image_info[image.file_path].filter_info = filters
        self.image_info[image.file_path].layers = image_info.layers
        self.current_image = image.file_path
        self.viewer.reset_view()
        if self.viewer.layers:
            self.viewer.layers[-1].selected = True

        for i, axis in enumerate(image.axis_order):
            if axis == "C":
                continue
            self.viewer.dims.set_point(i, image.shape[i] * image.normalized_scaling()[i] // 2)
        if self.image_info[image.file_path].roi is not None:
            self.set_roi()
        if image_info.image.mask is not None:
            self.set_mask()
        self.image_added.emit()

    def add_image(self, image: Optional[Image], replace=False):
        if image is None:
            image = self.settings.image

        if not image.channels:
            raise ValueError("Need non empty image")

        if image.file_path in self.image_info:
            raise ValueError("Image already added")

        self.image_info[image.file_path] = ImageInfo(image, [])

        channels = image.channels
        if self.image_info and not replace:
            channels = max(channels, *[x.image.channels for x in self.image_info.values()])

        self.channel_control.set_channels(channels)
        visibility = self.channel_control.channel_visibility
        limits = self.channel_control.get_limits()
        ranges = image.get_ranges()
        limits = [ranges[i] if x is None else x for i, x in zip(range(image.channels), limits)]
        gamma = self.channel_control.get_gamma()
        colormaps = [self.channel_control.selected_colormaps[i] for i in range(image.channels)]
        parameters = ImageParameters(
            limits, visibility, gamma, colormaps, image.normalized_scaling(), len(self.viewer.layers)
        )

        self._prepare_layers(image, parameters, replace)

        return image

    def _prepare_layers(self, image, parameters, replace):
        worker = prepare_layers(image, parameters, replace)
        worker.returned.connect(self._add_image)
        self.worker_list.append(worker)
        worker.start()

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

            if image_info.roi is not None:
                self._shift_layer(image_info.roi, translate_2d)
        self.viewer.reset_view()

    def change_visibility(self, name: str, index: int):
        for image_info in self.image_info.values():
            if len(image_info.layers) > index:
                image_info.layers[index].visible = self.channel_control.channel_visibility[index]
                if self.channel_control.channel_visibility[index]:
                    image_info.layers[index].colormap = self.channel_control.selected_colormaps[index]
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

    def closeEvent(self, event):
        for worker in self.worker_list:
            worker.quit()
        super().closeEvent(event)

    def get_tool_tip_text(self) -> str:
        image = self.settings.image
        image_info = self.image_info[image.file_path]
        text_list = [_print_dict(image_info.roi_info.annotations.get(el, {})) for el in self.components]

        return " ".join(text_list)

    def event(self, event: QEvent):
        if event.type() == QEvent.ToolTip and self.components:
            text = self.get_tool_tip_text()
            if text:
                QToolTip.showText(event.globalPos(), text)
        return super().event(event)


class NapariQtViewer(QtViewer):
    def dragEnterEvent(self, event):  # pylint: disable=R0201
        """
        ignore napari reading mechanism
        """
        event.ignore()


@dataclass
class ImageParameters:
    limits: List[Tuple[float, float]]
    visibility: List[bool]
    gamma: List[float]
    colormaps: List[Colormap]
    scaling: Tuple[Union[float, int]]
    layers: int = 0


def _prepare_layers(image: Image, param: ImageParameters, replace: bool) -> Tuple[ImageInfo, bool]:
    image_layers = []
    for i in range(image.channels):
        lim = list(param.limits[i])
        if lim[1] == lim[0]:
            lim[1] += 1
        blending = "additive" if i != 0 else "translucent"
        data = image.get_channel(i)

        layer = NapariImage(
            data,
            colormap=param.colormaps[i],
            visible=param.visibility[i],
            blending=blending,
            scale=param.scaling,
            contrast_limits=lim,
            gamma=param.gamma[i],
            name=f"channel {i}; {param.layers + i}",
        )
        image_layers.append(layer)
    return ImageInfo(image, image_layers, []), replace


prepare_layers = thread_worker(_prepare_layers)


def _print_dict(dkt: dict, indent=""):
    res = []
    for k, v in dkt.items():
        if isinstance(v, dict):
            res.append(f"{indent}{k}:\n{_print_dict(v, indent+'  ')}")
        else:
            res.append(f"{indent}{k}: {v}")
    return "\n".join(res)


enum_register.register_class(LabelEnum)
