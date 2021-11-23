import itertools
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Dict, List, MutableMapping, Optional, Tuple, Union

import napari
import numpy as np
import qtawesome as qta
from napari.components import ViewerModel as Viewer
from napari.layers import Layer, Points
from napari.layers.image import Image as NapariImage
from napari.layers.labels import Labels
from napari.qt import QtStateButton, QtViewer
from napari.qt.threading import thread_worker
from packaging.version import parse as parse_version
from qtpy.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QMenu, QPushButton, QSpinBox, QToolTip, QVBoxLayout, QWidget
from superqt import QEnumComboBox, ensure_main_thread
from vispy.color import Color, Colormap
from vispy.geometry.rect import Rect
from vispy.scene import BaseCamera

from PartSegCore.class_generator import enum_register
from PartSegCore.color_image import calculate_borders
from PartSegCore.image_operations import NoiseFilterType, gaussian, median
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image

from ..common_backend.base_settings import BaseSettings
from .advanced_tabs import RENDERING_LIST, RENDERING_MODE_NAME
from .channel_control import ChannelProperty, ColorComboBoxGroup
from .qt_modal import QtPopup

try:
    from napari._qt.qt_viewer_buttons import QtViewerPushButton
except ImportError:
    from napari._qt.widgets.qt_viewer_buttons import QtViewerPushButton


class QtNDisplayButton(QtStateButton):
    def __init__(self, viewer):
        super().__init__(
            "ndisplay_button",
            viewer.dims,
            "ndisplay",
            viewer.dims.events.ndisplay,
            2,
            3,
        )


ORDER_DICT = {"xy": [0, 1, 2, 3], "zy": [0, 2, 1, 3], "zx": [0, 3, 1, 2]}
NEXT_ORDER = {"xy": "zy", "zy": "zx", "zx": "xy"}

ColorInfo = Dict[int, Union[str, List[float]]]

napari_rendering = parse_version(napari.__version__) >= parse_version("0.4.11")


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
        if self.value == 0:  # pylint: disable=W0143
            return "Don't show"
        return self.name.replace("_", " ")


class SearchType(Enum):
    Highlight = 0
    Zoom_in = 1


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
        self.roi_alternative_selection = "ROI"
        self.additional_layers: List[Layer] = []
        self._search_type = SearchType.Highlight
        self._last_component = 1

        self.viewer = Viewer(ndisplay=ndisplay)
        self.viewer.theme = self.settings.theme_name
        self.viewer_widget = NapariQtViewer(self.viewer)
        if hasattr(self.viewer_widget.canvas, "background_color_override"):
            self.viewer_widget.canvas.background_color_override = "black"
        self.channel_control = ColorComboBoxGroup(settings, name, channel_property, height=30)
        self.ndim_btn = QtNDisplayButton(self.viewer)
        self.reset_view_button = QtViewerPushButton(self.viewer, "home", "Reset view", self._reset_view)
        self.points_view_button = QtViewerPushButton(
            self.viewer, "new_points", "Show points", self.toggle_points_visibility
        )
        self.search_roi_btn = SearchROIButton(self)
        # self.search_roi_btn.setDisabled(True)
        self.search_roi_btn.clicked.connect(self._search_component)
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
        self.btn_layout.addWidget(self.search_roi_btn)
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
        settings.roi_changed.connect(self.set_roi)
        settings.roi_clean.connect(self.set_roi)
        settings.image_changed.connect(self.set_image)
        settings.image_spacing_changed.connect(self.update_spacing_info)
        settings.points_changed.connect(self.update_points)
        settings.connect_to_profile(RENDERING_MODE_NAME, self.update_rendering)
        settings.labels_changed.connect(self.update_roi_coloring)
        settings.connect_to_profile(f"{name}.image_state.opacity", self.update_roi_coloring)
        settings.connect_to_profile(f"{name}.image_state.only_border", self.update_roi_border)
        settings.connect_to_profile(f"{name}.image_state.border_thick", self.update_roi_border)
        settings.connect_to_profile(f"{name}.image_state.show_label", self.update_roi_labeling)
        settings.connect_to_profile("mask_presentation_opacity", self.update_mask_parameters)
        settings.connect_to_profile("mask_presentation_color", self.update_mask_parameters)
        # settings.labels_changed.connect(self.paint_layer)
        self.old_scene: BaseCamera = self.viewer_widget.view.scene

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
            with suppress(KeyError):
                self.viewer_widget.view.camera.set_state(dkt["camera"])

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

    def _active_layer(self):
        if hasattr(self.viewer.layers, "selection"):
            return self.viewer.layers.selection.active
        return self.viewer.active_layer

    def _coordinates(self):
        active_layer = self._active_layer()
        if active_layer is None:
            return
        if (
            hasattr(self.viewer, "cursor")
            and hasattr(self.viewer.cursor, "position")
            and hasattr(active_layer, "world_to_data")
        ):
            return [int(x) for x in active_layer.world_to_data(self.viewer.cursor.position)]
        return [int(x) for x in active_layer.coordinates]

    def print_info(self, event=None):
        cords = self._coordinates()
        if cords is None:
            return
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

    def mask_opacity(self) -> float:
        """Get mask opacity"""
        return self.settings.get_from_profile("mask_presentation_opacity", 1)

    def mask_color(self) -> ColorInfo:
        """Get mask marking color"""
        color = Color(np.divide(self.settings.get_from_profile("mask_presentation_color", [255, 255, 255]), 255))
        return {0: "black", 1: color.rgba}

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
        elif self.points_layer is not None and self.points_layer in self.viewer.layers:
            self.points_view_button.setVisible(False)
            self.points_layer.data = np.empty((0, 4))

    def set_roi(self, roi_info: Optional[ROIInfo] = None, image: Optional[Image] = None) -> None:
        image = self.get_image(image)
        if roi_info is None:
            roi_info = self.settings.roi_info
        image_info = self.image_info[image.file_path]
        if image_info.roi is not None and image_info.roi in self.viewer.layers:
            if hasattr(self.viewer.layers, "selection"):
                self.viewer.layers.selection.clear()
                self.viewer.layers.selection.add(image_info.roi)
            else:
                self.viewer.layers.unselect_all()
                image_info.roi.selected = True
            self.viewer.layers.remove_selected()
            image_info.roi = None

        image_info.roi_info = roi_info
        image_info.roi_count = max(roi_info.bound_info) if roi_info.bound_info else 0

        if roi_info.roi is None:
            self.search_roi_btn.setDisabled(True)
            return
        self.search_roi_btn.setDisabled(False)

        self.add_roi_layer(image_info)
        image_info.roi.color = self.get_roi_view_parameters(image_info)
        image_info.roi.opacity = self.settings.get_from_profile(f"{self.name}.image_state.opacity", 1.0)

    def get_roi_view_parameters(self, image_info: ImageInfo) -> ColorInfo:
        colors = self.settings.label_colors / 255
        if (
            self.settings.get_from_profile(f"{self.name}.image_state.show_label", LabelEnum.Show_results)
            == LabelEnum.Not_show
            or image_info.roi_count == 0
            or colors.size == 0
        ):
            return {x: [0, 0, 0, 0] for x in range(image_info.roi_count + 1)}

        res = {x: colors[(x - 1) % colors.shape[0]] for x in range(image_info.roi_count + 1)}
        res[0] = [0, 0, 0, 0]
        return res

    def update_roi_coloring(self):
        for image_info in self.image_info.values():
            if image_info.roi is None:
                continue
            image_info.roi.color = self.get_roi_view_parameters(image_info)
            image_info.roi.opacity = self.settings.get_from_profile(f"{self.name}.image_state.opacity", 1.0)

    def remove_all_roi(self):
        if hasattr(self.viewer.layers, "selection"):
            self.viewer.layers.selection.clear()
        else:
            self.viewer.layers.unselect_all()
        for image_info in self.image_info.values():
            if image_info.roi is None:
                continue
            if hasattr(self.viewer.layers, "selection"):
                self.viewer.layers.selection.add(image_info.roi)
            else:
                image_info.roi.selected = True
            image_info.roi = None

        self.viewer.layers.remove_selected()

    def update_roi_border(self) -> None:
        for image_info in self.image_info.values():
            if image_info.roi is None:
                continue
            roi = image_info.roi_info.alternative.get(self.roi_alternative_selection, image_info.roi_info.roi)
            border_thick = self.settings.get_from_profile(f"{self.name}.image_state.border_thick", 1) == 1
            only_border = self.settings.get_from_profile(f"{self.name}.image_state.only_border", True)
            alternative = image_info.roi.metadata.get("alternative", self.roi_alternative_selection)
            if only_border and border_thick != 1:
                data = calculate_borders(
                    roi.transpose(ORDER_DICT[self._current_order]),
                    self.settings.get_from_profile(f"{self.name}.image_state.border_thick", 1) // 2,
                    self.viewer.dims.ndisplay == 2,
                ).transpose(np.argsort(ORDER_DICT[self._current_order]))
                image_info.roi.data = data
            else:
                if image_info.roi.metadata.get("border_thick", 1) != 1 or alternative != self.roi_alternative_selection:
                    image_info.roi.data = roi
                image_info.roi.contour = only_border

            image_info.roi.metadata["border_thick"] = border_thick
            image_info.roi.metadata["alternative"] = self.roi_alternative_selection

    @ensure_main_thread
    def update_rendering(self):
        rendering = self.settings.get_from_profile(RENDERING_MODE_NAME, RENDERING_LIST[0])
        for image_info in self.image_info.values():
            if image_info.roi is not None and hasattr(image_info.roi, "rendering"):
                image_info.roi.rendering = rendering

    @ensure_main_thread
    def update_roi_labeling(self):
        for image_info in self.image_info.values():
            if image_info.roi is not None:
                image_info.roi.color = self.get_roi_view_parameters(image_info)

    def add_roi_layer(self, image_info: ImageInfo):
        if image_info.roi_info.roi is None:
            return
        roi = image_info.roi_info.alternative.get(self.roi_alternative_selection, image_info.roi_info.roi)
        border_thick = self.settings.get_from_profile(f"{self.name}.image_state.border_thick", 1) == 1
        kwargs = {
            "scale": image_info.image.normalized_scaling(),
            "name": "ROI",
            "blending": "translucent",
            "metadata": {"border_thick": border_thick, "alternative": self.roi_alternative_selection},
        }
        if napari_rendering:
            kwargs["rendering"] = self.settings.get_from_profile(RENDERING_MODE_NAME, RENDERING_LIST[0])

        only_border = self.settings.get_from_profile(f"{self.name}.image_state.only_border", True)
        if only_border and border_thick != 1:

            data = calculate_borders(
                roi.transpose(ORDER_DICT[self._current_order]),
                border_thick // 2,
                self.viewer.dims.ndisplay == 2,
            ).transpose(np.argsort(ORDER_DICT[self._current_order]))
            image_info.roi = self.viewer.add_labels(data, **kwargs)
            image_info.roi.contour = False
        else:
            image_info.roi = self.viewer.add_labels(roi, **kwargs)
            image_info.roi.contour = only_border

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
            if hasattr(self.viewer.layers, "selection"):
                self.viewer.layers.selection.clear()
                self.viewer.layers.selection.add(image_info.mask)
            else:
                self.viewer.layers.unselect_all()
                image_info.mask.selected = True
            self.viewer.layers.remove_selected()
            image_info.mask = None

        if mask is None:
            return

        mask_marker = mask == 0

        layer = self.viewer.add_labels(mask_marker, scale=image.normalized_scaling(), blending="additive", name="Mask")
        layer.color = self.mask_color()
        layer.opacity = self.mask_opacity()
        layer.visible = self.mask_chk.isChecked()
        image_info.mask = layer

    def update_mask_parameters(self):
        opacity = self.mask_opacity()
        colormap = self.mask_color()
        for image_info in self.image_info.values():
            if image_info.mask is not None:
                image_info.mask.opacity = opacity
                image_info.mask.color = colormap

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
            if hasattr(self.viewer.layers, "selection"):
                self.viewer.layers.selection.clear()
                self.viewer.layers.selection.add(self.viewer.layers[-1])
            else:
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
            channels = max(channels, *(x.image.channels for x in self.image_info.values()))

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
        self.viewer.layers.clear()
        self.viewer_widget.close()
        super().closeEvent(event)

    def get_tool_tip_text(self) -> str:
        image = self.settings.image
        image_info = self.image_info[image.file_path]
        text_list = []
        for el in self.components:
            data = image_info.roi_info.annotations.get(el, {})
            if data:
                try:
                    text_list.append(_print_dict(data))
                except ValueError:  # pragma: no cover
                    logging.warning("Wrong value provided as layer annotation.")
        return " ".join(text_list)

    def event(self, event: QEvent):
        if event.type() == QEvent.ToolTip and self.components:
            text = self.get_tool_tip_text()
            if text:
                QToolTip.showText(event.globalPos(), text)
        return super().event(event)

    def _search_component(self):
        max_components = max(max(image_info.roi_info.bound_info) for image_info in self.image_info.values())

        dial = SearchComponentModal(self, self._search_type, self._last_component, max_components)
        dial.show_right_of_mouse()

    def component_unmark(self, _num):
        self.viewer.layers.selection.clear()
        for el in self.additional_layers:
            if "timer" in el.metadata:
                el.metadata["timer"].stop()
            self.viewer.layers.selection.add(el)
        self.viewer.layers.remove_selected()
        self.additional_layers = []

    def _mark_layer(self, num: int, flash: bool, image_info: ImageInfo):
        bound_info = image_info.roi_info.bound_info.get(num, None)
        if bound_info is None:
            return
        # TODO think about marking on bright background
        slices = bound_info.get_slices()
        slices[image_info.image.stack_pos] = slice(None)
        component_mark = image_info.roi_info.roi[tuple(slices)] == num
        translate_grid = image_info.roi.translate_grid + (bound_info.lower) * image_info.roi.scale
        translate_grid[image_info.image.stack_pos] = 0
        self.additional_layers.append(
            self.viewer.add_labels(
                component_mark,
                scale=image_info.roi.scale,
                blending="additive",
                color={0: "black", 1: "white"},
                opacity=0.5,
            )
        )
        self.additional_layers[-1].translate_grid = translate_grid
        if flash:
            layer = self.additional_layers[-1]

            def flash_fun(layer_=layer):
                opacity = layer_.opacity + 0.1
                if opacity > 1:
                    opacity = 0.1
                layer_.opacity = opacity

            timer = QTimer()
            timer.setInterval(100)
            timer.timeout.connect(flash_fun)
            timer.start()
            layer.metadata["timer"] = timer

    def component_mark(self, num: int, flash: bool = False):
        self.component_unmark(num)
        self._search_type = SearchType.Highlight
        self._last_component = num

        bounding_box = self._bounding_box(num)
        if bounding_box is None:
            return

        for image_info in self.image_info.values():
            self._mark_layer(num, flash, image_info)

        lower_bound, upper_bound = bounding_box
        self._update_point(lower_bound, upper_bound)

        if self.viewer.dims.ndisplay == 2:
            l_bound = lower_bound[-2:][::-1]
            u_bound = upper_bound[-2:][::-1]
            rect = Rect(self.viewer_widget.view.camera.get_state()["rect"])
            if rect.contains(*l_bound) and rect.contains(*u_bound):
                return
            size = u_bound - l_bound
            rect.size = tuple(np.max([rect.size, size * 1.2], axis=0))
            pos = rect.pos
            if rect.left > l_bound[0]:
                pos = l_bound[0], pos[1]
            if rect.right < u_bound[0]:
                pos = pos[0] + u_bound[0] - rect.right, pos[1]
            if rect.bottom > l_bound[1]:
                pos = pos[0], l_bound[1]
            if rect.top < u_bound[1]:
                pos = pos[0], pos[1] + (u_bound[1] - rect.top)
            rect.pos = pos
            self.viewer_widget.view.camera.set_state({"rect": rect})

    def component_zoom(self, num):
        self.component_unmark(num)
        self._search_type = SearchType.Zoom_in
        self._last_component = num

        bounding_box = self._bounding_box(num)
        if bounding_box is None:
            return

        lower_bound, upper_bound = bounding_box
        diff = upper_bound - lower_bound
        frame = diff * 0.2
        if self.viewer.dims.ndisplay == 2:
            rect = Rect(pos=(lower_bound - frame)[-2:][::-1], size=(diff + 2 * frame)[-2:][::-1])
            self.set_state({"camera": {"rect": rect}})
        self._update_point(lower_bound, upper_bound)

    def _update_point(self, lower_bound, upper_bound):
        point = (lower_bound + upper_bound) / 2
        current_point = self.viewer.dims.point
        for i in range(self.viewer.dims.ndim - self.viewer.dims.ndisplay):
            if not (lower_bound[i] <= current_point[i] <= upper_bound[i]):
                self.viewer.dims.set_point(i, point[i])

    def _data_to_world(self, layer: Layer, cords):
        return layer._transforms[1:3].simplified(cords)

    def _bounding_box(self, num) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        lower_bound_list = []
        upper_bound_list = []
        for image_info in self.image_info.values():
            bound_info = image_info.roi_info.bound_info.get(num, None)
            if bound_info is None:
                continue
            lower_bound_list.append(self._data_to_world(image_info.roi, bound_info.lower))  # pylint: disable=W0212
            upper_bound_list.append(self._data_to_world(image_info.roi, bound_info.upper))  # pylint: disable=W0212

        if not lower_bound_list:
            return

        lower_bound = np.min(lower_bound_list, axis=0)
        upper_bound = np.min(upper_bound_list, axis=0)
        return lower_bound, upper_bound


class NapariQtViewer(QtViewer):
    def __init__(self, viewer):
        super().__init__(viewer, show_welcome_screen=False)
        self.widget(0).layout().setContentsMargins(0, 5, 0, 2)

    def dragEnterEvent(self, event):  # pylint: disable=R0201
        """
        ignore napari reading mechanism
        """
        event.ignore()

    def close(self):
        self.dockConsole.deleteLater()
        self.dockLayerList.deleteLater()
        self.dockLayerControls.deleteLater()
        if hasattr(self, "activityDock"):
            self.activityDock.deleteLater()
        return super().close()

    def closeEvent(self, event):
        self.close()
        super().closeEvent(event)


class SearchComponentModal(QtPopup):
    def __init__(self, image_view: ImageView, search_type: SearchType, component_num: int, max_components):
        super().__init__(image_view)
        self.image_view = image_view
        self.zoom_to = QEnumComboBox(self, SearchType)
        self.zoom_to.setCurrentEnum(search_type)
        self.zoom_to.currentEnumChanged.connect(self._component_num_changed)
        self.component_selector = QSpinBox()
        self.component_selector.valueChanged.connect(self._component_num_changed)
        self.component_selector.setMaximum(max_components)
        self.component_selector.setValue(component_num)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Component:"))
        layout.addWidget(self.component_selector)
        layout.addWidget(QLabel("Selection:"))
        layout.addWidget(self.zoom_to)
        self.frame.setLayout(layout)

    def _component_num_changed(self):
        if self.zoom_to.currentEnum() == SearchType.Highlight:
            self.image_view.component_mark(self.component_selector.value(), flash=True)
        else:
            self.image_view.component_zoom(self.component_selector.value())

    def closeEvent(self, event):
        super().closeEvent(event)
        self.image_view.component_unmark(0)


@dataclass
class ImageParameters:
    limits: List[Tuple[float, float]]
    visibility: List[bool]
    gamma: List[float]
    colormaps: List[Colormap]
    scaling: Tuple[Union[float, int]]
    layers: int = 0


class SearchROIButton(QPushButton):
    def __init__(self, image_view: ImageView):
        super().__init__(qta.icon("fa5s.search"), "")
        self.image_view = image_view
        image_view.settings.theme_changed.connect(self._theme_changed)
        self._theme_changed()

    def _theme_changed(self):
        color = self.image_view.settings.theme.text
        self.setIcon(qta.icon("fa5s.search", color=QColor(*color.as_rgb_tuple())))


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


def _print_dict(dkt: MutableMapping, indent="") -> str:
    if not isinstance(dkt, MutableMapping):
        logging.error(f"{type(dkt)} instead of dict passed to _print_dict")
        return indent + str(dkt)
    res = []
    for k, v in dkt.items():
        if isinstance(v, MutableMapping):
            res.append(f"{indent}{k}:\n{_print_dict(v, indent+'  ')}")
        else:
            res.append(f"{indent}{k}: {v}")
    return "\n".join(res)


enum_register.register_class(LabelEnum)
