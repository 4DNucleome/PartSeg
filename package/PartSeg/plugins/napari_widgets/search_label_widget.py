import warnings

import numpy as np
from magicgui.widgets import Container, HBox, PushButton, SpinBox, create_widget
from napari import Viewer
from napari.layers import Labels
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer
from vispy.geometry import Rect

from PartSeg.common_gui.napari_image_view import SearchType, get_highlight_colormap
from PartSegCore.roi_info import ROIInfo

HIGHLIGHT_LABEL_NAME = ".Highlight"


class SearchLabel(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(layout="vertical")
        self.napari_viewer = napari_viewer
        self.search_type = create_widget(annotation=SearchType, label="Search type")
        self.search_type.changed.connect(self._component_num_changed)
        self.component_selector = SpinBox(name="Label number", min=0, max=2**31 - 1)
        self.component_selector.changed.connect(self._component_num_changed)
        self.labels_layer = create_widget(annotation=Labels, label="ROI", options={})
        self.labels_layer.changed.connect(self._update_roi_info)
        self.zoom_factor = create_widget(annotation=float, label="Zoom factor", value=1.2)
        self.zoom_factor.changed.connect(self._component_num_changed)
        self.stop = PushButton(name="Stop")
        self.stop.clicked.connect(self._stop)
        self.roi_info = None

        layout = HBox(
            widgets=(
                self.search_type,
                self.component_selector,
            )
        )
        layout2 = HBox(
            widgets=(
                self.labels_layer,
                self.zoom_factor,
            )
        )
        self.insert(0, layout)
        self.insert(1, layout2)
        self.insert(2, self.stop)

        self._update_roi_info()

    def _update_roi_info(self):
        if self.labels_layer.value is None:
            return
        self.roi_info = ROIInfo(self.labels_layer.value.data)
        self._component_num_changed()

    def _stop(self):
        if HIGHLIGHT_LABEL_NAME in self.napari_viewer.layers:
            self.napari_viewer.layers[HIGHLIGHT_LABEL_NAME].metadata["timer"].stop()
            del self.napari_viewer.layers[HIGHLIGHT_LABEL_NAME]

    def _highlight(self):
        num = self.component_selector.value
        labels = self.labels_layer.value
        bound_info = self.roi_info.bound_info.get(num, None)
        if bound_info is None:
            self._stop()
            return
        slices = bound_info.get_slices()
        component_mark = self.roi_info.roi[tuple(slices)] == num
        translate_grid = labels.translate + bound_info.lower * labels.scale
        if HIGHLIGHT_LABEL_NAME in self.napari_viewer.layers:
            self.napari_viewer.layers[HIGHLIGHT_LABEL_NAME].data = component_mark
        else:
            layer = self.napari_viewer.add_labels(
                component_mark,
                name=HIGHLIGHT_LABEL_NAME,
                scale=labels.scale,
                blending="translucent",
                opacity=0.7,
                **get_highlight_colormap(),
            )

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

        self.napari_viewer.layers[HIGHLIGHT_LABEL_NAME].translate = translate_grid
        self._shift_if_need(labels, bound_info)

    def _shift_if_need(self, labels, bound_info):
        if self.napari_viewer.dims.ndisplay != 2:
            return
        lower_bound = self._data_to_world(labels, bound_info.lower)
        upper_bound = self._data_to_world(labels, bound_info.upper)
        self._update_point(lower_bound, upper_bound)
        l_bound = lower_bound[-2:][::-1]
        u_bound = upper_bound[-2:][::-1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Public access to Window.qt_viewer")
            warnings.filterwarnings("ignore", "Access to QtViewer.view is deprecated")
            rect = Rect(self.napari_viewer.window.qt_viewer.view.camera.get_state()["rect"])
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Public access to Window.qt_viewer")
            warnings.filterwarnings("ignore", "Access to QtViewer.view is deprecated")
            self.napari_viewer.window.qt_viewer.view.camera.set_state({"rect": rect})

    @staticmethod
    def _data_to_world(layer: Labels, cords):
        return layer._transforms[1:3].simplified(cords)  # pylint: disable=protected-access

    def _zoom(self):
        self._stop()
        if self.napari_viewer.dims.ndisplay != 2:
            show_info("Zoom in does not work in 3D mode")
        num = self.component_selector.value
        labels = self.labels_layer.value
        bound_info = self.roi_info.bound_info.get(num, None)
        if bound_info is None:
            return
        lower_bound = self._data_to_world(labels, bound_info.lower)
        upper_bound = self._data_to_world(labels, bound_info.upper)
        diff = upper_bound - lower_bound
        frame = diff * (self.zoom_factor.value - 1)
        if self.napari_viewer.dims.ndisplay == 2:
            rect = Rect(pos=(lower_bound - frame)[-2:][::-1], size=(diff + 2 * frame)[-2:][::-1])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Public access to Window.qt_viewer")
                warnings.filterwarnings("ignore", "Access to QtViewer.view is deprecated")
                self.napari_viewer.window.qt_viewer.view.camera.set_state({"rect": rect})
        self._update_point(lower_bound, upper_bound)

    def _update_point(self, lower_bound, upper_bound):
        point = (lower_bound + upper_bound) / 2
        current_point = self.napari_viewer.dims.point[-len(lower_bound) :]
        dims = len(lower_bound) - self.napari_viewer.dims.ndisplay
        start_dims = self.napari_viewer.dims.ndim - len(lower_bound)
        for i in range(dims):
            if not (lower_bound[i] <= current_point[i] <= upper_bound[i]):
                self.napari_viewer.dims.set_point(start_dims + i, point[i])

    def _component_num_changed(self):
        if self.roi_info is None:
            return
        if self.search_type.value == SearchType.Highlight:
            self._highlight()
        else:
            self._zoom()
