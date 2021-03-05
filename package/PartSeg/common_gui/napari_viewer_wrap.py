from typing import List, Optional

from napari import Viewer as NViewer
from napari.utils.colormaps import Colormap
from qtpy.QtWidgets import QCheckBox, QFormLayout, QPushButton, QWidget

from PartSeg.common_backend.base_settings import BaseSettings


class SynchronizeWidget(QWidget):
    def __init__(self, settings: BaseSettings, viewer: NViewer, partseg_viewer_name, parent=None):
        super().__init__(parent=parent)
        self.settings = settings
        self.viewer = viewer
        self.layer_name_dict = {}
        self.partseg_viewer_name = partseg_viewer_name
        self.sync_image_chk = QCheckBox()
        self.sync_image_btn = QPushButton("Sync image")
        self.sync_image_btn.clicked.connect(self.sync_image)
        self.sync_image_chk.stateChanged.connect(self._sync_image)
        self.sync_ROI_chk = QCheckBox()
        self.sync_ROI_btn = QPushButton("Sync ROI")
        self.sync_ROI_btn.clicked.connect(self.sync_roi)
        self.sync_ROI_chk.stateChanged.connect(self._sync_roi)
        self.sync_additional_chk = QCheckBox()
        self.sync_additional_btn = QPushButton("Sync additional")
        self.sync_additional_btn.clicked.connect(self.sync_additional)
        self.sync_additional_chk.stateChanged.connect(self._sync_additional)
        self.sync_points_chk = QCheckBox()
        self.sync_points_btn = QPushButton("Sync points")
        self.sync_points_btn.clicked.connect(self.sync_points)
        self.sync_points_chk.stateChanged.connect(self._sync_points)
        layout = QFormLayout()
        layout.addRow(self.sync_image_chk, self.sync_image_btn)
        layout.addRow(self.sync_ROI_chk, self.sync_ROI_btn)
        layout.addRow(self.sync_additional_chk, self.sync_additional_btn)
        layout.addRow(self.sync_points_chk, self.sync_points_btn)
        self.setLayout(layout)

        self.sync_image_chk.stateChanged.connect(self.sync_image_btn.setDisabled)
        self.sync_ROI_chk.stateChanged.connect(self.sync_ROI_btn.setDisabled)
        self.sync_additional_chk.stateChanged.connect(self.sync_additional_btn.setDisabled)
        self.settings.image_changed.connect(self._sync_image)
        self.settings.roi_changed.connect(self._sync_roi)
        self.settings.additional_layers_changed.connect(self._sync_additional)
        self.settings.points_changed.connect(self._sync_points)

    def get_colormaps(self) -> List[Optional[Colormap]]:
        channel_num = self.settings.image.channels
        if not self.partseg_viewer_name:
            return [None for _ in range(channel_num)]
        colormaps_name = [self.settings.get_channel_info(self.partseg_viewer_name, i) for i in range(channel_num)]
        return [self.settings.colormap_dict[name][0] for name in colormaps_name]

    def _clean_layers(self, layers_list):
        for name in layers_list:
            try:
                del self.viewer.layers[name]
            except (KeyError, ValueError):  # pragma: no cover pylint: disable=W0703
                pass

    def _substitute_image_layer(self, name, data, scale, cmap, name_list):
        if name in name_list and name in self.viewer.layers:
            try:
                layer = self.viewer.layers[name]
                layer.data = data
                if scale is not None:
                    layer.scale = scale
                name_list.remove(name)
                return layer
            except Exception:  # pragma: no cover  pylint: disable=W0703
                del self.viewer.layers[name]
        layer = self.viewer.add_image(
            data,
            name=name,
            scale=scale,
            blending="additive",
            colormap=cmap,
        )
        try:
            name_list.remove(layer.name)
        except KeyError:  # pragma: no cover
            pass
        return layer

    def _substitute_labels_layer(self, name, data, scale, name_list):
        if name in name_list and name in self.viewer.layers:
            try:
                layer = self.viewer.layers[name]
                layer.data = data
                if scale is not None:
                    layer.scale = scale
                name_list.remove(name)
                return layer
            except Exception:  # pragma: no cover pylint: disable=W0703
                del self.viewer.layers[name]
        layer = self.viewer.add_labels(
            data,
            name=name,
            scale=scale,
        )
        try:
            name_list.remove(layer.name)
        except KeyError:  # pragma: no cover
            pass
        return layer

    def _substitute_points_layer(self, name, data, scale, name_list):
        if name in name_list and name in self.viewer.layers:
            try:
                layer = self.viewer.layers[name]
                layer.data = data
                layer.scale = scale
                name_list.remove(name)
                return layer
            except Exception:  # pragma: no cover pylint: disable=W0703
                del self.viewer.layers[name]
        layer = self.viewer.add_points(
            data,
            name=name,
            scale=scale,
        )
        try:
            name_list.remove(layer.name)
        except KeyError:  # pragma: no cover
            pass
        return layer

    def _sync_image(self):
        if self.sync_image_chk.isChecked():
            self.sync_image()

    def sync_image(self):
        """
        Sync image and mask from PartSeg
        """
        image_layers = set(self.layer_name_dict.get("image", []))
        image = self.settings.image
        colormap_list = self.get_colormaps()
        current_layers = []
        scale = image.normalized_scaling()
        for i in range(image.channels):

            channel_name = image.channel_names[i]
            layer = self._substitute_image_layer(
                channel_name, image.get_channel(i), scale, colormap_list[i], image_layers
            )
            current_layers.append(layer.name)
        if self.settings.mask is not None:
            layer = self._substitute_labels_layer("Mask", self.settings.mask, scale, image_layers)
            current_layers.append(layer.name)
        self._clean_layers(image_layers)
        self.layer_name_dict["image"] = current_layers

    def _sync_roi(self):
        if self.sync_ROI_chk.isChecked():
            self.sync_roi()

    def sync_roi(self):
        """
        Sync ROI and alternative representation form PartSeg
        """
        roi_layers = set(self.layer_name_dict.get("roi", []))
        roi_info = self.settings.roi_info
        scale = self.settings.image.normalized_scaling()
        if roi_info.roi is None:
            self._clean_layers(roi_layers)
            self.layer_name_dict["roi"] = []
            return
        self._substitute_labels_layer("ROI", roi_info.roi, scale, roi_layers)
        current_layers = ["ROI"]
        for name, data in roi_info.alternative.items():
            layer = self._substitute_labels_layer(name, data, scale, roi_layers)
            current_layers.append(layer.name)
        self._clean_layers(roi_layers)
        self.layer_name_dict["roi"] = current_layers

    def _sync_additional(self):
        if self.sync_additional_chk.isChecked():
            self.sync_additional()

    def sync_additional(self):
        """
        Sync additional layers from PartSeg
        :return:
        """
        additional_layers = set(self.layer_name_dict.get("additional", []))
        scale = self.settings.image.normalized_scaling()
        current_layers = []
        for value in self.settings.additional_layers.values():
            try:
                data = self.settings.image.fit_array_to_image(value.data)
                local_scale = scale
            except ValueError:
                data = value.data
                local_scale = None
            if value.layer_type == "labels":
                layer = self._substitute_labels_layer(value.name, data, local_scale, additional_layers)
            else:
                layer = self._substitute_image_layer(value.name, data, local_scale, None, additional_layers)
            current_layers.append(layer.name)
        self._clean_layers(additional_layers)
        self.layer_name_dict["additional"] = current_layers

    def _sync_points(self):
        if self.sync_points_chk.isChecked():
            self.sync_points()

    def sync_points(self):
        points_layers = set(self.layer_name_dict.get("points", []))

        if self.settings.points is not None:
            scale = self.settings.image.normalized_scaling()
            layer = self._substitute_points_layer("Points", self.settings.points, scale, points_layers)
            self.layer_name_dict["points"] = [layer.name]

        self._clean_layers(points_layers)


class Viewer(NViewer):
    _settings: BaseSettings
    _sync_widget: SynchronizeWidget

    _napari_app_id = False

    def __init__(self, settings: BaseSettings, partseg_viewer_name: str, **kwargs):
        super().__init__(**kwargs)
        self._settings = settings
        self._sync_widget = SynchronizeWidget(settings, self, partseg_viewer_name)
        self.window.add_dock_widget(self._sync_widget, area="left")

    def create_initial_layers(
        self, image: bool = True, roi: bool = True, additional_layers: bool = False, points: bool = True
    ):
        """
        Synchronize give set of layers

        :param bool image: synchronize image and mask layers
        :param bool roi: synchronize roi with alternative representation
        :param bool additional_layers: synchronize alternative representation
        :param bool points: synchronize points
        """
        if image:
            self._sync_widget.sync_image()
        if roi:
            self._sync_widget.sync_roi()
        if additional_layers:
            self._sync_widget.sync_additional()
        if points:
            self._sync_widget.sync_points()
