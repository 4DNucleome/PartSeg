from typing import Optional

from napari._qt.qt_viewer import QtViewer
from napari.components import ViewerModel as Viewer
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from qtpy.QtCore import Qt, Signal
from vispy.color import Colormap, ColorArray

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.channel_control import ChannelProperty, ColorComboBoxGroup
from PartSeg.common_gui.stack_image_view import ImageShowState
from PartSegCore.color_image import create_color_map
from PartSegImage import Image


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
        flags: int = Qt.Widget,
    ):
        super(ImageView, self).__init__(parent=parent, flags=flags)

        self.settings = settings
        self.channel_property = channel_property
        self.name = name
        self.image_layers = []

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

    def set_image(self, image: Optional[Image] = None):
        if image is None:
            image = self.settings.image

        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.channel_control.set_channels(image.channels)
        colors = [create_color_map(x) / 255.0 for x in self.channel_control.selected_colormaps]
        colors_array = [Colormap(ColorArray(x)) for x in colors]
        visibility = self.channel_control.channel_visibility

        for i in range(image.channels):
            self.image_layers.append(
                self.viewer.add_image(image.get_channel(i), colormap=colors_array[i], visible=visibility[i])
            )

        self.viewer.dims.set_point(image.time_pos, image.times // 2)
        self.viewer.dims.set_point(image.stack_pos, image.layers // 2)
        print(self.viewer.dims)

    def change_visibility(self, name: str, index: int):
        print("Aaa", name, index)
