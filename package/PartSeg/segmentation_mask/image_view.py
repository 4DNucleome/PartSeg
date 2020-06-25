from typing import List

import numpy as np
from napari.layers import Layer
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QToolTip
from vispy.app import MouseEvent

from PartSeg.common_gui.channel_control import ChannelProperty

from ..common_gui.napari_image_view import ImageView


class StackImageView(ImageView):
    """
    :cvar settings: StackSettings
    """

    def __init__(self, settings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        self.viewer_widget.canvas.events.mouse_press.connect(self.component_click)
        self.additional_layers: List[Layer] = []
        # self.image_area.pixmap.click_signal.connect(self.component_click)

    def component_unmark(self, _num):
        self.viewer.layers.unselect_all()
        for el in self.additional_layers:
            el.selected = True
        self.viewer.layers.remove_selected()
        self.additional_layers = []

    def component_mark(self, num):
        self.component_unmark(num)

        for image_info in self.image_info.values():
            bound_info = image_info.segmentation_info.bound_info.get(num, None)
            if bound_info is None:
                continue
            # TODO think about marking on bright background
            slices = bound_info.get_slices()
            slices[image_info.image.stack_pos] = slice(None)
            component_mark = image_info.segmentation_info.segmentation[tuple(slices)] == num
            translate_grid = image_info.segmentation.translate_grid + (bound_info.lower) * image_info.segmentation.scale
            translate_grid[image_info.image.stack_pos] = 0
            self.additional_layers.append(
                self.viewer.add_image(
                    component_mark,
                    scale=image_info.segmentation.scale,
                    blending="additive",
                    colormap="gray",
                    opacity=0.5,
                )
            )
            self.additional_layers[-1].translate_grid = translate_grid

    def component_click(self, event: MouseEvent):
        cords = np.array([int(x) for x in self.viewer.active_layer.coordinates])
        for image_info in self.image_info.values():
            if image_info.segmentation_info.segmentation is None:
                continue
            if not image_info.coords_in(cords):
                continue
            moved_coords = image_info.translated_coords(cords)
            component = image_info.segmentation_info.segmentation[tuple(moved_coords)]
            if component:
                self.component_clicked.emit(component)

    def event(self, event: QEvent):
        if event.type() == QEvent.ToolTip and self.components:
            # text = str(self.component)
            text_list = []
            for el in self.components:
                if self.settings.component_is_chosen(el):
                    text_list.append("☑{}".format(el))
                else:
                    text_list.append("☐{}".format(el))
            QToolTip.showText(event.globalPos(), " ".join(text_list))
        return super().event(event)
