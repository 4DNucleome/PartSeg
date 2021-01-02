from typing import List

import numpy as np
from napari.layers import Layer
from vispy.app import MouseEvent

from ..common_gui.channel_control import ChannelProperty
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
            bound_info = image_info.roi_info.bound_info.get(num, None)
            if bound_info is None:
                continue
            # TODO think about marking on bright background
            slices = bound_info.get_slices()
            slices[image_info.image.stack_pos] = slice(None)
            component_mark = image_info.roi_info.roi[tuple(slices)] == num
            translate_grid = image_info.roi.translate_grid + (bound_info.lower) * image_info.roi.scale
            translate_grid[image_info.image.stack_pos] = 0
            self.additional_layers.append(
                self.viewer.add_image(
                    component_mark,
                    scale=image_info.roi.scale,
                    blending="additive",
                    colormap="gray",
                    opacity=0.5,
                )
            )
            self.additional_layers[-1].translate_grid = translate_grid

    def component_click(self, _event: MouseEvent):
        if self.viewer.active_layer is None:
            return
        cords = np.array([int(x) for x in self.viewer.active_layer.coordinates])
        for image_info in self.image_info.values():
            if image_info.roi_info.roi is None:
                continue
            if not image_info.coords_in(cords):
                continue
            moved_coords = image_info.translated_coords(cords)
            component = image_info.roi_info.roi[tuple(moved_coords)]
            if component:
                self.component_clicked.emit(component)

    def get_tool_tip_text(self) -> str:
        text = super().get_tool_tip_text()
        text_list = []
        for el in self.components:
            if self.settings.component_is_chosen(el):
                text_list.append(f"☑{el}")
            else:
                text_list.append(f"☐{el}")
        if text:
            return " ".join(text_list) + "\n" + text
        return " ".join(text_list)
