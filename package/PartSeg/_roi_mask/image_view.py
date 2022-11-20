from vispy.app import MouseEvent

from PartSeg._roi_mask.stack_settings import StackSettings
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.napari_image_view import ImageInfo, ImageView, LabelEnum


class StackImageView(ImageView):
    """
    :cvar settings: StackSettings
    """

    settings: StackSettings

    def __init__(self, settings: StackSettings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        self.viewer_widget.canvas.events.mouse_press.connect(self.component_click)

    def refresh_selected(self):
        if (
            self.settings.get_from_profile(f"{self.name}.image_state.show_label", LabelEnum.Show_results)
            == LabelEnum.Show_selected
        ):
            self.update_roi_labeling()

    def component_click(self, _event: MouseEvent):
        cords = self._coordinates()
        if cords is None:
            return
        for image_info in self.image_info.values():
            if image_info.roi_info.roi is None:
                continue
            if not image_info.coords_in(cords):
                continue
            moved_coords = image_info.translated_coords(cords)
            if component := image_info.roi_info.roi[tuple(moved_coords)]:
                self.component_clicked.emit(component)

    def get_roi_view_parameters(self, image_info: ImageInfo):
        res = super().get_roi_view_parameters(image_info=image_info)

        if (
            self.settings.get_from_profile(f"{self.name}.image_state.show_label", LabelEnum.Show_results)
            != LabelEnum.Show_selected
        ):
            return res

        selected_components = set(self.settings.chosen_components())
        for num in list(res):
            if num not in selected_components:
                res[num] = [0, 0, 0, 0]
        return res

    def get_tool_tip_text(self) -> str:
        text = super().get_tool_tip_text()
        text_list = []
        for el in self.components:
            if self.settings.component_is_chosen(el):
                text_list.append(f"☑{el}")
            else:
                text_list.append(f"☐{el}")
        return " ".join(text_list) + "\n" + text if text else " ".join(text_list)
