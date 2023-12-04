from typing import Dict, Iterable, Optional, Tuple

from napari import Viewer
from napari.layers import Image
from napari.utils import Colormap
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QTabWidget

from PartSeg.common_gui.colormap_creator import ChannelPreview, ColormapCreator, ColormapList, save_colormap_in_settings
from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSegCore.custom_name_generate import custom_name_generate


class NapariColormapControl(ChannelPreview):
    def __init__(
        self, viewer: Viewer, colormap: Colormap, accepted: bool, name: str, removable: bool = False, used: bool = False
    ):
        super().__init__(colormap, accepted, name, removable, used)
        self.viewer = viewer
        self.apply_colormap_btn = QPushButton("Apply")
        layout: QHBoxLayout = self.layout()
        layout.removeWidget(self.checked)
        layout.insertWidget(0, self.apply_colormap_btn)

        self.apply_colormap_btn.clicked.connect(self.apply_colormap)
        viewer.layers.selection.events.changed.connect(self.update_preview)
        self.update_preview()

    def update_preview(self, _event=None):
        if len(self.viewer.layers.selection) == 1 and isinstance(next(iter(self.viewer.layers.selection)), Image):
            self.apply_colormap_btn.setEnabled(True)
            self.apply_colormap_btn.setToolTip("Apply colormap to selected layer")
        else:
            self.apply_colormap_btn.setEnabled(False)
            self.apply_colormap_btn.setToolTip("Select one image layer to apply colormap")

    def apply_colormap(self):
        if len(self.viewer.layers.selection) == 1 and isinstance(
            layer := next(iter(self.viewer.layers.selection)), Image
        ):
            layer.colormap = self.colormap


class NapariColormapList(ColormapList):
    def __init__(
        self,
        viewer: Viewer,
        colormap_map: Dict[str, Tuple[Colormap, bool]],
        selected: Optional[Iterable[str]] = None,
        parent=None,
    ):
        super().__init__(colormap_map, selected, parent)
        self.viewer = viewer

    def _create_colormap_preview(self, colormap: Colormap, name: str, removable: bool) -> ChannelPreview:
        return NapariColormapControl(self.viewer, colormap, False, name, removable)


class ImageColormap(QTabWidget):
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        settings = get_settings()
        self.settings = settings
        self.colormap_list = NapariColormapList(viewer, settings.colormap_dict, parent=self)
        self.colormap_creator = ColormapCreator(self)
        self.addTab(self.colormap_list, "List")
        self.addTab(self.colormap_creator, "Creator")

        self.colormap_creator.colormap_selected.connect(self.handle_new_colormap)

        self.colormap_list.edit_signal.connect(self.colormap_creator.set_colormap)
        self.colormap_list.edit_signal.connect(self._set_colormap_editor)

    def handle_new_colormap(self, colormap):
        rand_name = custom_name_generate(set(), self.settings.colormap_dict)
        save_colormap_in_settings(self.settings, colormap, rand_name)
        self.settings.dump()

    def _set_colormap_editor(self):
        self.setCurrentWidget(self.colormap_creator)
