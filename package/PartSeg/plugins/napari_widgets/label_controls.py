from napari import Viewer
from qtpy.QtWidgets import QTabWidget

from PartSeg.common_gui.colormap_creator import ColormapCreator, ColormapList, save_colormap_in_settings
from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSegCore.custom_name_generate import custom_name_generate


class ImageColormap(QTabWidget):
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        settings = get_settings()
        self.settings = settings
        self.colormap_list = ColormapList(settings.colormap_dict, parent=self)
        self.colormap_creator = ColormapCreator(self)
        self.addTab(self.colormap_list, "List")
        self.addTab(self.colormap_creator, "Creator")

        self.colormap_creator.colormap_selected.connect(self.handle_new_colormap)

    def handle_new_colormap(self, colormap):
        rand_name = custom_name_generate(set(), self.settings.colormap_dict)
        save_colormap_in_settings(self.settings, colormap, rand_name)
        self.settings.dump()
