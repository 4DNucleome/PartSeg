from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QScrollArea
import sys
import PartSegData
import numpy as np

from PartSeg.common_gui.channel_control import ColorComboBoxGroup
from PartSeg.common_gui.colormap_creator import PColormapCreator, PColormapList, ChannelPreview
from PartSeg.common_gui.universal_gui_part import ProgressCircle
from PartSeg.common_backend.base_settings import ViewSettings, ColormapDict
from PartSeg.utils.color_image.base_colors import default_colormap_dict, starting_colors

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        settings = ViewSettings()
        self.colormap_selector = PColormapCreator(settings)
        self.color_preview = PColormapList(settings, starting_colors)
        self.color_preview.edit_signal.connect(self.colormap_selector.set_colormap)
        self.test = ColorComboBoxGroup(settings, "aa")
        self.test.set_channels(4)

        layout = QVBoxLayout()
        layout.addWidget(self.test)
        layout.addWidget(self.colormap_selector)
        layout.addWidget(self.color_preview)
        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    widget = TestWidget()
    widget.show()
    app.exec_()


if __name__ == '__main__':
    main()
