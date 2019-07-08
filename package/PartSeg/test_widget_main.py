from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout
import sys
import PartSegData
import numpy as np

from PartSeg.common_gui.channel_control import ColorComboBoxGroup, ChannelProperty
from PartSeg.common_gui.stack_image_view import ImageView
from PartSeg.project_utils_qt.settings import ViewSettings
from PartSeg.tiff_image import ImageReader

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = ViewSettings()
        self.group = ColorComboBoxGroup(self.settings, "test", height=30)
        self.group.set_channels(3)
        self.channel_property = ChannelProperty(self.settings, "test2")
        self.image_view = ImageView(self.settings, self.channel_property, "test2")

        image = ImageReader.read_image(PartSegData.segmentation_analysis_default_image)
        self.settings.image = image

        layout = QVBoxLayout()
        layout.addWidget(self.group)
        layout.addWidget(self.image_view)
        layout.addWidget(self.channel_property)

        self.setLayout(layout)

    def show_info(self):
        print("aaaa", self.color_box1.is_checked())

    def show_info2(self):
        print("bbbb", self.color_box1.is_checked())


def main():
    app = QApplication(sys.argv)
    widget = TestWidget()
    widget.show()
    app.exec()


if __name__ == '__main__':
    main()
