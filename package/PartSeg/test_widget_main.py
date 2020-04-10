import sys

import PartSegData
import numpy as np
from napari.utils.theme import template as napari_template
from napari.resources import get_stylesheet
from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider
from qtpy.QtCore import Qt

from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.napari_image_view import ImageView
from PartSegImage import TiffImageReader

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = ViewSettings()
        self.prop = ChannelProperty(self.settings, "test")
        image = TiffImageReader.read_image("/home/czaki/Projekty/partseg/test_data/test_lsm.lsm")
        self.image_view = ImageView(self.settings, self.prop, "test")
        self.image_view.set_image(image)
        layout = QVBoxLayout()
        layout.addWidget(self.image_view)
        self.setLayout(layout)
        self.btn = QPushButton("Aaaa")
        self.btn.clicked.connect(self.load_image)
        layout.addWidget(self.btn)
        self.bar = QSlider(Qt.Horizontal)
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)
        self.setStyleSheet(napari_template(get_stylesheet(), **self.image_view.viewer.palette))

    def load_image(self):
        image = TiffImageReader.read_image(
            "/home/czaki/Projekty/partseg/test_data/stack1_components/stack1_component1.tif"
        )
        self.image_view.set_image(image)


def main():
    app = QApplication(sys.argv)
    widget = TestWidget()
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
