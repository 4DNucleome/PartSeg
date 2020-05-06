import sys
from glob import glob

import PartSegData
import numpy as np
from napari.utils.theme import template as napari_template
from napari.resources import get_stylesheet
from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel
from qtpy.QtCore import Qt

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.napari_image_view import ImageView
from PartSegImage import TiffImageReader

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = BaseSettings("")
        self.prop = ChannelProperty(self.settings, "test")
        image = TiffImageReader.read_image("/home/czaki/Projekty/partseg/test_data/test_lsm.lsm")
        self.image_view = ImageView(self.settings, self.prop, "test", ndisplay=2)
        self.image_view.set_image(image)
        layout = QVBoxLayout()
        layout.addWidget(self.image_view)
        self.setLayout(layout)
        self.btn = QPushButton("Aaaa")
        self.btn.clicked.connect(self.load_image)
        self.btn2 = QPushButton("Aaaa2")
        self.btn2.clicked.connect(self.load_image2)
        self.label = QLabel()
        layout.addWidget(self.btn)
        layout.addWidget(self.btn2)
        layout.addWidget(self.label)
        self.bar = QSlider(Qt.Horizontal)
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)
        self.setStyleSheet(napari_template(get_stylesheet(), **self.image_view.viewer.palette))
        self.image_view.text_info_change.connect(self.label.setText)
        self.file_list = glob("/home/czaki/Projekty/partseg/test_data/stack1_components/*[0-9].tif")
        self.index = 0

    def load_image(self):
        image = TiffImageReader.read_image(
            "/home/czaki/Projekty/partseg/test_data/stack1_components/stack1_component1.tif",
            "/home/czaki/Projekty/partseg/test_data/stack1_components/stack1_component1_mask.tif",
        )
        self.image_view.set_image(image)

    def load_image2(self):
        image = TiffImageReader.read_image(self.file_list[self.index], self.file_list[self.index][:-4] + "_mask.tif")
        self.index += 1
        self.image_view.add_image(image)
        self.image_view.grid_view()


def main():
    app = QApplication(sys.argv)
    widget = TestWidget()
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
