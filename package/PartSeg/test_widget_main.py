import sys
import os

import PartSegData
import numpy as np
from napari._qt.qt_viewer import QtViewer
from napari.utils.theme import template as napari_template
from napari.resources import resources_dir as napari_resources_dir
from napari.components import ViewerModel as Viewer
from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider
from qtpy.QtCore import Qt

from PartSeg.common_backend.base_settings import ViewSettings
from PartSegCore.color_image import default_colormap_dict, color_image_fun
from PartSegImage import TiffImageReader

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        colormaps_list = [default_colormap_dict.get(x, None) for x in ["BlackRed", "BlackGreen", "BlackBlue", None]]
        self.settings = ViewSettings()
        self.viewer = Viewer()
        self.qt_viewer = QtViewer(self.viewer)
        image = TiffImageReader.read_image("/home/czaki/Projekty/partseg/test_data/test_nucleus.tif")
        colored_image = color_image_fun(
            image.get_layer(0, 5), colors=colormaps_list[: image.channels], min_max=list(image.get_ranges())
        )
        layout = QVBoxLayout()
        self.viewer.add_image(colored_image, rgb=True)
        layout.addWidget(self.qt_viewer)
        self.setLayout(layout)
        self.btn = QPushButton("Aaaa")
        layout.addWidget(self.btn)
        self.bar = QSlider(Qt.Horizontal)
        self.bar.setRange(0, 100)
        layout.addWidget(self.bar)
        with open(os.path.join(napari_resources_dir, "stylesheet.qss"), "r") as f:
            raw_stylesheet = f.read()
        self.setStyleSheet(napari_template(raw_stylesheet, **self.viewer.palette))
        print(self.viewer.palette)


def main():
    app = QApplication(sys.argv)
    widget = TestWidget()
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
