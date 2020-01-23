from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout
import sys
import PartSegData
import numpy as np

from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.common_gui.label_create import LabelEditor, LabelShow, ColorShow

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        settings = ViewSettings()
        self.colormap_selector = LabelEditor(settings)
        self.show_ = LabelShow("a", settings.label_colors, False)
        self.show_.setMinimumHeight(50)
        self.show_.setMinimumWidth(50)
        self.color_show = ColorShow([255, 10, 203])
        layout = QVBoxLayout()
        layout.addWidget(self.colormap_selector)
        layout.addWidget(self.show_)
        layout.addWidget(self.color_show)
        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    widget = TestWidget()
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
