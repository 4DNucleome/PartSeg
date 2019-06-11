from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox
import sys
import PartSegData
import numpy as np

from PartSeg.common_gui.channel_control import ColorComboBoxGroup, ColorComboBox
from PartSeg.project_utils_qt.settings import ViewSettings

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = ViewSettings()
        self.group = ColorComboBoxGroup(self.settings, "test", 30)
        self.group.set_channels(3)
        self.ddd = ColorComboBox(0, self.settings.available_colormaps, base_height=30)
        layout = QVBoxLayout()
        layout.addWidget(self.group)
        layout.addWidget(self.ddd)

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
