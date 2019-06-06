from qtpy.QtWidgets import QApplication, QWidget, QHBoxLayout, QCheckBox
import sys
import PartSegData
import numpy as np

from PartSeg.common_gui.channel_control import ColorComboBox

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.color_box1 = ColorComboBox(color_maps, color="gray", base_height=40)
        self.color_box2 = ColorComboBox(color_maps, base_height=30)
        self.gauss = QCheckBox("Gauss")
        self.gauss.stateChanged.connect(self.color_box1.set_blur)
        self.gauss.stateChanged.connect(self.color_box2.set_blur)
        self.fixed_range = QCheckBox("fixed range")
        self.fixed_range.stateChanged.connect(self.color_box1.set_lock)
        self.fixed_range.stateChanged.connect(self.color_box2.set_lock)
        layout = QHBoxLayout()
        layout.addWidget(self.color_box1)
        layout.addWidget(self.color_box2)
        layout.addWidget(self.gauss)
        layout.addWidget(self.fixed_range)
        self.setLayout(layout)
        self.color_box1.state_changed.connect(self.show_info)
        self.color_box1.clicked.connect(self.show_info2)

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
