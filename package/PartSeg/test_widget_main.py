import sys

import numpy as np
from qtpy.QtWidgets import QApplication, QWidget

import PartSegData
from PartSeg.common_gui.channel_control import GammaInfoWidget

color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.widget = GammaInfoWidget(100, 10)


def main():
    app = QApplication(sys.argv)
    widget = GammaInfoWidget(300, 20)
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
