import sys

import numpy as np
from qtpy.QtWidgets import QApplication

import PartSegData
from PartSeg.common_gui.colormap_creator import ColormapCreator

color_maps = np.load(PartSegData.colors_file)


def main():
    app = QApplication(sys.argv)
    widget = ColormapCreator()
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
