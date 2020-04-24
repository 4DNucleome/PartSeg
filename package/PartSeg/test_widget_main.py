import sys

import numpy as np
from qtpy.QtWidgets import QApplication, QWidget

from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.common_gui.label_create import LabelEditor, LabelShow, ColorShow

from PartSeg.common_gui.import_image import DragAndDropFileList, FileList


color_maps = np.load(PartSegData.colors_file)


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.colormap_selector = DragAndDropFileList()
        self.colormap_selector.addItems(["aaa" * 20, "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"])
        self.colormap_selector2 = FileList()
        self.colormap_selector2.add_files(["aaa" * 20, "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"])
        layout = QVBoxLayout()
        layout.addWidget(self.colormap_selector)
        layout.addWidget(self.colormap_selector2)
        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    widget = GammaInfoWidget(300, 20)
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
