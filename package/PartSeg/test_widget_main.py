import sys

from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.import_image import DragAndDropFileList, FileList, ImportDialog
from PartSeg.segmentation_analysis.main_window import CONFIG_FOLDER
from PartSegCore.analysis.load_functions import load_dict


class TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = BaseSettings(CONFIG_FOLDER)
        self.colormap_selector = DragAndDropFileList()
        self.colormap_selector.addItems(["aaa" * 20, "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"])
        self.colormap_selector2 = FileList(load_dict, self.settings)

        layout = QVBoxLayout()
        layout.addWidget(self.colormap_selector)
        layout.addWidget(self.colormap_selector2)
        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    settings = BaseSettings(CONFIG_FOLDER)
    widget = ImportDialog(load_dict, settings)
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
