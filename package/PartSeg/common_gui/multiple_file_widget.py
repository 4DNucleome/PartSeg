from qtpy.QtWidgets import QWidget, QPushButton, QTreeWidget, QGridLayout
from typing import Callable, Any

from PartSeg.utils.io_utils import LoadBase


class MultipleFileWidget(QWidget):
    def __init__(self, get_state: Callable[[], Any], set_state: Callable[[Any], Any], load_dict: LoadBase):
        super().__init__()
        self.get_state = get_state
        self.set_state = set_state
        self.file_view = QTreeWidget()
        self.load_files = QPushButton("Load Files")

        layout = QGridLayout()
        layout.addWidget(self.file_view, 0, 0, 1, 2)
        layout.addWidget(self.load_files, 1, 0)

        self.setLayout(layout)
