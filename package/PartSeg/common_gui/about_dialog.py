from qtpy.QtWidgets import QDialog, QLabel, QPushButton, QGridLayout
from qtpy.QtCore import Qt
from .. import __version__


class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("About PartSeg")
        text = (
            f"<strong>PartSeg</strong> ({__version__})<br>"
            "PartSeg is gui and library for segmentation algorithms on high resolution microscopy<br><br>Webpage: "
            "<a href='https://4dnucleome.cent.uw.edu.pl/PartSeg/'>https://4dnucleome.cent.uw.edu.pl/PartSeg/</a>"
            "<br>Repository and issue tracker"
            "<a href='https://github.com/4DNucleome/PartSeg'>https://github.com/4DNucleome/PartSeg</a>"
        )
        text_label = QLabel(text)
        ok_but = QPushButton("Ok")
        ok_but.clicked.connect(self.accept)
        # text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout = QGridLayout()
        layout.addWidget(text_label, 0, 0, 1, 3)
        layout.addWidget(ok_but, 1, 2)
        self.setLayout(layout)
