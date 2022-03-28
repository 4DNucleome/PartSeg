import os

from packaging.version import parse as parse_version
from qtpy import QT_VERSION
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QGridLayout, QLabel, QPushButton

import PartSeg
from PartSeg.common_gui.universal_gui_part import TextShow


class AboutDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("About PartSeg")
        text = (
            f"<strong>PartSeg</strong> ({PartSeg.__version__})<br>"
            "PartSeg is gui and library for segmentation algorithms on high resolution microscopy<br><br>Webpage: "
            "<a href='https://4dnucleome.cent.uw.edu.pl/PartSeg/'>https://4dnucleome.cent.uw.edu.pl/PartSeg/</a>"
            "<br>Repository and issue tracker: "
            "<a href='https://github.com/4DNucleome/PartSeg'>https://github.com/4DNucleome/PartSeg</a>"
        )
        cite_as_text = (
            "Bokota, G., Sroka, J., Basu, S. et al. PartSeg: a tool for quantitative feature"
            " extraction from 3D microscopy images for dummies. BMC Bioinformatics 22, 72 (2021)."
            " https://doi.org/10.1186/s12859-021-03984-1"
        )
        dev_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(PartSeg.__file__))), "changelog.md")
        if os.path.exists(dev_path):
            with open(dev_path, encoding="utf-8") as ff:
                changelog_text = ff.read()
        else:
            changelog_text = PartSeg.changelog
        text_label = QLabel(self)
        text_label.setText(text)
        self.change_log = TextShow()
        self.change_log.setAcceptRichText(True)
        if parse_version(QT_VERSION) < parse_version("5.14.0"):  # pragma: no cover
            self.change_log.setText(changelog_text)
        else:
            self.change_log.setMarkdown(changelog_text)
        self.cite_as = TextShow(lines=3)
        if parse_version(QT_VERSION) < parse_version("5.14.0"):  # pragma: no cover
            self.cite_as.setText(cite_as_text)
        else:
            self.cite_as.setMarkdown(cite_as_text)
        ok_but = QPushButton("Ok")
        ok_but.clicked.connect(self.accept)
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout = QGridLayout()
        layout.addWidget(text_label, 0, 0, 1, 3)
        layout.addWidget(self.change_log, 1, 0, 1, 3)
        layout.addWidget(QLabel("Cite as:"), 2, 0, 1, 3)
        layout.addWidget(self.cite_as, 3, 0, 1, 3)
        layout.addWidget(ok_but, 4, 2)
        self.setLayout(layout)
