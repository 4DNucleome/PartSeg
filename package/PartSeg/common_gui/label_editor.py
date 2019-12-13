from qtpy.QtWidgets import QWidget, QColorDialog, QVBoxLayout
from qtpy.QtCore import Qt


class LabelEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.color_picker = QColorDialog()
        self.color_picker.setWindowFlag(Qt.Widget)
        self.color_picker.setOptions(QColorDialog.DontUseNativeDialog | QColorDialog.NoButtons)

        layout = QVBoxLayout()
        layout.addWidget(self.color_picker)
        self.setLayout(layout)