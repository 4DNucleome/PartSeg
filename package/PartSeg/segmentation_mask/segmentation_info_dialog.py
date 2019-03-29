from qtpy.QtWidgets import QWidget, QGridLayout, QListWidget, QPlainTextEdit, QLabel, QPushButton
from qtpy.QtCore import QEvent
from .stack_settings import StackSettings


class SegmentationInfoDialog(QWidget):
    def __init__(self, settings: StackSettings):
        super().__init__()
        self.settings = settings
        self.components = QListWidget()
        self.components.currentItemChanged.connect(self.change_component_info)
        self.description = QPlainTextEdit()
        self.description.setReadOnly(True)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.set_parameters_btn = QPushButton("Set parameters")
        layout = QGridLayout()
        layout.addWidget(QLabel("Compenents:"), 0, 0)
        layout.addWidget(QLabel("segmentation parameters:"), 0, 1)
        layout.addWidget(self.components, 1, 0)
        layout.addWidget(self.description, 1, 1)
        layout.addWidget(self.close_btn, 2, 0)
        layout.addWidget(self.set_parameters_btn, 2, 1)
        self.setLayout(layout)

    def change_component_info(self):
        if self.components.currentItem() is None:
            return
        text = self.components.currentItem().text()
        parameters = self.settings.components_parameters_dict[int(text)]
        self.description.setPlainText(f"Component {text}\n" + str(parameters))

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            index = self.components.currentRow()
            self.components.clear()
            self.components.addItems(list(map(str, self.settings.components_parameters_dict.keys())))
            self.components.setCurrentRow(index)
        return super().event(event)
