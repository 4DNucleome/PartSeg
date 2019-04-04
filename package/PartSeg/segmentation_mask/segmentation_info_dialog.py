from qtpy.QtWidgets import QWidget, QGridLayout, QListWidget, QPlainTextEdit, QLabel, QPushButton
from qtpy.QtCore import QEvent

from PartSeg.utils.mask.algorithm_description import mask_algorithm_dict
from .stack_settings import StackSettings
from typing import Callable


class SegmentationInfoDialog(QWidget):
    def __init__(self, settings: StackSettings, set_parameters: Callable[[str, dict], None]):
        super().__init__()
        self.settings = settings
        self.set_parameters = set_parameters
        self.components = QListWidget()
        self.components.currentItemChanged.connect(self.change_component_info)
        self.description = QPlainTextEdit()
        self.description.setReadOnly(True)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.set_parameters_btn = QPushButton("Reuse parameters")
        self.set_parameters_btn.clicked.connect(self.set_parameter_action)
        layout = QGridLayout()
        layout.addWidget(QLabel("Components:"), 0, 0)
        layout.addWidget(QLabel("segmentation parameters:"), 0, 1)
        layout.addWidget(self.components, 1, 0)
        layout.addWidget(self.description, 1, 1)
        layout.addWidget(self.close_btn, 2, 0)
        layout.addWidget(self.set_parameters_btn, 2, 1)
        self.setLayout(layout)
        self.setWindowTitle("Parameters preview")

    def change_component_info(self):
        if self.components.currentItem() is None:
            return
        text = self.components.currentItem().text()
        parameters = self.settings.components_parameters_dict[int(text)]
        if parameters is None:
            self.description.setPlainText("None")
        else:
            self.description.setPlainText(f"Component {text}\n" + parameters.pretty_print(mask_algorithm_dict))

    def set_parameter_action(self):
        if self.components.currentItem() is None:
            return
        text = self.components.currentItem().text()
        parameters = self.settings.components_parameters_dict[int(text)]
        self.set_parameters(parameters.algorithm, parameters.values)



    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            index = self.components.currentRow()
            self.components.clear()
            self.components.addItems(list(map(str, self.settings.components_parameters_dict.keys())))
            self.components.setCurrentRow(index)
        return super().event(event)
