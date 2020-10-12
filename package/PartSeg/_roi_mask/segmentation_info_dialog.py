from typing import Callable, Dict, Optional

from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QGridLayout, QLabel, QListWidget, QPlainTextEdit, QPushButton, QWidget

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.mask.algorithm_description import mask_algorithm_dict

from .stack_settings import StackSettings


class SegmentationInfoDialog(QWidget):
    def __init__(self, settings: StackSettings, set_parameters: Callable[[str, dict], None], additional_text=None):
        """

        :param settings:
        :param set_parameters: Function which set parameters of chosen in dialog.
        :param additional_text: Additional text on top of Window.
        """
        super().__init__()
        self.settings = settings
        self.parameters_dict = None
        self.set_parameters = set_parameters
        self.components = QListWidget()
        self.components.currentItemChanged.connect(self.change_component_info)
        self.description = QPlainTextEdit()
        self.description.setReadOnly(True)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.set_parameters_btn = QPushButton("Reuse parameters")
        self.set_parameters_btn.clicked.connect(self.set_parameter_action)
        self.additional_text_label = QLabel(additional_text)
        layout = QGridLayout()
        layout.addWidget(self.additional_text_label, 0, 0, 1, 2)
        if not additional_text:
            self.additional_text_label.setVisible(False)

        layout.addWidget(QLabel("Components:"), 1, 0)
        layout.addWidget(QLabel("segmentation parameters:"), 1, 1)
        layout.addWidget(self.components, 2, 0)
        layout.addWidget(self.description, 2, 1)
        layout.addWidget(self.close_btn, 3, 0)
        layout.addWidget(self.set_parameters_btn, 3, 1)
        self.setLayout(layout)
        self.setWindowTitle("Parameters preview")

    def set_parameters_dict(self, val: Optional[Dict[int, ROIExtractionProfile]]):
        self.parameters_dict = val

    def set_additional_text(self, text):
        self.additional_text_label.setText(text)
        self.additional_text_label.setVisible(bool(text))

    @property
    def get_parameters(self):
        if self.parameters_dict:
            return self.parameters_dict
        return self.settings.components_parameters_dict

    def change_component_info(self):
        if self.components.currentItem() is None:
            return
        text = self.components.currentItem().text()
        parameters = self.get_parameters[int(text)]
        if parameters is None:
            self.description.setPlainText("None")
        else:
            self.description.setPlainText(f"Component {text}\n" + parameters.pretty_print(mask_algorithm_dict))

    def set_parameter_action(self):
        if self.components.currentItem() is None:
            return
        text = self.components.currentItem().text()
        parameters = self.get_parameters[int(text)]
        self.set_parameters(parameters.algorithm, parameters.values)

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            index = self.components.currentRow()
            self.components.clear()
            self.components.addItems(list(map(str, self.get_parameters.keys())))
            self.components.setCurrentRow(index)
        return super().event(event)
