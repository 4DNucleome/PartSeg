from typing import Dict, NamedTuple

from qtpy.QtWidgets import QComboBox, QDialog, QGridLayout, QPushButton, QStackedWidget

from PartSegCore.image_transforming import TransformBase, image_transform_dict
from PartSegImage import Image

from .algorithms_description import FormWidget


class ImageAdjustTuple(NamedTuple):
    values: dict
    algorithm: TransformBase


class ImageAdjustmentDialog(QDialog):
    def __init__(self, image: Image, transform_dict: Dict[str, TransformBase] = None):
        super().__init__()
        if transform_dict is None:
            transform_dict = image_transform_dict
        self.choose = QComboBox()
        self.stacked = QStackedWidget()
        for key, val in transform_dict.items():
            self.choose.addItem(key)
            initial_values = val.calculate_initial(image)
            form_widget = FormWidget(val.get_fields_per_dimension(image.get_dimension_letters()), initial_values)
            self.stacked.addWidget(form_widget)

        self.choose.currentIndexChanged.connect(self.stacked.setCurrentIndex)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process)
        self.transform_dict = transform_dict
        self.result_val: ImageAdjustTuple = None

        layout = QGridLayout()
        layout.addWidget(self.choose, 0, 0, 1, 3)
        layout.addWidget(self.stacked, 1, 0, 1, 3)
        layout.addWidget(self.cancel_btn, 2, 0)
        layout.addWidget(self.process_btn, 2, 2)
        self.setLayout(layout)

    def process(self):
        values = self.stacked.currentWidget().get_values()
        algorithm = self.transform_dict[self.choose.currentText()]
        self.result_val = ImageAdjustTuple(values, algorithm)

        self.accept()
