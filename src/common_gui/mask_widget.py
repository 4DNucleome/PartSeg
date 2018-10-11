from PyQt5.QtWidgets import QWidget, QAbstractSpinBox, QCheckBox, QLabel, QHBoxLayout, QSpinBox, QVBoxLayout

from common_gui.dim_combobox import DimComboBox
from project_utils.algorithm_base import calculate_operation_radius
from project_utils.image_operations import RadiusType
from project_utils.settings import ImageSettings


class MaskWidget(QWidget):
    def __init__(self, settings: ImageSettings, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0,0,0,0)
        self.settings = settings
        self.dilate_radius = QSpinBox()
        self.dilate_radius.setRange(-100, 100)
        self.dilate_radius.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.dilate_radius.setSingleStep(1)

        self.dilate_dim = DimComboBox()
        self.radius_information = QLabel()

        # noinspection PyUnresolvedReferences
        self.dilate_radius.valueChanged.connect(self.dilate_change)
        # noinspection PyUnresolvedReferences
        self.dilate_dim.currentIndexChanged.connect(self.dilate_change)

        self.fill_holes = DimComboBox(self)
        self.max_hole_size = QSpinBox()
        self.max_hole_size.setRange(-1, 10000)
        self.max_hole_size.setValue(-1)
        self.max_hole_size.setSingleStep(100)
        layout = QVBoxLayout()
        layout1 = QHBoxLayout()
        layout1.addWidget(QLabel("Dilate radius (in pix)"))
        layout1.addWidget(self.dilate_radius)
        layout1.addWidget(self.dilate_dim)
        layout.addLayout(layout1)
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Fill holes:"))
        layout2.addWidget(self.fill_holes)
        layout2.addWidget(QLabel("Max hole size"))
        layout2.addWidget(self.max_hole_size)
        layout.addLayout(layout2)
        self.setLayout(layout)
        self.dilate_change()

    def get_dilate_radius(self):
        radius = calculate_operation_radius(self.dilate_radius.value(), self.settings.image_spacing,
                                            self.dilate_dim.value())
        if isinstance(radius, (list, tuple)):
            return list(map(int, radius))
        return int(radius)

    def dilate_change(self):
        if self.dilate_radius.value() == 0 or self.dilate_dim.value() == RadiusType.NO:
            self.radius_information.setText("Real radius: 0")
        else:
            dilate_radius = self.get_dilate_radius()
            if isinstance(dilate_radius, list):
                self.radius_information.setText(f"Real radius: {dilate_radius[::-1]}")
            else:
                self.radius_information.setText(f"Real radius: {dilate_radius}")
