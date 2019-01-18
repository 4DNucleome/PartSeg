from functools import partial

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget, QAbstractSpinBox, QCheckBox, QLabel, QHBoxLayout, QSpinBox, QVBoxLayout
from .dim_combobox import DimComboBox
from ..utils.segmentation.algorithm_base import calculate_operation_radius
from ..utils.image_operations import RadiusType
from ..project_utils_qt.settings import ImageSettings
from ..utils.mask_create import MaskProperty


def off_widget(widget: QWidget, combo_box: DimComboBox):
    widget.setDisabled(combo_box.value() == RadiusType.NO)


class MaskWidget(QWidget):
    values_changed = Signal()

    def __init__(self, settings: ImageSettings, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.settings = settings
        self.dilate_radius = QSpinBox()
        self.dilate_radius.setRange(-100, 100)
        self.dilate_radius.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.dilate_radius.setSingleStep(1)
        self.dilate_radius.setDisabled(True)
        self.dilate_dim = DimComboBox()
        self.dilate_dim.setToolTip("With minus radius mask will be eroded")
        # noinspection PyUnresolvedReferences
        self.dilate_dim.currentIndexChanged.connect(partial(off_widget, self.dilate_radius, self.dilate_dim))

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
        self.max_hole_size.setDisabled(True)
        self.max_hole_size.setToolTip("Maximum size of holes to be closed. -1 means that all holes will be closed")
        # noinspection PyUnresolvedReferences
        self.fill_holes.currentIndexChanged.connect(partial(off_widget, self.max_hole_size, self.fill_holes))

        self.save_components = QCheckBox()
        self.clip_to_mask = QCheckBox()

        # noinspection PyUnresolvedReferences
        self.dilate_radius.valueChanged.connect(self.values_changed.emit)
        # noinspection PyUnresolvedReferences
        self.dilate_dim.currentIndexChanged.connect(self.values_changed.emit)
        # noinspection PyUnresolvedReferences
        self.fill_holes.currentIndexChanged.connect(self.values_changed.emit)
        # noinspection PyUnresolvedReferences
        self.max_hole_size.valueChanged.connect(self.values_changed.emit)
        self.save_components.stateChanged.connect(self.values_changed.emit)
        self.clip_to_mask.stateChanged.connect(self.values_changed.emit)

        layout = QVBoxLayout()
        layout1 = QHBoxLayout()
        layout1.addWidget(QLabel("Dilate mask:"))
        layout1.addWidget(self.dilate_dim)
        layout1.addWidget(QLabel("radius (in pix):"))
        layout1.addWidget(self.dilate_radius)
        layout.addLayout(layout1)
        layout2 = QHBoxLayout()
        layout2.addWidget(QLabel("Fill holes:"))
        layout2.addWidget(self.fill_holes)
        layout2.addWidget(QLabel("Max size:"))
        layout2.addWidget(self.max_hole_size)
        layout.addLayout(layout2)
        layout3 = QHBoxLayout()
        comp_lab = QLabel("Save components:")
        comp_lab.setToolTip("save components information in mask. Dilation, "
                            "holes filing will be done separately for each component")
        self.save_components.setToolTip(comp_lab.toolTip())
        layout3.addWidget(comp_lab)
        layout3.addWidget(self.save_components)
        layout3.addStretch()
        clip_mask = QLabel("Clip to upper mask:")
        layout3.addWidget(clip_mask)
        layout3.addWidget(self.clip_to_mask)
        layout.addLayout(layout3)
        self.setLayout(layout)
        self.dilate_change()

    def get_dilate_radius(self):
        radius = calculate_operation_radius(self.dilate_radius.value(), self.settings.image_spacing,
                                            self.dilate_dim.value())
        if isinstance(radius, (list, tuple)):
            return [int(x + 0.5) for x in radius]
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

    def get_mask_property(self):
        return \
            MaskProperty(dilate=self.dilate_dim.value() if self.dilate_radius.value() != 0 else RadiusType.NO,
                         dilate_radius=self.dilate_radius.value() if self.dilate_dim.value() != RadiusType.NO else 0,
                         fill_holes=self.fill_holes.value() if self.max_hole_size.value() != 0 else RadiusType.NO,
                         max_holes_size=self.max_hole_size.value() if self.fill_holes.value() != RadiusType.NO else 0,
                         save_components=self.save_components.isChecked(),
                         clip_to_mask=self.clip_to_mask.isChecked())

    def set_mask_property(self, property: MaskProperty):
        self.dilate_dim.setValue(property.dilate)
        self.dilate_radius.setValue(property.dilate_radius)
        self.fill_holes.setValue(property.fill_holes)
        self.max_hole_size.setValue(property.max_holes_size)
        self.save_components.setChecked(property.save_components)
        self.clip_to_mask.setChecked(property.clip_to_mask)
