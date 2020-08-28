from functools import partial

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from PartSegCore.image_operations import RadiusType
from PartSegCore.mask_create import MaskProperty
from PartSegCore.segmentation.algorithm_base import calculate_operation_radius

from ..common_backend.base_settings import BaseSettings, ImageSettings
from .dim_combobox import DimComboBox


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
        # self.dilate_radius.setButtonSymbols(QAbstractSpinBox.NoButtons)
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
        self.reversed_check = QCheckBox()

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
        self.reversed_check.stateChanged.connect(self.values_changed.emit)

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
        comp_lab.setToolTip(
            "save components information in mask. Dilation, " "holes filing will be done separately for each component"
        )
        self.save_components.setToolTip(comp_lab.toolTip())
        layout3.addWidget(comp_lab)
        layout3.addWidget(self.save_components)
        layout3.addStretch()
        clip_mask = QLabel("Clip to previous mask:")
        clip_mask.setToolTip("Useful dilated new mask")
        layout3.addWidget(clip_mask)
        layout3.addWidget(self.clip_to_mask)
        layout.addLayout(layout3)
        layout4 = QHBoxLayout()
        layout4.addWidget(QLabel("Reversed mask:"))
        layout4.addWidget(self.reversed_check)
        layout4.addStretch(1)
        layout.addLayout(layout4)
        self.setLayout(layout)
        self.dilate_change()

    def get_dilate_radius(self):
        radius = calculate_operation_radius(
            self.dilate_radius.value(), self.settings.image_spacing, self.dilate_dim.value()
        )
        if isinstance(radius, (list, tuple)):
            return [int(x + 0.5) for x in radius]
        return int(radius)

    def dilate_change(self):
        if self.dilate_radius.value() == 0 or self.dilate_dim.value() == RadiusType.NO:
            self.radius_information.setText("Dilation radius: 0")
        else:
            dilate_radius = self.get_dilate_radius()
            if isinstance(dilate_radius, list):
                self.radius_information.setText(f"Dilation radius: {dilate_radius[::-1]}")
            else:
                self.radius_information.setText(f"Dilation radius: {dilate_radius}")

    def get_mask_property(self):
        return MaskProperty(
            dilate=self.dilate_dim.value() if self.dilate_radius.value() != 0 else RadiusType.NO,
            dilate_radius=self.dilate_radius.value() if self.dilate_dim.value() != RadiusType.NO else 0,
            fill_holes=self.fill_holes.value() if self.max_hole_size.value() != 0 else RadiusType.NO,
            max_holes_size=self.max_hole_size.value() if self.fill_holes.value() != RadiusType.NO else 0,
            save_components=self.save_components.isChecked(),
            clip_to_mask=self.clip_to_mask.isChecked(),
            reversed_mask=self.reversed_check.isChecked(),
        )

    def set_mask_property(self, prop: MaskProperty):
        self.dilate_dim.setValue(prop.dilate)
        self.dilate_radius.setValue(prop.dilate_radius)
        self.fill_holes.setValue(prop.fill_holes)
        self.max_hole_size.setValue(prop.max_holes_size)
        self.save_components.setChecked(prop.save_components)
        self.clip_to_mask.setChecked(prop.clip_to_mask)
        self.reversed_check.setChecked(prop.reversed_mask)


class MaskDialogBase(QDialog):
    def __init__(self, settings: BaseSettings):
        super().__init__()
        self.setWindowTitle("Mask manager")
        self.settings = settings
        main_layout = QVBoxLayout()
        self.mask_widget = MaskWidget(settings, self)
        main_layout.addWidget(self.mask_widget)
        try:
            mask_property = self.settings.get("mask_manager.mask_property")
            self.mask_widget.set_mask_property(mask_property)
        except KeyError:
            pass

        self.reset_next_btn = QPushButton("Reset Next")
        self.reset_next_btn.clicked.connect(self.reset_next_fun)
        if settings.history_redo_size() == 0:
            self.reset_next_btn.setDisabled(True)
        self.set_next_btn = QPushButton("Set Next")
        if settings.history_redo_size() == 0:
            self.set_next_btn.setDisabled(True)
        self.set_next_btn.clicked.connect(self.set_next)
        self.cancel = QPushButton("Cancel", self)
        self.cancel.clicked.connect(self.close)
        self.prev_button = QPushButton(f"Previous mask ({settings.history_size()})", self)
        if settings.history_size() == 0:
            self.prev_button.setDisabled(True)
        self.next_button = QPushButton(f"Next mask ({settings.history_redo_size()})", self)
        if settings.history_redo_size() == 0:
            self.next_button.setText("Next mask (new)")
        self.next_button.clicked.connect(self.next_mask)
        self.prev_button.clicked.connect(self.prev_mask)
        op_layout = QHBoxLayout()
        op_layout.addWidget(self.mask_widget.radius_information)
        main_layout.addLayout(op_layout)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.cancel)
        button_layout.addWidget(self.set_next_btn)
        button_layout.addWidget(self.reset_next_btn)
        main_layout.addLayout(button_layout)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        if self.settings.history_redo_size():
            mask_prop: MaskProperty = self.settings.history_next_element().mask_property
            self.mask_widget.set_mask_property(mask_prop)
        self.mask_widget.values_changed.connect(self.values_changed)

    def set_next(self):
        if self.settings.history_redo_size():
            self.mask_widget.set_mask_property(self.settings.history_next_element().mask_property)

    def values_changed(self):
        if (
            self.settings.history_redo_size()
            and self.mask_widget.get_mask_property() == self.settings.history_next_element().mask_property
        ):
            self.next_button.setText(f"Next mask ({self.settings.history_redo_size()})")
        else:
            self.next_button.setText("Next mask (new)")

    def reset_next_fun(self):
        self.settings.history_redo_clean()
        self.next_button.setText("Next mask (new)")
        self.reset_next_btn.setDisabled(True)

    def next_mask(self):
        raise NotImplementedError()

    def prev_mask(self):
        raise NotImplementedError()
