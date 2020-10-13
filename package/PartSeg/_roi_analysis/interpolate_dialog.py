from qtpy.QtWidgets import QDialog, QDoubleSpinBox, QGridLayout, QLabel, QPushButton


class InterpolateDialog(QDialog):
    def __init__(self, spacing, *args, **kwargs):
        super().__init__(*args, **kwargs)
        min_val = min(spacing)
        start_value = [x / min_val for x in spacing]
        info_label = QLabel()
        info_label.setText("This operation cannot be undone,\nit also update image spacing")
        self.start_value = start_value
        self.spacing = spacing
        self.x_spacing = QDoubleSpinBox()
        self.x_spacing.setRange(0, 100)
        self.x_spacing.setSingleStep(0.1)
        self.x_spacing.setDecimals(2)
        self.x_spacing.setValue(start_value[-1])
        self.y_spacing = QDoubleSpinBox()
        self.y_spacing.setRange(0, 100)
        self.y_spacing.setSingleStep(0.1)
        self.y_spacing.setDecimals(2)
        self.y_spacing.setValue(start_value[-2])
        self.z_spacing = QDoubleSpinBox()
        self.z_spacing.setRange(0, 100)
        self.z_spacing.setSingleStep(0.1)
        self.z_spacing.setDecimals(2)
        if len(start_value) == 3:
            self.z_spacing.setValue(start_value[0])
        else:
            self.z_spacing.setDisabled(True)

        self.accept_button = QPushButton("Interpolate")
        self.accept_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        layout = QGridLayout()
        layout.addWidget(info_label, 0, 0, 1, 2)
        layout.addWidget(QLabel("x scale"), 1, 0)
        layout.addWidget(self.x_spacing, 1, 1)
        layout.addWidget(QLabel("y scale"), 2, 0)
        layout.addWidget(self.y_spacing, 2, 1)
        if len(start_value) == 3:
            layout.addWidget(QLabel("z scale"), 3, 0)
            layout.addWidget(self.z_spacing, 3, 1)
        layout.addWidget(self.accept_button, 4, 0)
        layout.addWidget(self.cancel_button, 4, 1)
        self.setLayout(layout)

    def get_zoom_factor(self):
        if len(self.start_value) == 3:
            return 1, self.z_spacing.value(), self.y_spacing.value(), self.x_spacing.value()

        return 1, self.y_spacing.value(), self.x_spacing.value()

    def get_new_spacing(self):
        if len(self.start_value) == 3:
            return [
                self.spacing[0] / self.x_spacing.value(),
                self.spacing[1] / self.y_spacing.value(),
                self.spacing[2] / self.z_spacing.value(),
            ]

        return [self.spacing[0] / self.x_spacing.value(), self.spacing[1] / self.y_spacing.value(), 1]
