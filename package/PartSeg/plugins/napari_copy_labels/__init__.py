from napari import Viewer
from napari.layers.labels import Labels
from PyQt5.QtGui import QKeySequence
from qtpy.QtWidgets import QGridLayout, QLabel, QPushButton, QShortcut, QSpinBox, QWidget


class CopyLabelWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer

        self.copy_btn = QPushButton("Copy")
        self.lower = QSpinBox()
        self.lower.setSingleStep(1)
        self.upper = QSpinBox()
        self.upper.setSingleStep(1)
        layout = QGridLayout()
        layout.addWidget(QLabel("Lower layer"), 0, 0)
        layout.addWidget(QLabel("Upper layer"), 1, 0)
        layout.addWidget(self.lower, 0, 1)
        layout.addWidget(self.upper, 1, 1)
        layout.addWidget(self.copy_btn, 3, 0, 1, 2)
        self.shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self.shortcut.activated.connect(self.copy_action)

        self.setLayout(layout)

        self.viewer.events.active_layer.connect(self.activate_widget)
        self.copy_btn.clicked.connect(self.copy_action)

    def activate_widget(self, event):
        self.setVisible(isinstance(event.item, Labels))

    def copy_action(self):
        layer = self.viewer.active_layer
        if layer is None:
            return

        component_num = layer.selected_label
        z_position = self.viewer.dims.current_step[1]
        print(layer, component_num, z_position)
        mask = layer.data[0, z_position] == component_num
        start = max(0, self.lower.value())
        end = min(layer.shape[1], self.upper.value()) + 1
        for i in range(start, end):
            layer.data[0, i][mask] = component_num
