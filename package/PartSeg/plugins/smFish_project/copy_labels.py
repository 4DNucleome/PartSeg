import numpy as np
from napari import Viewer
from napari.layers.labels import Labels
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QCheckBox, QGridLayout, QLabel, QPushButton, QShortcut, QSpinBox, QWidget

from PartSeg.common_gui.flow_layout import FlowLayout


class CopyLabelWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.copy_btn = QPushButton("Copy")
        self.lower = QSpinBox()
        self.lower.setSingleStep(1)
        self.upper = QSpinBox()
        self.upper.setSingleStep(1)
        self.checkbox_layout = FlowLayout()
        self._components = set()

        layout = QGridLayout()
        layout.addWidget(QLabel("Lower layer"), 0, 0)
        layout.addWidget(QLabel("Upper layer"), 1, 0)
        layout.addWidget(self.lower, 0, 1)
        layout.addWidget(self.upper, 1, 1)
        layout.addLayout(self.checkbox_layout, 2, 0, 1, 2)
        layout.addWidget(self.copy_btn, 3, 0, 1, 2)
        self.shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self.shortcut.activated.connect(self.copy_action)

        self.setLayout(layout)

        self.viewer.events.active_layer.connect(self.activate_widget)
        self.copy_btn.clicked.connect(self.copy_action)

    def activate_widget(self, event):
        self.setVisible(isinstance(event.value, Labels))
        if isinstance(event.value, Labels):
            event.value.events.set_data.connect(self._shallow_update)
            event.value.events.selected_label.connect(self.update_items)

    def _shallow_update(self, event):
        label = event.source.selected_label
        if label in self._components:
            return
        self._components.add(label)
        self.refresh()

    def update_items(self, event):
        unique = np.unique(event.source.data)
        if unique[0] == 0:
            unique = unique[1:]
        self._components = set(unique)
        self.refresh()

    def refresh(self):
        checked = set()
        for _ in range(self.checkbox_layout.count()):
            w: QCheckBox = self.checkbox_layout.takeAt(0).widget()
            if w.isChecked():
                checked.add(w.text())
            w.deleteLater()
        for v in self._components:
            chk = QCheckBox(str(v))
            if chk.text() in checked:
                chk.setChecked(True)
            self.checkbox_layout.addWidget(chk)

    def copy_action(self):
        layer = self.viewer.active_layer
        if layer is None:
            return

        checked = set()
        for i in range(self.checkbox_layout.count()):
            w: QCheckBox = self.checkbox_layout.itemAt(i).widget()
            if w.isChecked():
                checked.add(int(w.text()))
        if not checked:
            checked = {layer.selected_label}
        z_position = self.viewer.dims.current_step[1]
        for component_num in checked:
            mask = layer.data[0, z_position] == component_num
            start = max(0, self.lower.value())
            end = min(layer.shape[1], self.upper.value()) + 1
            for i in range(start, end):
                layer.data[0, i][mask] = component_num
