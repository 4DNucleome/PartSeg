import numpy as np
from napari import Viewer
from napari.layers.labels import Labels
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QSpinBox,
    QWidget,
)

from PartSeg.common_gui.flow_layout import FlowLayout


class CopyLabelsWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.copy_btn = QPushButton("Copy")
        self.check_all_btn = QPushButton("Check all")
        self.uncheck_all_btn = QPushButton("Uncheck all")
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
        layout.addWidget(self.check_all_btn, 4, 0)
        layout.addWidget(self.uncheck_all_btn, 4, 1)
        self.shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self.shortcut.activated.connect(self.copy_action)

        self.setLayout(layout)

        self.viewer.layers.selection.events.active.connect(self.activate_widget)
        self.copy_btn.clicked.connect(self.copy_action)
        self.check_all_btn.clicked.connect(self._check_all)
        self.uncheck_all_btn.clicked.connect(self._uncheck_all)
        if isinstance(self.viewer.layers.selection.active, Labels):
            self._update_items(self.viewer.layers.selection.active)

    def activate_widget(self, event):
        is_labels = isinstance(event.value, Labels)
        if is_labels and not self.isVisible():
            self._update_items(event.value)
        self.setVisible(is_labels)
        if is_labels:
            event.value.events.set_data.connect(self._shallow_update)
            event.value.events.selected_label.connect(self.update_items)

    def _shallow_update(self, event):
        label = event.source.selected_label
        self.refresh()
        if label in self._components:
            return
        self._components.add(label)

    def update_items(self, event):
        self._update_items(event.source)

    def _update_items(self, layer):
        unique = np.unique(layer.data)
        if unique[0] == 0:
            unique = unique[1:]
        self._components = set(unique)
        self.refresh()

    def _check_all(self):
        for i in range(self.checkbox_layout.count()):
            w: QCheckBox = self.checkbox_layout.itemAt(i).widget()
            w.setChecked(True)

    def _uncheck_all(self):
        for i in range(self.checkbox_layout.count()):
            w: QCheckBox = self.checkbox_layout.itemAt(i).widget()
            w.setChecked(False)

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
        layer = self.viewer.layers.selection.active
        if layer is None:
            return

        leading_zeros = (0,) * (layer.data.ndim - 3)
        checked = set()
        for i in range(self.checkbox_layout.count()):
            w: QCheckBox = self.checkbox_layout.itemAt(i).widget()
            if w.isChecked():
                checked.add(int(w.text()))
        if not checked:
            checked = {layer.selected_label}
        z_position = self.viewer.dims.current_step[layer.dtype.ndim - 3]
        for component_num in checked:
            mask = layer.data[(*leading_zeros, z_position)] == component_num
            start = max(0, self.lower.value())
            end = min(layer.data.shape[1] - 1, self.upper.value()) + 1
            for i in range(start, end):
                layer.data[(*leading_zeros, i)][mask] = component_num
