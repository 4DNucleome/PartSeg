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
from superqt.utils import QSignalDebouncer

from PartSeg.common_gui.flow_layout import FlowLayout


class CopyLabelsWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.label = QLabel("Copy selected labels along z-axis")

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
        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.setRowStretch(0, 0)
        layout.addWidget(QLabel("Lower layer"), 1, 0)
        layout.addWidget(QLabel("Upper layer"), 2, 0)
        layout.addWidget(self.lower, 1, 1)
        layout.addWidget(self.upper, 2, 1)
        layout.addLayout(self.checkbox_layout, 3, 0, 1, 2)
        layout.setRowStretch(3, 1)
        layout.addWidget(self.copy_btn, 4, 0, 1, 2)
        layout.addWidget(self.check_all_btn, 5, 0)
        layout.addWidget(self.uncheck_all_btn, 5, 1)
        self.shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self.shortcut.activated.connect(self.copy_action)
        self.debouncer = QSignalDebouncer(parent=self)
        self.debouncer.setTimeout(200)
        self.debouncer.triggered.connect(self.update_items)

        self.setLayout(layout)

        self.viewer.layers.selection.events.active.connect(self.activate_widget)
        self.copy_btn.clicked.connect(self.copy_action)
        self.check_all_btn.clicked.connect(self._check_all)
        self.uncheck_all_btn.clicked.connect(self._uncheck_all)
        if isinstance(self.viewer.layers.selection.active, Labels):
            self._activate_widget(self.viewer.layers.selection.active)

    def activate_widget(self, event):
        self._activate_widget(event.value)

    def _activate_widget(self, label_layer: Labels):
        is_labels = isinstance(label_layer, Labels)
        if is_labels and not self.isVisible():
            self._update_items(label_layer)
        self.setVisible(is_labels)
        if is_labels:
            label_layer.events.set_data.connect(self._shallow_update)
            label_layer.events.selected_label.connect(self.debouncer.throttle)

    def _shallow_update(self, event):
        label_num = event.source.selected_label
        self._components.add(label_num)
        self.refresh()

    def update_items(self):
        self._update_items(self.viewer.layers.selection.active)

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
